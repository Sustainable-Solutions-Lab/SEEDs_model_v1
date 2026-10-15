"""
Simplified supply-demand equilibrium model: linear elasticities, regional aggregation, full emissions and constraints.
================================================================

Differences from the original S3_0_ds_emis_mc_full.py:
1. Replace PWL constraints with linear elasticity equations to avoid PWL-related solve failures.
2. Aggregate 194 countries into 34 regions using dict_v3 Region_market_agg.
3. Simplify market clearing to the global level only.

Full functionality, consistent with S3_0_ds_emis_mc_full.py:
- Emissions as e0_by_proc * Qs with abatement decisions per process driven by MACC
- Land carbon price objective term for LULUCF processes
- Optional nutrition and land constraints
- Monte Carlo simulation support (LinearModelCache, apply_linear_sample_updates, run_linear_mc)
- Scenario support (tax_unit, feed_reduction, ruminant_intake_cap)
- Supply/demand slack variables for robustness

Objectives:
- Solve quickly (seconds rather than hours).
- Preserve full emissions and abatement functionality.
- Support nutrition and land constraints.
- Support Monte Carlo simulation and scenario analysis.
"""

from __future__ import annotations
from feed_product_accounts import bounded_feed_credit
from armington_ces import (attach_ces_sources, update_ces_approximation,
                           prepare_quantity_projection, recover_projected_ces, ces_value)
import json
import copy
import math
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Mapping, Sequence
from dataclasses import dataclass
from collections import defaultdict

import gurobipy as gp
import pandas as pd
import numpy as np
import re
from market_balance_diagnostics import build_market_balance_diagnostics
from model_run_status import normalize_solver_status, solver_result_is_extractable
from runtime_data_cache import read_tabular_cached

from S3_5_land_use_change import LUCConfig

TC2CO2 = 44.0 / 12.0
LOSS_RATIO_EPS = 1e-9
KT_CO2E_TO_T_CO2E = 1000.0

# Cost-database v2 layers.  Four system strategies change exogenous scenario
# parameters; five process strategies price process-specific abatement.  A
# solve may own at most one explicit strategy cost, otherwise the same tonne of
# abatement would be charged more than once.
V2_SYSTEM_STRATEGY_COST_KEYS: Tuple[str, ...] = (
    "RuminantReduction",
    "LossWaste",
    "YieldRate",
    "FeedEfficiency",
)
V2_PROCESS_COST_KEYS: Tuple[str, ...] = (
    "EntericF",
    "Manure",
    "Residue",
    "Rice",
    "Fertilizer",
)
V2_ALL_COST_KEYS = V2_SYSTEM_STRATEGY_COST_KEYS + V2_PROCESS_COST_KEYS


def _resolve_unit_cost_attribution(
    active_strategy_cost_keys: Optional[Sequence[str]],
) -> Tuple[str, Tuple[str, ...]]:
    """Return ``(mode, keys)`` for a v2 unit-cost solve.

    ``None`` preserves the historical behaviour of pricing every process
    component.  An explicit empty sequence disables mitigation-cost ownership
    (used for coalition runs).  A singleton selects exactly one system or
    process strategy.  Multiple explicit owners are rejected because their
    abatement quantities overlap and are not identifiable inside one solve.
    """

    if active_strategy_cost_keys is None:
        return "process", V2_PROCESS_COST_KEYS
    keys = tuple(dict.fromkeys(str(key).strip() for key in active_strategy_cost_keys if str(key).strip()))
    unknown = sorted(set(keys) - set(V2_ALL_COST_KEYS))
    if unknown:
        raise ValueError(f"Unknown v2 cost strategy key(s): {unknown}")
    if not keys:
        return "none", ()
    if len(keys) != 1:
        raise ValueError(
            "Exactly one active_strategy_cost_key is allowed per solve; "
            "use singleton/Shapley attribution for multi-strategy scenarios"
        )
    key = keys[0]
    return (
        "strategy" if key in V2_SYSTEM_STRATEGY_COST_KEYS else "process",
        keys,
    )


def _cost_region_is_selected(region: Any, selected_regions: Optional[Sequence[str]]) -> bool:
    if selected_regions is None:
        return True
    aliases = {str(value).strip() for value in selected_regions if str(value).strip()}
    if not aliases:
        return False
    candidate = str(region).strip()
    if candidate in aliases:
        return True
    digits = "".join(ch for ch in candidate if ch.isdigit())
    candidate_m49 = f"'{digits.zfill(3)[-3:]}" if digits else ""
    alias_m49 = set()
    for value in aliases:
        value_digits = "".join(ch for ch in value if ch.isdigit())
        if value_digits:
            alias_m49.add(f"'{value_digits.zfill(3)[-3:]}")
    return bool(candidate_m49 and candidate_m49 in alias_m49)


def _macc_cost_coefficient_usd_per_kt(
    marginal_cost_usd_per_tco2e: Any,
) -> float:
    """Convert a MACC marginal cost from USD/tCO2e to USD/ktCO2e."""
    marginal_cost = float(marginal_cost_usd_per_tco2e)
    if not np.isfinite(marginal_cost):
        raise ValueError("MACC marginal cost must be finite")
    return KT_CO2E_TO_T_CO2E * marginal_cost


def _resolve_production_cost_weight(
    disable_production_cost_term: bool,
    production_cost_weight: Any,
) -> float:
    """Resolve the production-cost weight without demand-mode overrides."""
    if bool(disable_production_cost_term):
        return 0.0
    weight = float(production_cost_weight or 0.0)
    if not np.isfinite(weight) or weight <= 0.0:
        return 0.0
    return weight


def _build_production_cost_objective(
    *,
    qs: Mapping[Tuple[str, str, int], gp.Var],
    p0_by_key: Mapping[Tuple[str, str, int], float],
    hist_end_year: int,
    qty_scale: float,
    disable_production_cost_term: bool,
    production_cost_weight: Any,
    tax_unit_adder: Optional[Mapping[Tuple[str, str, int], float]] = None,
) -> Tuple[gp.LinExpr, float]:
    """Build the optional production-cost proxy and return its effective weight."""
    effective_weight = _resolve_production_cost_weight(
        disable_production_cost_term,
        production_cost_weight,
    )
    objective = gp.LinExpr(0.0)
    if effective_weight <= 0.0:
        return objective, effective_weight

    for key, qs_var in qs.items():
        _region, _commodity, year = key
        if year <= hist_end_year:
            continue
        unit_cost_val = float(p0_by_key.get(key, 0.0) or 0.0)
        if tax_unit_adder:
            try:
                unit_cost_val += float(tax_unit_adder.get(key, 0.0) or 0.0)
            except Exception:
                pass
        if unit_cost_val > 0.0:
            objective += effective_weight * unit_cost_val * qty_scale * qs_var
    return objective, effective_weight


def _clip_loss_ratio_delta(delta: Any) -> float:
    """Clamp additive losses-ratio deltas to the supported [-1, 1] range."""
    try:
        val = float(delta or 0.0)
    except Exception:
        return 0.0
    if not np.isfinite(val):
        return 0.0
    return max(-1.0, min(1.0, val))


def _final_loss_ratio_from_delta(baseline_loss_ratio: Any, delta: Any) -> float:
    """Return final waste/loss rate = clip(baseline + delta, [0, 1))."""
    try:
        base = float(baseline_loss_ratio or 0.0)
    except Exception:
        base = 0.0
    if not np.isfinite(base):
        base = 0.0
    base = max(0.0, min(1.0 - LOSS_RATIO_EPS, base))
    return max(0.0, min(1.0 - LOSS_RATIO_EPS, base + _clip_loss_ratio_delta(delta)))


def _loss_multiplier_from_delta(baseline_loss_ratio: Any, delta: Any) -> float:
    """D0_new = D0 * (1 + final_loss_ratio); never reduces direct D0."""
    return 1.0 + _final_loss_ratio_from_delta(baseline_loss_ratio, delta)


def _lookup_loss_delta(
    *,
    key: Tuple[str, str, int],
    all_key: Tuple[str, str, int],
    waste_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    losses_ratio_by: Optional[Dict[Tuple[str, str, int], float]] = None,
) -> Tuple[Optional[float], str]:
    """Prefer explicit losses_ratio deltas; fall back to legacy waste_reduction keys."""
    for source, mapping in (("losses_ratio_by", losses_ratio_by), ("waste_reduction_by", waste_reduction_by)):
        if not mapping:
            continue
        val = mapping.get(key)
        if val is None:
            val = mapping.get(all_key)
        if val is None:
            continue
        try:
            val_f = float(val)
        except Exception:
            continue
        if np.isfinite(val_f):
            return _clip_loss_ratio_delta(val_f), source
    return None, ""


def _require_positive_yield(
    yield_val: Any,
    *,
    region: Optional[str] = None,
    commodity: Optional[str] = None,
    year: Optional[int] = None,
    context: str = "",
) -> float:
    try:
        y = float(yield_val)
    except Exception as exc:
        raise ValueError(
            f"Invalid yield value {yield_val!r} in {context} "
            f"(region={region}, commodity={commodity}, year={year})"
        ) from exc
    if year == 2020:
        if not np.isfinite(y) or y <= 0:
            raise ValueError(
                f"Nonpositive yield {y!r} in {context} "
                f"(region={region}, commodity={commodity}, year={year})"
            )
    return y


def _cropland_yield0(
    node_data: Optional[Dict[str, Any]],
    *,
    region: Optional[str] = None,
    commodity: Optional[str] = None,
    year: Optional[int] = None,
    context: str = "",
) -> float:
    if not node_data:
        raise ValueError(
            f"Missing node data for yield0 in {context} "
            f"(region={region}, commodity={commodity}, year={year})"
        )
    if node_data.get('livestock_feed_account', False):
        return float('inf')
    if 'yield0' not in node_data:
        raise ValueError(
            f"Missing yield0 key in {context} "
            f"(region={region}, commodity={commodity}, year={year})"
        )
    check_year = 2020
    return _require_positive_yield(
        node_data.get('yield0'),
        region=region,
        commodity=commodity,
        year=check_year,
        context=context,
    )


def _parse_baseline_year(val: Any) -> Optional[int]:
    try:
        y = int(val)
    except Exception:
        return None
    return y if y > 0 else None


def _unpack_baseline_emission_key(
    key: Any,
) -> Optional[Tuple[Any, Any, int, Any]]:
    try:
        a, b, c, d = key
    except Exception:
        return None
    year_c = _parse_baseline_year(c)
    if year_c is not None:
        return a, b, year_c, d
    year_d = _parse_baseline_year(d)
    if year_d is not None:
        return a, c, year_d, b
    return None


def _normalize_trade_cap_ratio_arg(
    trade_cap_ratio: Any,
) -> Tuple[Optional[float], Dict[Tuple[str, str], float]]:
    """Return (default_ratio, ratio_by_(cap_key,item)) for trade caps."""
    ratio_by_key: Dict[Tuple[str, str], float] = {}
    if trade_cap_ratio is None:
        return None, ratio_by_key
    if isinstance(trade_cap_ratio, dict):
        default_ratio = None
        for raw_key, raw_val in trade_cap_ratio.items():
            try:
                val = float(raw_val)
            except Exception:
                continue
            if not np.isfinite(val) or val < 0:
                continue
            if isinstance(raw_key, tuple) and len(raw_key) == 2:
                cap_key, item = raw_key
                cap_key_s = str(cap_key).strip()
                item_s = str(item).strip()
                if cap_key_s and item_s:
                    ratio_by_key[(cap_key_s, item_s)] = val
            elif str(raw_key).strip().lower() in {'default', '*', 'all'}:
                default_ratio = val
        return default_ratio, ratio_by_key
    try:
        val = float(trade_cap_ratio)
    except Exception:
        return None, ratio_by_key
    if not np.isfinite(val) or val < 0:
        return None, ratio_by_key
    return val, ratio_by_key


def _lookup_trade_cap_ratio(
    cap_key: str,
    item: str,
    default_ratio: Optional[float],
    ratio_by_key: Dict[Tuple[str, str], float],
) -> Optional[float]:
    cap_key_s = str(cap_key).strip()
    item_s = str(item).strip()
    for key in (
        (cap_key_s, item_s),
        (cap_key_s, '*'),
        ('*', item_s),
        ('*', '*'),
    ):
        val = ratio_by_key.get(key)
        if val is not None:
            return float(val)
    return default_ratio


def _normalize_baseline_emissions(
    raw: Any,
    *,
    dict_v3_path: Optional[str],
) -> Dict[Tuple[str, str, int, str], float]:
    if not raw or not isinstance(raw, dict):
        return {}
    m49_to_country = _load_m49_to_country(dict_v3_path)
    country_to_m49 = {v: k for k, v in (m49_to_country or {}).items() if v}
    use_regional = is_region_aggregation_enabled()
    if use_regional and not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    out: Dict[Tuple[str, str, int, str], float] = {}
    for key, val in raw.items():
        parsed = _unpack_baseline_emission_key(key)
        if not parsed:
            continue
        region_raw, comm_raw, year_val, proc_raw = parsed
        region = str(region_raw).strip()
        comm = str(comm_raw).strip()
        proc = str(proc_raw).strip()
        if not region or not proc:
            continue
        try:
            val_f = float(val)
        except Exception:
            continue
        if not np.isfinite(val_f) or val_f <= 0:
            continue
        targets = {region}
        m49_norm = _norm_m49_code(region)
        if m49_norm:
            targets.add(m49_norm)
            country_name = m49_to_country.get(m49_norm)
            if country_name:
                targets.add(country_name)
        m49_alias = country_to_m49.get(region)
        if m49_alias:
            targets.add(m49_alias)
        if use_regional:
            region_key = get_region(region, m49=m49_norm, dict_v3_path=dict_v3_path)
            if region_key and region_key != 'OTHER':
                targets.add(region_key)
        for r_key in targets:
            out[(str(r_key), comm, year_val, proc)] = out.get((str(r_key), comm, year_val, proc), 0.0) + val_f
    return out



# LUC iterative loop


def _reconcile_base_land_stock(
    base_cropland_area: Dict[str, float],
    base_grassland_area: Dict[str, float],
    base_forest_area: Dict[str, float],
    base_cropland_demand: Dict[str, float],
    base_grassland_demand: Dict[str, float],
    *,
    logger: Optional[logging.Logger] = None,
    context: str = "base_land",
    tol: float = 1e-6,
    log_unmet: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Reconcile base land stock with base production-implied land demand.

    Multi-future rolling solves can fail when base cropland/grassland stock is
    materially below Q0/yield-implied land demand. Reconcile the base stock by
    moving surplus cropland/grassland first, then forest, while conserving
    total land.
    """
    crop_area = {str(k): float(v or 0.0) for k, v in (base_cropland_area or {}).items()}
    grass_area = {str(k): float(v or 0.0) for k, v in (base_grassland_area or {}).items()}
    forest_area = {str(k): float(v or 0.0) for k, v in (base_forest_area or {}).items()}

    adjusted = []
    unmet = []
    all_regions = (
        set(crop_area.keys())
        | set(grass_area.keys())
        | set(forest_area.keys())
        | {str(k) for k in (base_cropland_demand or {}).keys()}
        | {str(k) for k in (base_grassland_demand or {}).keys()}
    )

    for r in all_regions:
        crop = float(crop_area.get(r, 0.0) or 0.0)
        grass = float(grass_area.get(r, 0.0) or 0.0)
        forest = float(forest_area.get(r, 0.0) or 0.0)
        crop_need = max(float((base_cropland_demand or {}).get(r, 0.0) or 0.0), 0.0)
        grass_need = max(float((base_grassland_demand or {}).get(r, 0.0) or 0.0), 0.0)
        total_before = crop + grass + forest

        moved_g2c = 0.0
        moved_f2c = 0.0
        moved_c2g = 0.0
        moved_f2g = 0.0

        crop_deficit = max(crop_need - crop, 0.0)
        if crop_deficit > tol:
            grass_surplus = max(grass - grass_need, 0.0)
            moved_g2c = min(crop_deficit, grass_surplus)
            crop += moved_g2c
            grass -= moved_g2c
            crop_deficit = max(crop_need - crop, 0.0)
            if crop_deficit > tol:
                moved_f2c = min(crop_deficit, forest)
                crop += moved_f2c
                forest -= moved_f2c

        grass_deficit = max(grass_need - grass, 0.0)
        if grass_deficit > tol:
            crop_surplus = max(crop - crop_need, 0.0)
            moved_c2g = min(grass_deficit, crop_surplus)
            grass += moved_c2g
            crop -= moved_c2g
            grass_deficit = max(grass_need - grass, 0.0)
            if grass_deficit > tol:
                moved_f2g = min(grass_deficit, forest)
                grass += moved_f2g
                forest -= moved_f2g

        crop = max(crop, 0.0)
        grass = max(grass, 0.0)
        forest = max(forest, 0.0)

        total_after = crop + grass + forest
        if abs(total_after - total_before) > 1e-4 and logger is not None:
            logger.warning(
                "[LINEAR] %s land reconcile total drift: region=%s before=%.6e after=%.6e",
                context,
                r,
                total_before,
                total_after,
            )

        crop_area[r] = crop
        grass_area[r] = grass
        forest_area[r] = forest

        if any(abs(v) > tol for v in (moved_g2c, moved_f2c, moved_c2g, moved_f2g)):
            adjusted.append(
                (
                    r,
                    crop,
                    grass,
                    forest,
                    crop_need,
                    grass_need,
                    moved_g2c,
                    moved_f2c,
                    moved_c2g,
                    moved_f2g,
                )
            )

        rem_crop = max(crop_need - crop, 0.0)
        rem_grass = max(grass_need - grass, 0.0)
        if rem_crop > 1e-3 or rem_grass > 1e-3:
            unmet.append((r, rem_crop, rem_grass, total_before, crop_need, grass_need))

    if logger is not None and adjusted:
        logger.info(
            "[LINEAR] %s land reconcile active: adjusted_regions=%d",
            context,
            len(adjusted),
        )
        for sample in adjusted[:10]:
            r, crop, grass, forest, crop_need, grass_need, moved_g2c, moved_f2c, moved_c2g, moved_f2g = sample
            logger.info(
                "[LINEAR] %s land reconcile sample region=%s crop=%.6e grass=%.6e forest=%.6e "
                "(crop_need=%.6e grass_need=%.6e g2c=%.6e f2c=%.6e c2g=%.6e f2g=%.6e)",
                context,
                r,
                crop,
                grass,
                forest,
                crop_need,
                grass_need,
                moved_g2c,
                moved_f2c,
                moved_c2g,
                moved_f2g,
            )
    if logger is not None and unmet and log_unmet:
        logger.warning(
            "[LINEAR] %s land reconcile unmet demand remains in %d regions",
            context,
            len(unmet),
        )
        for sample in unmet[:10]:
            r, rem_crop, rem_grass, total_before, crop_need, grass_need = sample
            logger.warning(
                "[LINEAR] %s land reconcile unmet sample region=%s rem_crop=%.6e rem_grass=%.6e "
                "(total_before=%.6e crop_need=%.6e grass_need=%.6e)",
                context,
                r,
                rem_crop,
                rem_grass,
                total_before,
                crop_need,
                grass_need,
            )

    return crop_area, grass_area, forest_area

def _compute_luc_emissions_from_solution(
    qs: Dict[Tuple[str, str, int], float],
    idx: Dict[Tuple[str, str, int], Dict[str, Any]],
    years: List[int],
    hist_end_year: int,
    yield_t_per_ha_default: float,
    grassland_method: str,
    grass_area_by_region_year: Optional[Dict[Tuple[str, int], float]],
    luc_params: Optional[Dict[str, Any]],
    output_dir: Optional[str] = None,
    grassland_to_cropland: Optional[Dict[Tuple[str, int], float]] = None,
    grassland_to_forest: Optional[Dict[Tuple[str, int], float]] = None,
    cropland_to_grassland: Optional[Dict[Tuple[str, int], float]] = None,
    cropland_to_forest: Optional[Dict[Tuple[str, int], float]] = None,
    forest_to_cropland: Optional[Dict[Tuple[str, int], float]] = None,
    forest_to_grassland: Optional[Dict[Tuple[str, int], float]] = None,
    base_cropland_area: Optional[Dict[str, float]] = None,
    base_grassland_area: Optional[Dict[str, float]] = None,
    luc_shift_area_mode: str = 'abs',
    feed_background_pasture: Optional[Dict[Tuple[str, int], float]] = None,
) -> Tuple[
    Dict[Tuple[str, int], float],
    Dict[Tuple[str, int], float],
    Dict[Tuple[str, int], float],
    Dict[Tuple[str, int], float],
]:
    if not qs or not luc_params:
        return {}, {}, {}, {}

    logger = logging.getLogger(__name__)
    negative_rows: List[Dict[str, Any]] = []
    cfg = _extract_luc_params(luc_params)
    use_exp = bool(cfg.get('use_exponential_response', True))
    tau_veg = float(cfg.get('tau_veg', 20.0))
    tau_soil = float(cfg.get('tau_soil', 20.0))
    a_veg = 1.0 - math.exp(-1.0 / max(tau_veg, 1e-6)) if use_exp else 1.0
    a_soil = 1.0 - math.exp(-1.0 / max(tau_soil, 1e-6)) if use_exp else 1.0

    forest_c_ha = float(cfg.get('forest_c_ha', 150.0))
    cropland_c_ha = float(cfg.get('cropland_c_ha', 5.0))
    pasture_c_ha = float(cfg.get('pasture_c_ha', 10.0))
    forest_soil_c_ha = float(cfg.get('forest_soil_c_ha', 80.0))
    cropland_soil_c_ha = float(cfg.get('cropland_soil_c_ha', 50.0))
    pasture_soil_c_ha = float(cfg.get('pasture_soil_c_ha', 70.0))

    enable_shift = bool(cfg.get('enable_shift', True))
    tau_shift = float(cfg.get('tau_shift', 15.0))
    harvest_intensity = float(cfg.get('harvest_intensity', 1.0))
    if use_exp and tau_shift > 0:
        shift_veg_frac = 1.0 - math.exp(-tau_shift / max(tau_veg, 1e-6))
        shift_soil_frac = 1.0 - math.exp(-tau_shift / max(tau_soil, 1e-6))
    else:
        shift_veg_frac = 1.0
        shift_soil_frac = 1.0

    shift_area_mode = str(luc_shift_area_mode or 'abs').strip().lower()
    if shift_area_mode not in {'abs', 'delta_pos'}:
        shift_area_mode = 'abs'

    crop_demand = defaultdict(float)
    grass_demand = defaultdict(float)
    for (r, j, t), val in qs.items():
        if j == "Fish, Seafood":
            continue
        node_data = idx.get((r, j, t)) if idx else None
        yield_j = _cropland_yield0(
            node_data,
            region=r,
            commodity=j,
            year=t,
            context="luc_crop_area",
        )
        qs_raw = float(val or 0.0)
        coef_grass = 0.0
        if grassland_method == 'dynamic' and node_data is not None:
            coef_grass = float(node_data.get('grassland_coef', 0.0) or 0.0)
        if qs_raw < 0:
            negative_rows.append({
                'region': r,
                'commodity': j,
                'year': int(t),
                'issue': 'negative_qs',
                'qs_raw': qs_raw,
                'crop_need_raw_ha': qs_raw / yield_j,
                'grass_need_raw_ha': (coef_grass * qs_raw) if grassland_method == 'dynamic' else 0.0,
                'yield_used': yield_j,
                'grassland_coef': coef_grass,
            })
            qs_raw = 0.0

        crop_demand[(r, t)] += qs_raw / yield_j

        if grassland_method == 'dynamic':
            if coef_grass > 0:
                grass_demand[(r, t)] += coef_grass * qs_raw

    if grassland_method == 'dynamic':
        for key, area in (feed_background_pasture or {}).items():
            grass_demand[key] += float(area)

    if grassland_method == 'static' and grass_area_by_region_year:
        for key, val in grass_area_by_region_year.items():
            grass_demand[key] = float(val or 0.0)

    base_crop_demand = defaultdict(float)
    base_grass_demand = defaultdict(float)
    for (r, j, t), node_data in idx.items():
        if t != hist_end_year:
            continue
        if j == "Fish, Seafood":
            continue
        q0_val = float(node_data.get('Q0', 0.0) or 0.0)
        if q0_val <= 0:
            continue
        yield_j = _cropland_yield0(
            node_data,
            region=r,
            commodity=j,
            year=t,
            context="luc_base_demand",
        )
        if q0_val < 0:
            coef_grass = float(node_data.get('grassland_coef', 0.0) or 0.0)
            negative_rows.append({
                'region': r,
                'commodity': j,
                'year': int(t),
                'issue': 'negative_q0',
                'qs_raw': q0_val,
                'crop_need_raw_ha': q0_val / yield_j,
                'grass_need_raw_ha': coef_grass * q0_val,
                'yield_used': yield_j,
                'grassland_coef': coef_grass,
            })
            q0_val = 0.0
        base_crop_demand[r] += q0_val / yield_j
        coef_grass = float(node_data.get('grassland_coef', 0.0) or 0.0)
        if coef_grass > 0:
            base_grass_demand[r] += coef_grass * q0_val

    if grassland_method == 'dynamic':
        for (r,t),area in (feed_background_pasture or {}).items():
            if int(t)==hist_end_year:
                base_grass_demand[r] += float(area)
    if grassland_method == 'static' and grass_area_by_region_year:
        for (r, t), val in grass_area_by_region_year.items():
            if int(t) == hist_end_year:
                base_grass_demand[r] = float(val or 0.0)

    crop_area = defaultdict(float)
    grass_area = defaultdict(float)
    base_cropland_area = base_cropland_area or {}
    base_grassland_area = base_grassland_area or {}
    conv_crop_to_grass = cropland_to_grassland or {}
    conv_crop_to_forest = cropland_to_forest or {}
    conv_grass_to_crop = grassland_to_cropland or {}
    conv_grass_to_forest = grassland_to_forest or {}

    all_keys = set(crop_demand.keys()) | set(grass_demand.keys())
    all_keys |= set(conv_crop_to_grass.keys()) | set(conv_crop_to_forest.keys())
    all_keys |= set(conv_grass_to_crop.keys()) | set(conv_grass_to_forest.keys())

    for (r, t) in all_keys:
        crop_val = float(crop_demand.get((r, t), 0.0) or 0.0)
        grass_val = float(grass_demand.get((r, t), 0.0) or 0.0)
        crop_val -= float(conv_crop_to_grass.get((r, t), 0.0) or 0.0)
        crop_val -= float(conv_crop_to_forest.get((r, t), 0.0) or 0.0)
        grass_val -= float(conv_grass_to_crop.get((r, t), 0.0) or 0.0)
        grass_val -= float(conv_grass_to_forest.get((r, t), 0.0) or 0.0)

        offset_crop = float(base_cropland_area.get(r, 0.0) or 0.0) - float(base_crop_demand.get(r, 0.0) or 0.0)
        offset_grass = float(base_grassland_area.get(r, 0.0) or 0.0) - float(base_grass_demand.get(r, 0.0) or 0.0)
        crop_area[(r, t)] = crop_val + offset_crop
        grass_area[(r, t)] = grass_val + offset_grass

    years_sorted = sorted(set(years))
    future_years = [t for t in years_sorted if t > hist_end_year]
    prev_year = {years_sorted[i]: years_sorted[i - 1] if i > 0 else None
                 for i in range(len(years_sorted))}

    d_crop_pos = {}
    d_grass_pos = {}
    emis_crop = {}
    emis_pasture = {}

    regions = {r for (r, _, _) in idx.keys()} if idx else {r for (r, _, _) in qs.keys()}
    for r in regions:
        pool_veg_crop = 0.0
        pool_soil_crop = 0.0
        pool_veg_pasture = 0.0
        pool_soil_pasture = 0.0

        for t in future_years:
            prev_t = prev_year.get(t)
            if prev_t is None or prev_t < hist_end_year:
                crop_prev = float(base_cropland_area.get(r, base_crop_demand.get(r, 0.0) or 0.0) or 0.0)
                grass_prev = float(base_grassland_area.get(r, base_grass_demand.get(r, 0.0) or 0.0) or 0.0)
            else:
                crop_prev = crop_area.get((r, prev_t), 0.0)
                grass_prev = grass_area.get((r, prev_t), 0.0)

            crop_curr = crop_area.get((r, t), 0.0)
            grass_curr = grass_area.get((r, t), 0.0)
            d_crop = crop_curr - crop_prev
            d_grass = grass_curr - grass_prev
            d_crop_pos[(r, t)] = max(d_crop, 0.0)
            d_grass_pos[(r, t)] = max(d_grass, 0.0)

            delta_veg_crop = (forest_c_ha - cropland_c_ha) * d_crop
            delta_soil_crop = (forest_soil_c_ha - cropland_soil_c_ha) * d_crop
            delta_veg_pasture = (forest_c_ha - pasture_c_ha) * d_grass
            delta_soil_pasture = (forest_soil_c_ha - pasture_soil_c_ha) * d_grass

            if enable_shift and tau_shift > 0:
                if shift_area_mode == 'abs':
                    shift_crop = max(crop_curr, 0.0) / tau_shift
                    shift_grass = max(grass_curr, 0.0) / tau_shift
                else:
                    shift_crop = max(d_crop, 0.0) / tau_shift
                    shift_grass = max(d_grass, 0.0) / tau_shift

                delta_veg_crop += (forest_c_ha * shift_veg_frac - cropland_c_ha) * shift_crop * harvest_intensity
                delta_soil_crop += (forest_soil_c_ha * shift_soil_frac - cropland_soil_c_ha) * shift_crop * harvest_intensity
                delta_veg_pasture += (forest_c_ha * shift_veg_frac - pasture_c_ha) * shift_grass * harvest_intensity
                delta_soil_pasture += (forest_soil_c_ha * shift_soil_frac - pasture_soil_c_ha) * shift_grass * harvest_intensity

            pool_before_veg_crop = pool_veg_crop + delta_veg_crop
            pool_before_soil_crop = pool_soil_crop + delta_soil_crop
            pool_before_veg_pasture = pool_veg_pasture + delta_veg_pasture
            pool_before_soil_pasture = pool_soil_pasture + delta_soil_pasture

            emit_veg_crop = a_veg * pool_before_veg_crop
            emit_soil_crop = a_soil * pool_before_soil_crop
            emit_veg_pasture = a_veg * pool_before_veg_pasture
            emit_soil_pasture = a_soil * pool_before_soil_pasture

            emis_crop[(r, t)] = (emit_veg_crop + emit_soil_crop) * TC2CO2
            emis_pasture[(r, t)] = (emit_veg_pasture + emit_soil_pasture) * TC2CO2

            pool_veg_crop = pool_before_veg_crop * (1.0 - a_veg)
            pool_soil_crop = pool_before_soil_crop * (1.0 - a_soil)
            pool_veg_pasture = pool_before_veg_pasture * (1.0 - a_veg)
            pool_soil_pasture = pool_before_soil_pasture * (1.0 - a_soil)

    if negative_rows:
        if output_dir:
            out_path = Path(output_dir) / "negative_land_need_luc.csv"
            try:
                pd.DataFrame(negative_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                logger.warning(
                    "[LUC] clamped negative land need rows=%d -> %s",
                    len(negative_rows),
                    out_path,
                )
            except Exception as exc:
                logger.warning("[LUC] failed to write negative land need report: %s", exc)
        else:
            logger.warning("[LUC] clamped negative land need rows=%d (no output_dir)", len(negative_rows))

    return emis_crop, emis_pasture, d_crop_pos, d_grass_pos


def solve_with_luc_iteration(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    time_limit: float = 300.0,
    dict_v3_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    gurobi_log_path: Optional[str] = None,
    solver_method: Optional[int] = None,
    solver_threads: Optional[int] = None,
    use_relative_price: bool = False,
    relative_price_bounds: Tuple[float, float] = (0.1, 10.0),
    price_bounds_mode: str = 'absolute',
    price_bounds_p0_mult: Tuple[float, float] = (0.1, 10.0),
    price_bounds: Tuple[float, float] = (1e-6, 1e6),
    price_wedge_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    price_wedge_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    price_wedge_by_region: Optional[Dict[str, float]] = None,
    market_clearing_mode: str = 'country_trade',
    armington_sigma_by_comm: Optional[Dict[Any, float]] = None,
    trade_base_net_import: Optional[Dict[Tuple[str, str], float]] = None,
    trade_base_volume: Optional[Dict[Tuple[str, str], float]] = None,
    trade_cap_region_volume: Optional[Dict[Tuple[str, str], float]] = None,
    trade_cap_region_map: Optional[Dict[Any, str]] = None,
    trade_cap_ratio: Any = None,
    trade_cap_exempt_pairs: Optional[set] = None,
    armington_trade_scale: Optional[float] = None,
    armington_trade_slack_penalty: Optional[float] = None,
    supply_curtailment_enabled: bool = False,
    supply_curtailment_penalty: Optional[float] = None,
    zero_price_shutdown_enabled: bool = False,
    zero_demand_production_shutdown: bool = False,
    qty_scale: float = 1.0,
    land_scale: float = 1.0,
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    tax_unit_adder: Optional[Dict[Tuple[str, str, int], float]] = None,
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    land_soft_constraints_enabled: bool = False,
    land_slack_max_rate: Optional[float] = None,
    land_slack_penalty: Optional[float] = None,
    land_delta_anchor_to_available_stock: bool = False,
    grass_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    forest_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    forest_global_target_slack_enabled: bool = False,
    forest_global_target_slack_penalty: Optional[float] = None,
    forest_global_target_slack_max_rate: Optional[float] = None,
    forest_nonneg_ratio: float = 1.0,
    cropland_nonforest_expand_ratio: float = 1.0,
    pasture_nonforest_expand_ratio: float = 1.0,
    base_cropland_by_region: Optional[Dict[str, float]] = None,
    base_grassland_by_region: Optional[Dict[str, float]] = None,
    base_forest_by_region: Optional[Dict[str, float]] = None,
    background_forest_to_cropland_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_forest_to_grassland_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_cropland_to_forest_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_grassland_to_forest_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    land_demand_calibration_mode: str = 'none',
    yield_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    grassland_method: str = 'dynamic',
    grassland_conversion_penalty: float = 0.0,
    grassland_to_cropland_cost_mode: str = 'per_ha_cost',
    cropland_to_grassland_penalty: float = 0.0,
    land_conversion_allocation_mode: str = 'priority_nonforest_pasture_forest',
    land_conversion_priority_penalty_per_ha: float = 1e6,
    land_priority_weight_grassland_to_cropland: float = 1.0,
    land_priority_weight_forest_to_cropland: float = 100.0,
    land_priority_weight_forest_to_grassland: float = 100.0,
    luc_direct_carbon_price: bool = False,
    limit_reforestation_to_released_ag_land: bool = True,
    prevent_land_conversion_cycles: bool = True,
    reforestation_physical_cap_enabled: bool = True,
    reforestation_max_forest_increase_ratio: Optional[float] = 0.30,
    ruminant_intake_cap: Optional[Dict[Tuple[str, int], float]] = None,
    ruminant_commodities: Optional[List[str]] = None,
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    hist_max_production: Optional[Dict[Tuple[str, str], float]] = None,
    max_iterations: int = 5,
    convergence_tolerance: float = 0.01,
    luc_params: Optional[Dict[str, Any]] = None,
    luc_shift_area_mode: str = 'abs',
    # demand method
    demand_method: str = 'elasticity',  # 'elasticity' | 'nutrition' | 'nutrition_band' | 'nutrition_anchor'
    nutrition_profile_xlsx: Optional[str] = None,
    nutrition_profile_sheet: Any = 0,
    nutrition_indicator: str = 'energy',
    nutrition_use_baseyear_for_future: bool = True,
    nutrition_band_epsilon: float = 0.1,
    nutrition_feed_t_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    nutrition_residual_demand_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    bioenergy_crop_demand_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    energy_crop_land_requirement_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    waste_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    losses_ratio_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_mode: Optional[str] = None,
    feed_crop_link_base: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_coeff: Optional[Dict[Tuple[str, str, str, int], float]] = None,
    feed_background_crop: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_background_pasture: Optional[Dict[Tuple[str, int], float]] = None,
    feed_crop_link_livestock: Optional[List[str]] = None,
    feed_crop_link_override: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_credit: Optional[Dict[Tuple[str, str, int], float]] = None,
    # market slack
    max_slack_rate: Optional[float] = 0.1,
    max_shortage_slack_rate: Any = "inherit",
    max_excess_slack_rate: Any = "inherit",
    slack_penalty: Optional[float] = 1e6,
    disable_production_cost_term: bool = True,
    production_cost_weight: float = 1.0,
    
    cross_terms_top_n: Optional[int] = None,
    cross_terms_scale: Optional[float] = None,
    exclude_commodities: Optional[List[str]] = None,
    # unit cost
    unit_cost_data: Optional[Dict[Tuple[str, str], float]] = None,
    baseline_scenario_result: Optional[Dict[str, Any]] = None,
    process_cost_mapping: Optional[Dict[str, str]] = None,
    cost_calculation_method: str = 'MACC',
    active_strategy_cost_keys: Optional[Sequence[str]] = None,
    strategy_cost_regions: Optional[Sequence[str]] = None,
    cost_database_metadata: Optional[Mapping[str, Any]] = None,
    cost_strategy_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    post_solve_violation_tol: Optional[float] = 1e-6,
    post_solve_violation_top_n: int = 20,
) -> Dict:
    logger = logging.getLogger(__name__)

    luc_penalty: Dict[Tuple[str, int], Dict[str, float]] = {}
    history = []
    prev_total = None

    for iteration in range(1, max_iterations + 1):
        result = solve_linear_regional(
            nodes=nodes,
            commodities=commodities,
            years=years,
            time_limit=time_limit,
            dict_v3_path=dict_v3_path,
            output_dir=output_dir,
            gurobi_log_path=gurobi_log_path,
            solver_method=solver_method,
            solver_threads=solver_threads,
            use_relative_price=use_relative_price,
            relative_price_bounds=relative_price_bounds,
            price_bounds_mode=price_bounds_mode,
            price_bounds_p0_mult=price_bounds_p0_mult,
            price_bounds=price_bounds,
            price_wedge_by_region_comm_year=price_wedge_by_region_comm_year,
            price_wedge_by_region_comm=price_wedge_by_region_comm,
            price_wedge_by_region=price_wedge_by_region,
            market_clearing_mode=market_clearing_mode,
            armington_sigma_by_comm=armington_sigma_by_comm,
            trade_base_net_import=trade_base_net_import,
            trade_base_volume=trade_base_volume,
            trade_cap_region_volume=trade_cap_region_volume,
            trade_cap_region_map=trade_cap_region_map,
            trade_cap_ratio=trade_cap_ratio,
            trade_cap_exempt_pairs=trade_cap_exempt_pairs,
            armington_trade_scale=armington_trade_scale,
            armington_trade_slack_penalty=armington_trade_slack_penalty,
            qty_scale=qty_scale,
            land_scale=land_scale,
            population_by_country_year=population_by_country_year,
            income_mult_by_country_year=income_mult_by_country_year,
            macc_path=macc_path,
            land_carbon_price_by_year=land_carbon_price_by_year,
            tax_unit_adder=tax_unit_adder,
            luc_opt_mode='iterative',
            luc_params=luc_params,
            luc_shift_area_mode=luc_shift_area_mode,
            luc_penalty_by_region_year=luc_penalty,
            nutrition_rhs=nutrition_rhs,
            nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
            land_area_limits=land_area_limits,
            land_soft_constraints_enabled=land_soft_constraints_enabled,
            land_slack_max_rate=land_slack_max_rate,
            land_slack_penalty=land_slack_penalty,
            grass_area_by_region_year=grass_area_by_region_year,
            forest_area_by_region_year=forest_area_by_region_year,
            forest_global_target_slack_enabled=forest_global_target_slack_enabled,
            forest_global_target_slack_penalty=forest_global_target_slack_penalty,
            forest_global_target_slack_max_rate=forest_global_target_slack_max_rate,
            forest_nonneg_ratio=forest_nonneg_ratio,
            cropland_nonforest_expand_ratio=cropland_nonforest_expand_ratio,
            pasture_nonforest_expand_ratio=pasture_nonforest_expand_ratio,
            base_cropland_by_region=base_cropland_by_region,
            base_grassland_by_region=base_grassland_by_region,
            base_forest_by_region=base_forest_by_region,
            background_forest_to_cropland_by_region_year=background_forest_to_cropland_by_region_year,
            background_forest_to_grassland_by_region_year=background_forest_to_grassland_by_region_year,
            background_cropland_to_forest_by_region_year=background_cropland_to_forest_by_region_year,
            background_grassland_to_forest_by_region_year=background_grassland_to_forest_by_region_year,
            land_demand_calibration_mode=land_demand_calibration_mode,
            yield_by_region_comm_year=yield_by_region_comm_year,
            yield_t_per_ha_default=yield_t_per_ha_default,
            grassland_method=grassland_method,
            grassland_conversion_penalty=grassland_conversion_penalty,
            grassland_to_cropland_cost_mode=grassland_to_cropland_cost_mode,
            cropland_to_grassland_penalty=cropland_to_grassland_penalty,
            land_conversion_allocation_mode=land_conversion_allocation_mode,
            land_conversion_priority_penalty_per_ha=land_conversion_priority_penalty_per_ha,
            land_priority_weight_grassland_to_cropland=land_priority_weight_grassland_to_cropland,
            land_priority_weight_forest_to_cropland=land_priority_weight_forest_to_cropland,
            land_priority_weight_forest_to_grassland=land_priority_weight_forest_to_grassland,
            luc_direct_carbon_price=luc_direct_carbon_price,
            limit_reforestation_to_released_ag_land=limit_reforestation_to_released_ag_land,
            prevent_land_conversion_cycles=prevent_land_conversion_cycles,
            reforestation_physical_cap_enabled=reforestation_physical_cap_enabled,
            reforestation_max_forest_increase_ratio=reforestation_max_forest_increase_ratio,
            ruminant_intake_cap=ruminant_intake_cap,
            ruminant_commodities=ruminant_commodities,
            max_growth_rate_per_period=max_growth_rate_per_period,
            max_decline_rate_per_period=max_decline_rate_per_period,
            hist_end_year=hist_end_year,
            hist_max_production=hist_max_production,
            max_slack_rate=max_slack_rate,
            max_shortage_slack_rate=max_shortage_slack_rate,
            max_excess_slack_rate=max_excess_slack_rate,
            slack_penalty=slack_penalty,
            supply_curtailment_enabled=supply_curtailment_enabled,
            supply_curtailment_penalty=supply_curtailment_penalty,
            zero_price_shutdown_enabled=zero_price_shutdown_enabled,
            zero_demand_production_shutdown=zero_demand_production_shutdown,
            land_delta_anchor_to_available_stock=land_delta_anchor_to_available_stock,
            cross_terms_top_n=cross_terms_top_n,
            cross_terms_scale=cross_terms_scale,
            exclude_commodities=exclude_commodities,
            unit_cost_data=unit_cost_data,
            baseline_scenario_result=baseline_scenario_result,
            process_cost_mapping=process_cost_mapping,
            cost_calculation_method=cost_calculation_method,
            active_strategy_cost_keys=active_strategy_cost_keys,
            strategy_cost_regions=strategy_cost_regions,
            cost_database_metadata=cost_database_metadata,
            cost_strategy_metadata=cost_strategy_metadata,
            demand_method=demand_method,
            nutrition_profile_xlsx=nutrition_profile_xlsx,
            nutrition_indicator=nutrition_indicator,
            nutrition_use_baseyear_for_future=nutrition_use_baseyear_for_future,
            nutrition_band_epsilon=nutrition_band_epsilon,
            nutrition_feed_t_by_country_comm_year=nutrition_feed_t_by_country_comm_year,
            nutrition_residual_demand_by_country_comm_year=nutrition_residual_demand_by_country_comm_year,
            bioenergy_crop_demand_by_country_comm_year=bioenergy_crop_demand_by_country_comm_year,
            energy_crop_land_requirement_by_region_year=energy_crop_land_requirement_by_region_year,
            waste_reduction_by=waste_reduction_by,
            losses_ratio_by=losses_ratio_by,
            feed_crop_link_mode=feed_crop_link_mode,
            feed_crop_link_base=feed_crop_link_base,
            feed_crop_link_coeff=feed_crop_link_coeff,
        feed_background_crop=feed_background_crop,
        feed_background_pasture=feed_background_pasture,
            feed_crop_link_livestock=feed_crop_link_livestock,
            feed_crop_link_override=feed_crop_link_override,
            feed_crop_link_credit=feed_crop_link_credit,
            post_solve_violation_tol=post_solve_violation_tol,
            post_solve_violation_top_n=post_solve_violation_top_n,
        )

        status = result.get('status')
        if status != gp.GRB.OPTIMAL:
            logger.warning(f"[LUC_ITER] iteration {iteration} non-optimal status={status}")
            result['luc_iter_history'] = history
            return result

        emis_crop, emis_pasture, d_crop_pos, d_grass_pos = _compute_luc_emissions_from_solution(
            qs=result.get('Qs', {}),
            idx=result.get('idx', {}),
            years=years,
            hist_end_year=hist_end_year,
            yield_t_per_ha_default=yield_t_per_ha_default,
            grassland_method=grassland_method,
            grass_area_by_region_year=grass_area_by_region_year,
            output_dir=output_dir,
            grassland_to_cropland=result.get('grassland_to_cropland'),
            grassland_to_forest=result.get('grassland_to_forest'),
            cropland_to_grassland=result.get('cropland_to_grassland'),
            cropland_to_forest=result.get('cropland_to_forest'),
            forest_to_cropland=result.get('forest_to_cropland'),
            forest_to_grassland=result.get('forest_to_grassland'),
            base_cropland_area=result.get('base_cropland_area'),
            base_grassland_area=result.get('base_grassland_area'),
            luc_params=luc_params,
            luc_shift_area_mode=luc_shift_area_mode,
            feed_background_pasture=feed_background_pasture,
        )

        total_luc = sum(abs(v) for v in emis_crop.values()) + sum(abs(v) for v in emis_pasture.values())
        history.append({'iteration': iteration, 'total_luc': total_luc})
        logger.info(f"[LUC_ITER] iteration {iteration}: total_luc={total_luc:.6g}")

        if prev_total is not None:
            change = abs(total_luc - prev_total) / max(abs(prev_total), 1e-6)
            if change < convergence_tolerance:
                result['luc_iter_history'] = history
                return result

        prev_total = total_luc
        luc_penalty = {}
        for key in set(d_crop_pos.keys()) | set(d_grass_pos.keys()):
            crop_area = d_crop_pos.get(key, 0.0)
            grass_area = d_grass_pos.get(key, 0.0)
            crop_emis = max(0.0, emis_crop.get(key, 0.0))
            grass_emis = max(0.0, emis_pasture.get(key, 0.0))
            if crop_area <= 0 and grass_area <= 0:
                continue
            entry: Dict[str, float] = {}
            if crop_area > 0:
                entry['crop'] = crop_emis / max(crop_area, 1e-6)
            if grass_area > 0:
                entry['pasture'] = grass_emis / max(grass_area, 1e-6)
            if entry:
                luc_penalty[key] = entry

    result['luc_iter_history'] = history
    return result




# Helper functions


def _is_lulucf_process(name: str) -> bool:
    """Determine whether an emissions process concerns LULUCF (land-use change)."""
    if not name:
        return False
    s = str(name).lower()
    keys = ['forest', 'net forest', 'afforest', 'deforest', 'savanna', 
            'drained organic', 'organic soil', 'peat', 'lulucf', 'land use',
            'ag land abandonment', 'abandonment']
    return any(k in s for k in keys)


def _extract_luc_params(luc_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize LUC parameter dict with defaults."""
    params = luc_params or {}
    cveg = params.get('cveg', {}) or {}
    csoil = params.get('csoil', {}) or {}

    def _fval(val: Any, dflt: float) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(dflt)

    return {
        'forest_c_ha': _fval(cveg.get('forest', 150.0), 150.0),
        'cropland_c_ha': _fval(cveg.get('cropland', 5.0), 5.0),
        'pasture_c_ha': _fval(cveg.get('pasture', 10.0), 10.0),
        'othernat_c_ha': _fval(cveg.get('othernat', 10.0), 10.0),
        'forest_soil_c_ha': _fval(csoil.get('forest', 80.0), 80.0),
        'cropland_soil_c_ha': _fval(csoil.get('cropland', 50.0), 50.0),
        'pasture_soil_c_ha': _fval(csoil.get('pasture', 70.0), 70.0),
        'othernat_soil_c_ha': _fval(csoil.get('othernat', csoil.get('pasture', 70.0)), 70.0),
        'tau_veg': _fval(params.get('tau_veg', 20.0), 20.0),
        'tau_soil': _fval(params.get('tau_soil', 20.0), 20.0),
        'tau_shift': _fval(params.get('tau_shift', 15.0), 15.0),
        'harvest_intensity': _fval(params.get('harvest_intensity', 1.0), 1.0),
        'enable_shift': int(params.get('enable_shift', 1) or 0) == 1,
        'use_exponential_response': bool(params.get('use_exponential_response', True)),
    }


def _read_macc(macc_path: Optional[str]) -> pd.DataFrame:
    """Read marginal abatement cost curve (MACC) data."""
    if not macc_path:
        return pd.DataFrame()
    try:
        return pd.read_pickle(macc_path)
    except Exception:
        try:
            return pickle.load(open(macc_path, 'rb'))
        except Exception:
            return pd.DataFrame()



# Helpers for nutrition-driven demand


def _norm_m49_code(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.startswith("'"):
        s = s[1:]
    s = s.strip()
    if not s:
        return None
    if s.count('.') == 1:
        left, right = s.split('.', 1)
        if left.isdigit() and right.strip('0') == '':
            s = left
    if s.isdigit():
        return f"'{s.zfill(3)}"
    digits = ''.join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    return f"'{digits.zfill(3)}"


def _build_report_luc_gross_ef_by_region(
    luc_params: Optional[Dict[str, Any]],
    regions: List[str],
    dict_v3_path: Optional[str],
    luc_shift_area_mode: str = 'abs',
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Build report-aligned one-year EF for forest->crop/pasture transitions.

    Returns tCO2/ha coefficients keyed by solver region.  Country-specific
    cveg/csoil values are used when solver regions are M49 countries; otherwise
    the generic LUCE parameter fallback is used.
    """
    cfg = _extract_luc_params(luc_params)
    use_exp = bool(cfg.get('use_exponential_response', True))
    tau_veg = float(cfg.get('tau_veg', 20.0))
    tau_soil = float(cfg.get('tau_soil', 20.0))
    a_veg = 1.0 - math.exp(-1.0 / max(tau_veg, 1e-6)) if use_exp else 1.0
    a_soil = 1.0 - math.exp(-1.0 / max(tau_soil, 1e-6)) if use_exp else 1.0

    enable_shift = bool(cfg.get('enable_shift', True))
    tau_shift = float(cfg.get('tau_shift', 15.0))
    harvest_intensity = float(cfg.get('harvest_intensity', 1.0))
    if use_exp and tau_shift > 0:
        shift_veg_frac = 1.0 - math.exp(-tau_shift / max(tau_veg, 1e-6))
        shift_soil_frac = 1.0 - math.exp(-tau_shift / max(tau_soil, 1e-6))
    else:
        shift_veg_frac = 1.0
        shift_soil_frac = 1.0

    shift_area_mode = str(luc_shift_area_mode or 'abs').strip().lower()
    if shift_area_mode not in {'abs', 'delta_pos'}:
        shift_area_mode = 'abs'
    include_shift = bool(enable_shift and tau_shift > 0 and shift_area_mode == 'delta_pos')

    def _ef(
        forest_c_ha: float,
        target_c_ha: float,
        forest_soil_c_ha: float,
        target_soil_c_ha: float,
    ) -> float:
        # Same first-year response as luc_emission_module: add carbon to the
        # pool first, then emit a_veg/a_soil of the pool in the report year.
        veg_term = (float(forest_c_ha) - float(target_c_ha)) * a_veg
        soil_term = (float(forest_soil_c_ha) - float(target_soil_c_ha)) * a_soil
        if include_shift:
            veg_term += (
                float(forest_c_ha) * shift_veg_frac - float(target_c_ha)
            ) * a_veg * harvest_intensity / tau_shift
            soil_term += (
                float(forest_soil_c_ha) * shift_soil_frac - float(target_soil_c_ha)
            ) * a_soil * harvest_intensity / tau_shift
        return max(0.0, (veg_term + soil_term) * TC2CO2)

    generic_crop = _ef(
        float(cfg.get('forest_c_ha', 150.0)),
        float(cfg.get('cropland_c_ha', 5.0)),
        float(cfg.get('forest_soil_c_ha', 80.0)),
        float(cfg.get('cropland_soil_c_ha', 50.0)),
    )
    generic_pasture = _ef(
        float(cfg.get('forest_c_ha', 150.0)),
        float(cfg.get('pasture_c_ha', 10.0)),
        float(cfg.get('forest_soil_c_ha', 80.0)),
        float(cfg.get('pasture_soil_c_ha', 70.0)),
    )

    crop_ef_by_region: Dict[str, float] = {}
    pasture_ef_by_region: Dict[str, float] = {}
    m49_by_region: Dict[str, str] = {}
    for region in regions:
        m49 = _norm_m49_code(region)
        if m49:
            m49_by_region[str(region)] = m49

    country_specific_regions = 0
    if m49_by_region and luc_params and dict_v3_path:
        try:
            from luc_emission_module import (
                _build_country_luc_values,
                _build_region_maps,
                _load_dict_v3_region,
            )

            cveg_lookup = luc_params.get('cveg_lookup')
            csoil_lookup = luc_params.get('csoil_lookup')
            if cveg_lookup and csoil_lookup:
                region_df = _load_dict_v3_region(dict_v3_path)
                _, _, m49_attrs, _ = _build_region_maps(region_df)
                m49_list = sorted(set(m49_by_region.values()))
                cveg_by_m49, csoil_by_m49 = _build_country_luc_values(
                    m49_list,
                    cveg_lookup,
                    csoil_lookup,
                    m49_attrs,
                )
                for region, m49 in m49_by_region.items():
                    cveg_vals = cveg_by_m49.get(m49)
                    csoil_vals = csoil_by_m49.get(m49)
                    if not cveg_vals or not csoil_vals:
                        continue
                    crop_ef_by_region[region] = _ef(
                        cveg_vals['forest'],
                        cveg_vals['cropland'],
                        csoil_vals['forest'],
                        csoil_vals['cropland'],
                    )
                    pasture_ef_by_region[region] = _ef(
                        cveg_vals['forest'],
                        cveg_vals['pasture'],
                        csoil_vals['forest'],
                        csoil_vals['pasture'],
                    )
                    country_specific_regions += 1
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "[LINEAR][LUC] failed to build country-specific report EF; "
                    "fallback to generic LUCE EF. error=%s",
                    exc,
                )

    fallback_regions = 0
    for region in regions:
        r_key = str(region)
        if r_key not in crop_ef_by_region:
            crop_ef_by_region[r_key] = generic_crop
            pasture_ef_by_region[r_key] = generic_pasture
            fallback_regions += 1

    meta = {
        'a_veg': a_veg,
        'a_soil': a_soil,
        'include_shift': include_shift,
        'shift_area_mode': shift_area_mode,
        'country_specific_regions': country_specific_regions,
        'fallback_regions': fallback_regions,
        'generic_crop_tco2_per_ha': generic_crop,
        'generic_pasture_tco2_per_ha': generic_pasture,
    }
    return crop_ef_by_region, pasture_ef_by_region, meta


def _build_report_luc_to_othernat_ef_by_region(
    luc_params: Optional[Dict[str, Any]],
    regions: List[str],
    dict_v3_path: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Build signed first-year EF for agricultural abandonment to other natural land.

    Coefficients are tCO2/ha and follow the same source-minus-target pool
    convention as grassland->cropland. Negative values mean natural recovery is
    a removal in the report year.
    """
    cfg = _extract_luc_params(luc_params)
    use_exp = bool(cfg.get('use_exponential_response', True))
    tau_veg = float(cfg.get('tau_veg', 20.0))
    tau_soil = float(cfg.get('tau_soil', 20.0))
    a_veg = 1.0 - math.exp(-1.0 / max(tau_veg, 1e-6)) if use_exp else 1.0
    a_soil = 1.0 - math.exp(-1.0 / max(tau_soil, 1e-6)) if use_exp else 1.0

    def _ef(
        source_c_ha: float,
        target_c_ha: float,
        source_soil_c_ha: float,
        target_soil_c_ha: float,
    ) -> float:
        veg_term = (float(source_c_ha) - float(target_c_ha)) * a_veg
        soil_term = (float(source_soil_c_ha) - float(target_soil_c_ha)) * a_soil
        return (veg_term + soil_term) * TC2CO2

    generic_crop = _ef(
        float(cfg.get('cropland_c_ha', 5.0)),
        float(cfg.get('othernat_c_ha', 10.0)),
        float(cfg.get('cropland_soil_c_ha', 50.0)),
        float(cfg.get('othernat_soil_c_ha', cfg.get('pasture_soil_c_ha', 70.0))),
    )
    generic_pasture = _ef(
        float(cfg.get('pasture_c_ha', 10.0)),
        float(cfg.get('othernat_c_ha', 10.0)),
        float(cfg.get('pasture_soil_c_ha', 70.0)),
        float(cfg.get('othernat_soil_c_ha', cfg.get('pasture_soil_c_ha', 70.0))),
    )

    crop_ef_by_region: Dict[str, float] = {}
    pasture_ef_by_region: Dict[str, float] = {}
    m49_by_region: Dict[str, str] = {}
    for region in regions:
        m49 = _norm_m49_code(region)
        if m49:
            m49_by_region[str(region)] = m49

    country_specific_regions = 0
    if m49_by_region and luc_params and dict_v3_path:
        try:
            from luc_emission_module import (
                _build_country_luc_values,
                _build_region_maps,
                _load_dict_v3_region,
            )

            cveg_lookup = luc_params.get('cveg_lookup')
            csoil_lookup = luc_params.get('csoil_lookup')
            if cveg_lookup and csoil_lookup:
                region_df = _load_dict_v3_region(dict_v3_path)
                _, _, m49_attrs, _ = _build_region_maps(region_df)
                m49_list = sorted(set(m49_by_region.values()))
                cveg_by_m49, csoil_by_m49 = _build_country_luc_values(
                    m49_list,
                    cveg_lookup,
                    csoil_lookup,
                    m49_attrs,
                )
                try:
                    default_othernat_soil = float(
                        (luc_params.get('csoil', {}) or {}).get(
                            'othernat',
                            (luc_params.get('csoil', {}) or {}).get('pasture', cfg.get('pasture_soil_c_ha', 70.0)),
                        )
                    )
                except Exception:
                    default_othernat_soil = float(cfg.get('pasture_soil_c_ha', 70.0))
                if not np.isfinite(default_othernat_soil):
                    default_othernat_soil = float(cfg.get('pasture_soil_c_ha', 70.0))
                for region, m49 in m49_by_region.items():
                    cveg_vals = cveg_by_m49.get(m49)
                    csoil_vals = csoil_by_m49.get(m49)
                    if not cveg_vals or not csoil_vals:
                        continue
                    try:
                        othernat_soil = float(csoil_vals.get('othernat', default_othernat_soil))
                    except Exception:
                        othernat_soil = default_othernat_soil
                    if not np.isfinite(othernat_soil):
                        othernat_soil = default_othernat_soil
                    crop_ef_by_region[region] = _ef(
                        cveg_vals['cropland'],
                        cveg_vals['othernat'],
                        csoil_vals['cropland'],
                        othernat_soil,
                    )
                    pasture_ef_by_region[region] = _ef(
                        cveg_vals['pasture'],
                        cveg_vals['othernat'],
                        csoil_vals['pasture'],
                        othernat_soil,
                    )
                    country_specific_regions += 1
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "[LINEAR][LUC] failed to build country-specific othernat recovery EF; "
                    "fallback to generic LUCE EF. error=%s",
                    exc,
                )

    fallback_regions = 0
    for region in regions:
        r_key = str(region)
        if r_key not in crop_ef_by_region:
            crop_ef_by_region[r_key] = generic_crop
            pasture_ef_by_region[r_key] = generic_pasture
            fallback_regions += 1

    meta = {
        'a_veg': a_veg,
        'a_soil': a_soil,
        'country_specific_regions': country_specific_regions,
        'fallback_regions': fallback_regions,
        'generic_crop_to_othernat_tco2_per_ha': generic_crop,
        'generic_pasture_to_othernat_tco2_per_ha': generic_pasture,
    }
    return crop_ef_by_region, pasture_ef_by_region, meta


def _build_report_luc_grass_to_crop_ef_by_region(
    luc_params: Optional[Dict[str, Any]],
    regions: List[str],
    dict_v3_path: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Build report-aligned one-year EF for pasture/grassland->cropland.

    Coefficients are signed tCO2/ha values: positive means conversion releases
    carbon in the first report year; negative means the cropland pool is larger
    than the pasture pool under the country-specific LUCE parameters.
    """
    cfg = _extract_luc_params(luc_params)
    use_exp = bool(cfg.get('use_exponential_response', True))
    tau_veg = float(cfg.get('tau_veg', 20.0))
    tau_soil = float(cfg.get('tau_soil', 20.0))
    a_veg = 1.0 - math.exp(-1.0 / max(tau_veg, 1e-6)) if use_exp else 1.0
    a_soil = 1.0 - math.exp(-1.0 / max(tau_soil, 1e-6)) if use_exp else 1.0

    def _ef(
        pasture_c_ha: float,
        crop_c_ha: float,
        pasture_soil_c_ha: float,
        crop_soil_c_ha: float,
    ) -> float:
        veg_term = (float(pasture_c_ha) - float(crop_c_ha)) * a_veg
        soil_term = (float(pasture_soil_c_ha) - float(crop_soil_c_ha)) * a_soil
        return (veg_term + soil_term) * TC2CO2

    generic_ef = _ef(
        float(cfg.get('pasture_c_ha', 10.0)),
        float(cfg.get('cropland_c_ha', 5.0)),
        float(cfg.get('pasture_soil_c_ha', 70.0)),
        float(cfg.get('cropland_soil_c_ha', 50.0)),
    )

    ef_by_region: Dict[str, float] = {}
    m49_by_region: Dict[str, str] = {}
    for region in regions:
        m49 = _norm_m49_code(region)
        if m49:
            m49_by_region[str(region)] = m49

    country_specific_regions = 0
    if m49_by_region and luc_params and dict_v3_path:
        try:
            from luc_emission_module import (
                _build_country_luc_values,
                _build_region_maps,
                _load_dict_v3_region,
            )

            cveg_lookup = luc_params.get('cveg_lookup')
            csoil_lookup = luc_params.get('csoil_lookup')
            if cveg_lookup and csoil_lookup:
                region_df = _load_dict_v3_region(dict_v3_path)
                _, _, m49_attrs, _ = _build_region_maps(region_df)
                m49_list = sorted(set(m49_by_region.values()))
                cveg_by_m49, csoil_by_m49 = _build_country_luc_values(
                    m49_list,
                    cveg_lookup,
                    csoil_lookup,
                    m49_attrs,
                )
                for region, m49 in m49_by_region.items():
                    cveg_vals = cveg_by_m49.get(m49)
                    csoil_vals = csoil_by_m49.get(m49)
                    if not cveg_vals or not csoil_vals:
                        continue
                    ef_by_region[region] = _ef(
                        cveg_vals['pasture'],
                        cveg_vals['cropland'],
                        csoil_vals['pasture'],
                        csoil_vals['cropland'],
                    )
                    country_specific_regions += 1
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "[LINEAR][LUC] failed to build country-specific grass->crop EF; "
                    "fallback to generic LUCE EF. error=%s",
                    exc,
                )

    fallback_regions = 0
    for region in regions:
        r_key = str(region)
        if r_key not in ef_by_region:
            ef_by_region[r_key] = generic_ef
            fallback_regions += 1

    meta = {
        'a_veg': a_veg,
        'a_soil': a_soil,
        'country_specific_regions': country_specific_regions,
        'fallback_regions': fallback_regions,
        'generic_grass_to_crop_tco2_per_ha': generic_ef,
    }
    return ef_by_region, meta


def _normalize_nutrition_item_name(name: Any) -> str:
    s = str(name).strip()
    if not s:
        return s
    s = s.replace('–', '-').replace('—', '-')
    s = s.replace('Milk - Excluding Butter', 'Milk')
    s = re.sub(r'\s*-\s*', '-', s)
    s = s.replace('Milk-Excluding Butter-', 'Milk-')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _normalize_comp_item_name(name: Any) -> str:
    s = str(name).strip()
    if not s:
        return s
    s = re.sub(r'\s*-\s*', '-', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _load_m49_to_country(dict_v3_path: Optional[str]) -> Dict[str, str]:
    if dict_v3_path is None:
        try:
            from config_paths import get_src_base
            dict_v3_path = str(Path(get_src_base()) / 'dict_v3.xlsx')
        except Exception:
            return {}
    try:
        df = pd.read_excel(dict_v3_path, sheet_name='region')
    except Exception:
        return {}
    if 'M49_Country_Code' not in df.columns or 'Region_label_new' not in df.columns:
        return {}
    out: Dict[str, str] = {}
    for _, row in df[['M49_Country_Code', 'Region_label_new']].dropna().iterrows():
        m49 = _norm_m49_code(row['M49_Country_Code'])
        country = str(row['Region_label_new']).strip()
        if m49 and country and country.lower() != 'no':
            out[m49] = country
    return out


def _load_nutrition_item_mapping(dict_v3_path: Optional[str], indicator: str) -> Dict[str, Tuple[str, float]]:
    if dict_v3_path is None:
        try:
            from config_paths import get_src_base
            dict_v3_path = str(Path(get_src_base()) / 'dict_v3.xlsx')
        except Exception:
            return {}
    try:
        df = pd.read_excel(dict_v3_path, sheet_name='Emis_item')
    except Exception:
        return {}
    if 'Item_Emis' not in df.columns or 'Item_Nutrition_Map' not in df.columns:
        return {}
    col_map = {
        'energy': 'kcal_per_100g',
        'protein': 'g_protein_per_100g',
        'fat': 'g_fat_per_100g',
    }
    value_col = col_map.get(indicator)
    if not value_col or value_col not in df.columns:
        return {}
    out: Dict[str, Tuple[str, float]] = {}
    for _, row in df[['Item_Emis', 'Item_Nutrition_Map', value_col]].dropna().iterrows():
        comm = str(row['Item_Emis']).strip()
        item_nut = str(row['Item_Nutrition_Map']).strip()
        if not comm or not item_nut or item_nut.lower() in {'no', 'nan'}:
            continue
        val = pd.to_numeric(row[value_col], errors='coerce')
        if not np.isfinite(val) or val <= 0:
            continue
        per_ton = float(val) * 10000.0
        key = _normalize_nutrition_item_name(item_nut).lower()
        out[key] = (comm, per_ton)
    return out


def _extend_years_by_group(df: pd.DataFrame,
                           key_cols: List[str],
                           value_col: str,
                           years: List[int]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    pivot = df.pivot_table(index=key_cols, columns='year', values=value_col, aggfunc='last')
    for y in sorted(set(years)):
        if y not in pivot.columns:
            pivot[y] = np.nan
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.ffill(axis=1)
    out = pivot.reset_index().melt(id_vars=key_cols, var_name='year', value_name=value_col)
    out['year'] = out['year'].astype(int)
    return out[out['year'].isin(years)]


def _load_nonfood_commodities(dict_v3_path: Optional[str]) -> set:
    if dict_v3_path is None:
        try:
            from config_paths import get_src_base
            dict_v3_path = str(Path(get_src_base()) / 'dict_v3.xlsx')
        except Exception:
            return set()
    try:
        df = pd.read_excel(dict_v3_path, sheet_name='Emis_item')
    except Exception:
        return set()
    if 'Item_Emis' not in df.columns or 'Item_Nutrition_Map' not in df.columns:
        return set()
    mask = df['Item_Nutrition_Map'].astype(str).str.strip().str.lower() == 'non food'
    return set(df.loc[mask, 'Item_Emis'].astype(str).str.strip().tolist())


def _load_crop_commodities(dict_v3_path: Optional[str]) -> set:
    if dict_v3_path is None:
        try:
            from config_paths import get_src_base
            dict_v3_path = str(Path(get_src_base()) / 'dict_v3.xlsx')
        except Exception:
            return set()
    try:
        df = pd.read_excel(dict_v3_path, sheet_name='Emis_item')
    except Exception:
        return set()
    if 'Item_Emis' not in df.columns or 'Item_Cat2' not in df.columns:
        return set()
    mask = df['Item_Cat2'].astype(str).str.strip().str.lower() == 'crop'
    return set(df.loc[mask, 'Item_Emis'].astype(str).str.strip().tolist())


def _load_item_demand_extra_and_map(dict_v3_path: Optional[str]) -> Tuple[set, Dict[str, List[str]]]:
    if dict_v3_path is None:
        try:
            from config_paths import get_src_base
            dict_v3_path = str(Path(get_src_base()) / 'dict_v3.xlsx')
        except Exception:
            return set(), {}
    if not dict_v3_path or not Path(dict_v3_path).exists():
        return set(), {}
    try:
        df = pd.read_excel(dict_v3_path, sheet_name='Emis_item')
    except Exception:
        return set(), {}
    if 'Item_Emis' not in df.columns:
        return set(), {}
    if 'Item_Demand_Map' not in df.columns or 'Item_Demand_Extra' not in df.columns:
        return set(), {}
    extra_items: set = set()
    demand_map: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        comm = str(row.get('Item_Emis', '')).strip()
        if not comm or comm.lower() in {'nan', 'no'}:
            continue
        extra_val = row.get('Item_Demand_Extra', None)
        extra_flag = False
        if extra_val is not None and not (isinstance(extra_val, float) and np.isnan(extra_val)):
            if isinstance(extra_val, (int, float)) and float(extra_val) == 1.0:
                extra_flag = True
            else:
                s = str(extra_val).strip().lower()
                if s in {'1', '1.0', 'true', 'yes', 'y'}:
                    extra_flag = True
        if extra_flag:
            extra_items.add(comm)
        raw_map = row.get('Item_Demand_Map', None)
        if raw_map is None or (isinstance(raw_map, float) and np.isnan(raw_map)):
            continue
        raw_map = str(raw_map).strip()
        if not raw_map or raw_map.lower() in {'nan', 'no'}:
            continue
        parts = [p.strip() for p in re.split(r"[;|]+", raw_map) if p.strip()]
        if parts:
            demand_map[comm] = parts
    return extra_items, demand_map


def _load_demand_composition_food_feed_ratio(path: str) -> Dict[Tuple[str, str], float]:
    if not path or not Path(path).exists():
        return {}
    try:
        df = pd.read_excel(path, sheet_name='ratio')
    except Exception:
        return {}
    df.columns = [str(c).strip() for c in df.columns]
    m49_col = 'M49_Country_Code' if 'M49_Country_Code' in df.columns else None
    if not m49_col:
        for c in df.columns:
            if 'm49' in str(c).lower():
                m49_col = c
                break
    if not m49_col or 'Item' not in df.columns or 'Element' not in df.columns or 'Y2020' not in df.columns:
        return {}
    df['m49_code'] = df[m49_col].apply(_norm_m49_code)
    df = df.dropna(subset=['m49_code'])
    df['Item'] = df['Item'].astype(str).apply(_normalize_comp_item_name)
    df['Y2020'] = pd.to_numeric(df['Y2020'], errors='coerce')
    df = df.dropna(subset=['Y2020'])
    df['Element'] = df['Element'].astype(str).str.strip().str.lower()
    df = df[df['Element'].isin({'food', 'feed'})]
    if df.empty:
        return {}
    grouped = df.groupby(['m49_code', 'Item'], as_index=False)['Y2020'].sum()
    out: Dict[Tuple[str, str], float] = {}
    for r in grouped.itertuples(index=False):
        m49 = r.m49_code
        item = str(r.Item).strip()
        if not m49 or not item:
            continue
        out[(m49, item)] = float(r.Y2020)
    return out


def _load_demand_composition_losses_ratio(path: str) -> Dict[Tuple[str, str], float]:
    if not path or not Path(path).exists():
        return {}
    try:
        df = pd.read_excel(path, sheet_name='ratio')
    except Exception:
        return {}
    df.columns = [str(c).strip() for c in df.columns]
    m49_col = 'M49_Country_Code' if 'M49_Country_Code' in df.columns else None
    if not m49_col:
        for c in df.columns:
            if 'm49' in str(c).lower():
                m49_col = c
                break
    if not m49_col or 'Item' not in df.columns or 'Element' not in df.columns or 'Y2020' not in df.columns:
        return {}
    df['m49_code'] = df[m49_col].apply(_norm_m49_code)
    df = df.dropna(subset=['m49_code'])
    df['Item'] = df['Item'].astype(str).apply(_normalize_comp_item_name)
    df['Y2020'] = pd.to_numeric(df['Y2020'], errors='coerce')
    df = df.dropna(subset=['Y2020'])
    df['Element'] = df['Element'].astype(str).str.strip().str.lower()
    df = df[df['Element'] == 'losses']
    if df.empty:
        return {}
    grouped = df.groupby(['m49_code', 'Item'], as_index=False)['Y2020'].sum()
    out: Dict[Tuple[str, str], float] = {}
    for r in grouped.itertuples(index=False):
        m49 = r.m49_code
        item = str(r.Item).strip()
        if not m49 or not item:
            continue
        out[(m49, item)] = float(r.Y2020)
    return out


def _filter_nutrition_profile_years_for_step(
    long_df: pd.DataFrame,
    *,
    years: List[int],
    hist_end_year: int,
    use_baseyear_for_future: bool,
) -> pd.DataFrame:
    requested = {int(year) for year in years}
    year_values = pd.to_numeric(long_df["year"], errors="coerce")
    if use_baseyear_for_future and any(year > hist_end_year for year in requested):
        keep = year_values.isin(requested) | year_values.le(hist_end_year)
    else:
        keep = year_values.isin(requested)
    return long_df.loc[keep].copy()



def build_nutrition_demand_map(nutrition_xlsx: str,
                               dict_v3_path: Optional[str],
                               indicator: str,
                               years: List[int],
                               population_by_country_year: Dict[Tuple[str, int], float],
                               use_regional_agg: Optional[bool] = None,
                               hist_end_year: int = 2020,
                               use_baseyear_for_future: bool = True,
                               feed_t_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
                               waste_reduction_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
                               losses_ratio_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
                               country_by_m49: Optional[Dict[str, str]] = None,
                               nutrition_profile_sheet: Any = 0,
                               separate_nonfood_demand: bool = False,
                               ) -> Dict[Tuple[str, str, int], float]:
    logger = logging.getLogger(__name__)
    if not nutrition_xlsx or not Path(nutrition_xlsx).exists():
        logger.warning(f"[NUT_DEMAND] 找不到营养数据文件: {nutrition_xlsx}")
        return {}
    item_map = _load_nutrition_item_mapping(dict_v3_path, indicator)
    if not item_map:
        logger.warning("[NUT_DEMAND] Item_Nutrition_Map 映射为空，无法构建营养需求")
        return {}
    df = read_tabular_cached(nutrition_xlsx, sheet_name=nutrition_profile_sheet)
    logger.info(
        "[NUT_DEMAND] nutrition profile source=%s sheet=%s",
        nutrition_xlsx,
        nutrition_profile_sheet,
    )
    df.columns = [str(c).strip() for c in df.columns]
    required_cols = {'M49_Country_Code', 'Item', 'Element'}
    if not required_cols.issubset(df.columns):
        logger.warning("[NUT_DEMAND] Nutrition_profile 缺少必要列")
        return {}
    element_map = {
        'energy': 'Food supply (kcal/capita/day)',
        'protein': 'Protein supply quantity (g/capita/day)',
        'fat': 'Fat supply quantity (g/capita/day)',
    }
    element_name = element_map.get(indicator, element_map['energy'])
    df = df[df['Element'].astype(str).str.strip().str.lower() == element_name.lower()].copy()
    if df.empty:
        logger.warning(f"[NUT_DEMAND] Nutrition_profile 中找不到元素: {element_name}")
        return {}
    year_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit()]
    if not year_cols:
        logger.warning("[NUT_DEMAND] Nutrition_profile 未包含年份列")
        return {}
    df['m49_code'] = df['M49_Country_Code'].apply(_norm_m49_code)
    df = df.dropna(subset=['m49_code'])
    long_df = df.melt(id_vars=['m49_code', 'Item'], value_vars=year_cols,
                      var_name='year', value_name='nutrient_pc_day')
    long_df['year'] = long_df['year'].astype(str).str.lstrip('Y').astype(int)
    long_df = _filter_nutrition_profile_years_for_step(
        long_df,
        years=years,
        hist_end_year=hist_end_year,
        use_baseyear_for_future=use_baseyear_for_future,
    )
    long_df['nutrient_pc_day'] = pd.to_numeric(long_df['nutrient_pc_day'], errors='coerce')
    long_df = long_df.dropna(subset=['nutrient_pc_day'])
    if long_df.empty:
        logger.warning("[NUT_DEMAND] Nutrition_profile 转长表后为空")
        return {}
    long_df['item_norm'] = long_df['Item'].astype(str).apply(_normalize_nutrition_item_name)
    if use_baseyear_for_future:
        # Base-period value: latest year <= hist_end_year.
        hist_df = long_df[long_df['year'] <= hist_end_year].copy()
        if hist_df.empty:
            logger.warning("[NUT_DEMAND] 基期营养数据为空")
        base_vals = (
            hist_df.sort_values('year')
            .groupby(['m49_code', 'item_norm'], as_index=False)
            .last()[['m49_code', 'item_norm', 'nutrient_pc_day']]
        )
        future_years = sorted([y for y in years if y > hist_end_year])
        if future_years and not base_vals.empty:
            fut_rows = []
            for r in base_vals.itertuples(index=False):
                for y in future_years:
                    fut_rows.append({
                        'm49_code': r.m49_code,
                        'Item': r.item_norm,
                        'year': y,
                        'nutrient_pc_day': r.nutrient_pc_day,
                        'item_norm': r.item_norm
                    })
            future_df = pd.DataFrame(fut_rows)
        else:
            future_df = pd.DataFrame(columns=['m49_code', 'Item', 'year', 'nutrient_pc_day', 'item_norm'])
        hist_keep = long_df[
            long_df['year'].le(hist_end_year) & long_df['year'].isin(years)
        ].copy()
        hist_keep = hist_keep[['m49_code', 'Item', 'year', 'nutrient_pc_day', 'item_norm']]
        long_df = pd.concat([hist_keep, future_df], ignore_index=True)
    else:
        long_df = _extend_years_by_group(
            long_df[['m49_code', 'item_norm', 'year', 'nutrient_pc_day']],
            key_cols=['m49_code', 'item_norm'],
            value_col='nutrient_pc_day',
            years=years,
        )
        long_df['Item'] = long_df['item_norm']
        long_df = long_df.dropna(subset=['nutrient_pc_day'])
    missing_items = sorted({i for i in long_df['item_norm'].unique() if i.lower() not in item_map})
    long_df['map_tuple'] = long_df['item_norm'].str.lower().map(item_map)
    long_df = long_df.dropna(subset=['map_tuple'])
    long_df[['commodity', 'nutrient_per_ton']] = pd.DataFrame(
        long_df['map_tuple'].tolist(), index=long_df.index
    )
    long_df = long_df.dropna(subset=['commodity', 'nutrient_per_ton'])
    if long_df.empty:
        logger.warning("[NUT_DEMAND] 营养Item映射后为空")
        return {}
    extra_items, demand_item_map = _load_item_demand_extra_and_map(dict_v3_path)
    food_ratio_lookup: Dict[Tuple[str, str], float] = {}
    loss_ratio_lookup: Dict[Tuple[str, str], float] = {}
    if extra_items:
        try:
            from config_paths import get_input_base
            comp_path = str(Path(get_input_base()) / 'Production_Trade' / 'Demand_composition.xlsx')
        except Exception:
            comp_path = ''
        food_ratio_lookup = _load_demand_composition_food_feed_ratio(comp_path)
        loss_ratio_lookup = _load_demand_composition_losses_ratio(comp_path)
        multi_mapped = {k: v for k, v in demand_item_map.items() if k in extra_items and len(v) > 1}
        if multi_mapped:
            items = sorted(multi_mapped.keys())
            logger.error("[NUT_DEMAND] Item_Demand_Extra 多项映射触发报错: %s", items)
            logger.error("[NUT_DEMAND] Item_Demand_Extra 多项映射明细: %s", multi_mapped)
            raise ValueError(
                f"[NUT_DEMAND] Item_Demand_Extra has multiple Item_Demand_Map entries: {items}"
            )

    country_by_m49_map: Dict[str, str] = {}
    if waste_reduction_by_country_comm_year or losses_ratio_by_country_comm_year:
        if country_by_m49:
            country_by_m49_map = {str(k): str(v) for k, v in country_by_m49.items() if k and v}
        if not country_by_m49_map:
            country_by_m49_map = _load_m49_to_country(dict_v3_path)

    feed_map_norm: Dict[Tuple[str, str, int], float] = {}
    if feed_t_by_country_comm_year:
        country_to_m49: Optional[Dict[str, str]] = None
        if dict_v3_path:
            m49_to_country = _load_m49_to_country(dict_v3_path)
            if m49_to_country:
                country_to_m49 = {v: k for k, v in m49_to_country.items() if v}
        for key, val in feed_t_by_country_comm_year.items():
            try:
                country, comm, year = key
            except Exception:
                continue
            m49_code = _norm_m49_code(country)
            if not m49_code and country_to_m49:
                m49_code = country_to_m49.get(str(country).strip())
            if not m49_code:
                continue
            try:
                year_val = int(year)
            except Exception:
                continue
            comm_val = str(comm).strip()
            if not comm_val:
                continue
            try:
                val_f = float(val or 0.0)
            except Exception:
                continue
            if not np.isfinite(val_f) or val_f <= 0:
                continue
            feed_map_norm[(m49_code, comm_val, year_val)] = feed_map_norm.get(
                (m49_code, comm_val, year_val), 0.0
            ) + val_f

    use_regional = is_region_aggregation_enabled() if use_regional_agg is None else bool(use_regional_agg)
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)

    rows = []
    missing_ratio = 0
    missing_feed = 0
    feed_added = 0
    for r in long_df.itertuples(index=False):
        m49_code = str(r.m49_code)
        pop = population_by_country_year.get((m49_code, int(r.year)))
        if pop is None or pop <= 0:
            continue
        nutrient_per_ton = float(r.nutrient_per_ton)
        if nutrient_per_ton <= 0:
            continue
        qty_t = float(r.nutrient_pc_day) / nutrient_per_ton * 365.0 * float(pop)
        comm = str(r.commodity)
        if extra_items and comm in extra_items:
            mapped = demand_item_map.get(comm, [])
            if len(mapped) > 1:
                raise ValueError(
                    f"[NUT_DEMAND] Item_Demand_Extra item has multiple Item_Demand_Map entries: {comm} -> {mapped}"
                )
            food_qty = float(qty_t)
            if separate_nonfood_demand:
                rows.append((m49_code, comm, int(r.year), food_qty))
                continue
            feed_add = 0.0
            if feed_map_norm:
                feed_val = feed_map_norm.get((m49_code, comm, int(r.year)))
                if feed_val is not None and np.isfinite(feed_val) and feed_val > 0:
                    feed_add = float(feed_val)
                    feed_added += 1
                else:
                    missing_feed += 1
            base_qty = qty_t + feed_add
            ratios = []
            if mapped:
                item = _normalize_comp_item_name(mapped[0])
                ratio_val = food_ratio_lookup.get((m49_code, item))
                if ratio_val is not None and np.isfinite(ratio_val) and ratio_val > 0:
                    ratios.append(float(ratio_val))
            if ratios:
                ff_ratio = float(np.mean(ratios))
            else:
                ff_ratio = None

            losses_ratio_2020 = None
            if mapped:
                item = _normalize_comp_item_name(mapped[0])
                loss_val = loss_ratio_lookup.get((m49_code, item))
                if loss_val is not None and np.isfinite(loss_val):
                    losses_ratio_2020 = float(loss_val)

            loss_delta = None
            if country_by_m49_map:
                country_name = country_by_m49_map.get(m49_code)
                if country_name:
                    loss_delta, _ = _lookup_loss_delta(
                        key=(country_name, comm, int(r.year)),
                        all_key=(country_name, 'All', int(r.year)),
                        waste_reduction_by=waste_reduction_by_country_comm_year,
                        losses_ratio_by=losses_ratio_by_country_comm_year,
                    )
            if loss_delta is None:
                loss_delta, _ = _lookup_loss_delta(
                    key=(m49_code, comm, int(r.year)),
                    all_key=(m49_code, 'All', int(r.year)),
                    waste_reduction_by=waste_reduction_by_country_comm_year,
                    losses_ratio_by=losses_ratio_by_country_comm_year,
                )
            if loss_delta is not None:
                qty_t = base_qty * _loss_multiplier_from_delta(
                    losses_ratio_2020 if losses_ratio_2020 is not None else 0.0,
                    loss_delta,
                )
            elif ff_ratio is not None and ff_ratio > 0:
                denom = ff_ratio
                if denom > 0:
                    factor = min(1.0 / denom, 2000.0)
                    # food_qty comes from the nutrition profile and needs the
                    # Food+Feed share adjustment. feed_add is already an
                    # explicit feed tonnage, so do not inflate it again.
                    qty_t = food_qty * factor + feed_add
            else:
                missing_ratio += 1
                qty_t = base_qty
        rows.append((m49_code, comm, int(r.year), qty_t))
    if not rows:
        logger.warning("[NUT_DEMAND] 需求计算结果为空")
        return {}
    if missing_ratio:
        logger.warning("[NUT_DEMAND] missing Food+Feed ratio for %d country-item pairs (Item_Demand_Extra)", missing_ratio)
    if feed_map_norm and (feed_added or missing_feed):
        logger.info("[NUT_DEMAND] Item_Demand_Extra feed_t merged: used=%d, missing=%d", feed_added, missing_feed)

    if use_regional:
        out_rows = []
        for m49_code, comm, year, qty_t in rows:
            region_key = get_region(m49_code, m49=m49_code, dict_v3_path=dict_v3_path)
            out_rows.append((region_key, comm, year, qty_t))
        out_df = pd.DataFrame(out_rows, columns=['region', 'commodity', 'year', 'demand_t'])
    else:
        out_df = pd.DataFrame(rows, columns=['region', 'commodity', 'year', 'demand_t'])
    out = out_df.groupby(['region', 'commodity', 'year'])['demand_t'].sum().to_dict()
    n_regions = out_df['region'].nunique()
    n_comm = out_df['commodity'].nunique()
    logger.info(f"[NUT_DEMAND] 已构建营养驱动需求: {len(out)} 条，覆盖区域/国家={n_regions}，商品={n_comm}")
    if missing_items:
        logger.warning(f"[NUT_DEMAND] Nutrition_profile 未匹配到 Item_Nutrition_Map 的Item数量={len(missing_items)}，示例: {missing_items[:10]}")
    return {(k[0], k[1], k[2]): float(v) for k, v in out.items()}


def _nutrition_rhs_mismatches_for_active_step(
    nutrition_rhs: Dict[Tuple[str, int], float],
    totals_by_rt: Dict[Tuple[str, int], float],
    *,
    active_region_years: set[Tuple[str, int]],
    hist_end_year: int,
) -> List[Tuple[str, int, float, float]]:
    """Compare profile totals only for region-years in the current rolling step."""
    mismatches: List[Tuple[str, int, float, float]] = []
    for (region, year), rhs in nutrition_rhs.items():
        if year <= hist_end_year or (region, year) not in active_region_years:
            continue
        total = totals_by_rt.get((region, year), 0.0)
        rhs_value = float(rhs)
        tolerance = max(1e-6 * abs(rhs_value), 1e-3)
        if total + tolerance < rhs_value:
            mismatches.append((region, year, rhs_value, total))
    return mismatches



def _write_nutrition_missing_report(missing_keys: List[Tuple[str, str, int]],
                                    nutrition_profile_xlsx: Optional[str],
                                    nutrition_profile_sheet: Any,
                                    dict_v3_path: Optional[str],
                                    indicator: str,
                                    population_by_country_year: Optional[Dict[Tuple[str, int], float]],
                                    hist_end_year: int,
                                    use_baseyear_for_future: bool,
                                    output_dir: Optional[str],
                                    nonfood_commodities: Optional[set]) -> Optional[str]:
    if not missing_keys:
        return None
    if not nutrition_profile_xlsx or not Path(nutrition_profile_xlsx).exists():
        return None

    dict_path = dict_v3_path
    if dict_path is None:
        try:
            from config_paths import get_src_base
            dict_path = str(Path(get_src_base()) / 'dict_v3.xlsx')
        except Exception:
            dict_path = None

    item_map_by_comm: Dict[str, str] = {}
    nut_factor_by_comm: Dict[str, float] = {}
    if dict_path and Path(dict_path).exists():
        try:
            df = pd.read_excel(dict_path, sheet_name='Emis_item')
            df.columns = [str(c).strip() for c in df.columns]
            val_col = {
                'energy': 'kcal_per_100g',
                'protein': 'g_protein_per_100g',
                'fat': 'g_fat_per_100g',
            }.get(indicator, 'kcal_per_100g')
            for _, row in df.iterrows():
                comm = str(row.get('Item_Emis', '')).strip()
                item_nut = str(row.get('Item_Nutrition_Map', '')).strip()
                if comm:
                    if item_nut and item_nut.lower() not in {'no', 'nan'}:
                        item_map_by_comm.setdefault(comm, _normalize_nutrition_item_name(item_nut))
                    val = pd.to_numeric(row.get(val_col), errors='coerce')
                    if pd.notna(val) and val > 0:
                        nut_factor_by_comm[comm] = float(val)
        except Exception:
            pass

    # Load nutrition profile (element filtered)
    profile_df = read_tabular_cached(nutrition_profile_xlsx, sheet_name=nutrition_profile_sheet)
    profile_df.columns = [str(c).strip() for c in profile_df.columns]
    element_map = {
        'energy': 'Food supply (kcal/capita/day)',
        'protein': 'Protein supply quantity (g/capita/day)',
        'fat': 'Fat supply quantity (g/capita/day)',
    }
    element_name = element_map.get(indicator, element_map['energy'])
    profile_df = profile_df[profile_df['Element'].astype(str).str.strip().str.lower() == element_name.lower()].copy()
    profile_df['m49_code'] = profile_df['M49_Country_Code'].apply(_norm_m49_code)
    profile_df = profile_df.dropna(subset=['m49_code'])

    year_cols = [c for c in profile_df.columns if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit()]
    long_df = profile_df.melt(id_vars=['m49_code', 'Item'], value_vars=year_cols,
                              var_name='year', value_name='nutrient_pc_day')
    long_df['year'] = long_df['year'].astype(str).str.lstrip('Y').astype(int)
    long_df['nutrient_pc_day'] = pd.to_numeric(long_df['nutrient_pc_day'], errors='coerce')
    long_df['item_norm'] = long_df['Item'].astype(str).apply(_normalize_nutrition_item_name)

    base_val: Dict[Tuple[str, str], float] = {}
    year_val: Dict[Tuple[str, str, int], float] = {}
    if use_baseyear_for_future:
        hist_df = long_df[long_df['year'] <= hist_end_year].copy()
        hist_df = hist_df.dropna(subset=['nutrient_pc_day'])
        if not hist_df.empty:
            hist_df = hist_df.sort_values('year')
            for r in hist_df.groupby(['m49_code', 'item_norm'], as_index=False).last().itertuples(index=False):
                base_val[(r.m49_code, r.item_norm)] = float(r.nutrient_pc_day)
    else:
        long_df = long_df.dropna(subset=['nutrient_pc_day'])
        for r in long_df.itertuples(index=False):
            year_val[(r.m49_code, r.item_norm, int(r.year))] = float(r.nutrient_pc_day)

    profile_items = set((r.m49_code, r.item_norm) for r in long_df[['m49_code', 'item_norm']].dropna().itertuples(index=False))

    # Country names
    m49_to_country = _load_m49_to_country(dict_path)

    rows = []
    for (r, j, t) in missing_keys:
        if nonfood_commodities and j in nonfood_commodities:
            continue
        reason = 'unknown'
        item_nut = item_map_by_comm.get(j, '')
        if not item_nut:
            reason = 'missing_item_nutrition_map'
        elif j not in nut_factor_by_comm:
            reason = 'nutrient_factor_missing'
        else:
            item_norm = _normalize_nutrition_item_name(item_nut)
            if (r, item_norm) not in profile_items:
                reason = 'profile_missing_row'
            else:
                if use_baseyear_for_future:
                    if (r, item_norm) not in base_val:
                        reason = 'profile_base_year_missing'
                    elif population_by_country_year is not None:
                        pop = population_by_country_year.get((r, t))
                        if pop is None or pop <= 0:
                            reason = 'population_missing'
                else:
                    if (r, item_norm, t) not in year_val:
                        reason = 'profile_year_missing'
                    elif population_by_country_year is not None:
                        pop = population_by_country_year.get((r, t))
                        if pop is None or pop <= 0:
                            reason = 'population_missing'

        rows.append({
            'M49_Country_Code': r,
            'country_name': m49_to_country.get(r, ''),
            'commodity': j,
            'year': t,
            'reason': reason,
            'item_nutrition_map': item_nut,
        })

    if not rows:
        return None
    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'nutrition_missing_report.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    return str(out_path)


# Linear-model cache class for Monte Carlo simulation


@dataclass
class LinearModelCache:
    """
    Cache the linear regional model for in-place constraint updates during Monte Carlo simulation.
    
    Similar to ModelCache in S3_0_ds_emis_mc_full.py, but for the linear model:
    - No PWL variables (lnQs, lnQd, lnPc, lnPnet) are needed.
    - Constraint coefficients can be updated directly; they are constants in the linear model.
    
    Usage:
    1. cache = build_linear_model_cache(nodes, commodities, years, ...)
    2. apply_linear_sample_updates(cache, pop_mult=..., yield_mult=..., e0_mult=...)
    3. cache.model.optimize()
    4. Read results: cache.Qs[key].X, cache.Qd[key].X, ...
    """
    # Gurobi model
    model: gp.Model
    
    # Decision variables
    Pc: Dict[Tuple[str, int], gp.Var]           # Global price {(commodity, year): var}
    Qs: Dict[Tuple[str, str, int], gp.Var]      # Regional supply {(region, commodity, year): var}
    Qd: Dict[Tuple[str, str, int], gp.Var]      # Regional demand {(region, commodity, year): var}
    supply_curtailment: Dict[Tuple[str, str, int], gp.Var]  # supply curtailment {(region, commodity, year): var}
    net_import: Dict[Tuple[str, str, int], gp.Var]  # Regional net imports {(region, commodity, year): var}
    armington_slack_pos: Dict[Tuple[str, str, int], gp.Var]  # Armington slack +
    armington_slack_neg: Dict[Tuple[str, str, int], gp.Var]  # Armington slack
    Eij: Dict[Tuple[str, str, int], gp.Var]     # Regional emissions {(region, commodity, year): var}
    Cij: Dict[Tuple[str, str, int], gp.Var]     # Regional abatement costs {(region, commodity, year): var}
    excess: Dict[Tuple[str, int], gp.Var]       # Surplus {(commodity, year): var}
    shortage: Dict[Tuple[str, int], gp.Var]     # Shortage {(commodity, year): var}
    
    # Constraint references for in-place RHS updates
    constr_supply: Dict[Tuple[str, str, int], gp.Constr]   # Supply constraints
    constr_demand: Dict[Tuple[str, str, int], gp.Constr]   # Demand constraints
    constr_Edef: Dict[Tuple[str, str, int], gp.Constr]     # Emissions definition constraints
    nutri_constr: Dict[Tuple[str, int], gp.Constr]         # Nutrition constraints
    land_constr: Dict[Tuple[str, int], gp.Constr]          # Land constraints
    rumi_intake_constr: Dict[Tuple[str, int], gp.Constr]   # Ruminant demand cap constraints (Phase 2)
    
    # MACC components
    abatement_vars: Dict[Tuple[str, str, int, str, int], gp.Var]      # Abatement variables
    abatement_caps: Dict[Tuple[str, str, int, str, int], gp.Constr]   # Abatement upper-bound constraints
    abatement_cost_vars: Dict[Tuple[str, str, int, str, int], gp.Var]  # MACC cost-segment variables
    abatement_cost_caps: Dict[Tuple[str, str, int, str, int], gp.Constr]  # Cost-segment upper-bound constraints
    abatement_req_vars: Dict[Tuple[str, str, int, str], gp.Var]  # Baseline abatement relative to BASE
    abatement_req_constr: Dict[Tuple[str, str, int, str], gp.Constr]  # Baseline abatement constraints
    abatement_costs: Dict[Tuple[str, str, int, str, int], float]      # Marginal abatement cost
    proc_cap_basecoeff: Dict[Tuple[str, str, int, str, int], Optional[float]]   # MACC baseline coefficients (None for constant upper bounds)
    
    # Calibration parameters for MC updates
    alpha_s: Dict[Tuple[str, str, int], float]  # Supply intercept
    alpha_d: Dict[Tuple[str, str, int], float]  # Demand intercept
    eps_s: Dict[Tuple[str, str, int], float]    # Supply price elasticity
    eps_d: Dict[Tuple[str, str, int], float]    # Demand price elasticity
    eps_pop: Dict[Tuple[str, str, int], float]  # Population elasticity
    eps_inc: Dict[Tuple[str, str, int], float]  # Income elasticity
    eta_y: Dict[Tuple[str, str, int], float]    # Yield elasticity
    eta_temp: Dict[Tuple[str, str, int], float] # Temperature elasticity
    
    # Baseline values for MC update calculations
    Q0: Dict[Tuple[str, str, int], float]       # Base-period supply
    D0: Dict[Tuple[str, str, int], float]       # Base-period demand
    P0: Dict[Tuple[str, str, int], float]       # Base-period price
    Ymult0: Dict[Tuple[str, str, int], float]   # Base-period yield multiplier
    Tmult0: Dict[Tuple[str, str, int], float]   # Base-period temperature multiplier
    pop_base: Dict[Tuple[str, str, int], float] # Base-period population
    inc_base: Dict[Tuple[str, str, int], float] # Base-period income
    disable_production_cost_term: bool
    production_cost_weight: float
    slack_penalty: float
    armington_trade_slack_penalty: float
    
    # Emissions intensity
    e0_by_region: Dict[Tuple[str, str, int], Dict[str, float]]  # Emissions intensity {key: {process: e0}}
    
    # Metadata
    regions: List[str]
    commodities: List[str]
    years: List[int]
    hist_end_year: int
    idx: Dict[Tuple[str, str, int], Dict]  # Original aggregated-data index
    supply_curtailment_penalty: float = 0.0
    supply_curtailment_enabled: bool = False
    nutrition_import_pos: Optional[Dict[Tuple[str, str, int], gp.Var]] = None
    nutrition_export_pos: Optional[Dict[Tuple[str, str, int], gp.Var]] = None
    nutrition_supply_driven: bool = False
    nutrition_trade_penalty: float = 0.0
    nutrition_export_headroom_penalty: float = 0.0
    nutrition_export_headroom_weight_by_key: Optional[Dict[Tuple[str, str, int], float]] = None
    tax_unit_adder: Optional[Dict[Tuple[str, str, int], float]] = None



# Region definitions loaded from dict_v3 Region_market_agg


# Cache: {country_name: region} and {m49_code: region}
_REGION_BY_COUNTRY: Dict[str, str] = {}
_REGION_BY_M49: Dict[str, str] = {}
_REGIONS_LOADED = False
# Enable regional aggregation: True uses Region_market_agg; False processes countries directly.
_USE_REGIONAL_AGGREGATION = True

def _load_region_mapping_from_dict_v3(dict_v3_path: Optional[str] = None) -> None:
    """
    Load the Region_market_agg mapping from the region sheet in dict_v3.xlsx.
    
    34 market regions:
    - 14 large countries/standalone regions: Argentina, Australia, Bangladesh, Brazil, Canada, China, 
      Ethiopia, India, Indonesia, Mexico, New Zealand, Nigeria, Pakistan, Russia, 
      Tanzania, Turkey, U.S.
    - 20 regional aggregates: AFR-Central, AFR-East, AFR-Southern, AFR-West, 
      AMR-Central-Caribbean, AMR-South, ASIA-Central, ASIA-East, ASIA-South, 
      ASIA-Southeast, EUR-Atlantic, EUR-Boreal, EUR-Continental, EUR-Mediterranean,
      MENA-Gulf, MENA-Mediterranean, OCEA-Pacific
    """
    global _REGION_BY_COUNTRY, _REGION_BY_M49, _REGIONS_LOADED
    
    if _REGIONS_LOADED:
        return
    
    # Determine the dict_v3 path.
    if dict_v3_path is None:
        try:
            from config_paths import get_src_base
            dict_v3_path = str(Path(get_src_base()) / 'dict_v3.xlsx')
        except ImportError:
            dict_v3_path = str((Path(__file__).resolve().parents[2] / 'src' / 'dict_v3.xlsx').resolve())
    
    logger = logging.getLogger(__name__)
    
    try:
        df = pd.read_excel(dict_v3_path, sheet_name='region')
        
        # Use Region_label_new as the country name, consistent with universe.countries.
        # Match M49 codes using M49_Country_Code.
        # Region_market_agg specifies the target region.
        df = df[['Region_label_new', 'M49_Country_Code', 'Region_market_agg']].dropna()
        df = df[df['Region_market_agg'] != 'no']
        df = df[df['Region_label_new'] != 'no']  # Exclude invalid countries as well.
        
        for _, row in df.iterrows():
            country = str(row['Region_label_new']).strip()
            m49 = _norm_m49_code(row['M49_Country_Code'])
            region = str(row['Region_market_agg']).strip()
            if not m49:
                continue
            _REGION_BY_COUNTRY[country] = region
            _REGION_BY_M49[m49] = region
        
        _REGIONS_LOADED = True
        logger.info(f"[REGION] 已从 dict_v3 加载 {len(_REGION_BY_COUNTRY)} 个国家的市场区域映射")
        logger.info(f"[REGION] 共 {len(set(_REGION_BY_COUNTRY.values()))} 个市场区域")
        
    except Exception as e:
        logger.warning(f"[REGION] 无法加载 dict_v3 区域映射: {e}，使用默认区域")
        _REGIONS_LOADED = True  # Prevent repeated attempts.


def get_region(country: str, m49: Optional[str] = None, dict_v3_path: Optional[str] = None) -> str:
    """
    Get the market region corresponding to a country.

    Prefer matching by M49 code, then by country name.
    """
    global _REGIONS_LOADED

    if not _USE_REGIONAL_AGGREGATION:
        return str(country).strip()

    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    
    # Prefer M49 matching.
    if not m49:
        m49 = _norm_m49_code(country)
    if m49:
        m49_clean = _norm_m49_code(m49)
        if m49_clean in _REGION_BY_M49:
            return _REGION_BY_M49[m49_clean]
    
    # Match by country name.
    country_clean = str(country).strip()
    if country_clean in _REGION_BY_COUNTRY:
        return _REGION_BY_COUNTRY[country_clean]
    
    # Default region; this should not occur if dict_v3 includes all countries.
    return 'OTHER'


def get_all_regions(dict_v3_path: Optional[str] = None) -> List[str]:
    """Get the list of all 34 market regions."""
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    
    regions = set(_REGION_BY_COUNTRY.values())
    return sorted(regions)


def reset_region_cache() -> None:
    """Reset the region cache for testing."""
    global _REGION_BY_COUNTRY, _REGION_BY_M49, _REGIONS_LOADED
    _REGION_BY_COUNTRY = {}
    _REGION_BY_M49 = {}
    _REGIONS_LOADED = False


def set_region_aggregation(enabled: bool) -> None:
    """Enable regional aggregation; False processes data at country level."""
    global _USE_REGIONAL_AGGREGATION
    _USE_REGIONAL_AGGREGATION = bool(enabled)


def is_region_aggregation_enabled() -> bool:
    """Return the current regional aggregation setting."""
    return _USE_REGIONAL_AGGREGATION



# Load and aggregate historical maximum production.


def load_historical_max_production(
    csv_path: str,
    dict_v3_path: Optional[str] = None,
    item_column: str = 'Item',
    production_column: str = 'max_production_t',
    m49_column: str = 'M49_Country_Code',
    use_regional_aggregation: Optional[bool] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Load historical maximum production from S0_19_historical_max_production.csv and aggregate by region.
    
    Functionality:
    1. Read the CSV (M49_Country_Code, Item, max_production_t).
    2. Map M49 codes to the 34 market regions (Region_market_agg).
    3. Sum production for each region-commodity pair.
    
    Parameters:
        csv_path: Path to S0_19_historical_max_production.csv.
        dict_v3_path: Path to dict_v3.xlsx, used for regional mapping.
        item_column: Item column name; S0_19 outputs 'Item', containing Item_Production_Map values.
        production_column: Production column name.
        m49_column: M49 code column name.
    
    Returns:
        Dict[(region_or_country, commodity), max_production_t]: historical maximum production by region or country.
    
    Notes:
        - S0_19 outputs Item in Item_Production_Map format, consistent with universe.commodities.
        - Returned commodity keys match the commodity names used in the model.
    """
    logger = logging.getLogger(__name__)
    
    if use_regional_aggregation is None:
        use_regional_aggregation = _USE_REGIONAL_AGGREGATION
    if dict_v3_path is None:
        try:
            from config_paths import get_src_base
            dict_v3_path = str(Path(get_src_base()) / 'dict_v3.xlsx')
        except ImportError:
            dict_v3_path = str((Path(__file__).resolve().parents[2] / 'src' / 'dict_v3.xlsx').resolve())
    # Ensure regional mappings are loaded, as required in regional mode.
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    
    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"[HIST_MAX] 历史最大产量文件不存在: {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(path, dtype={m49_column: str})
    except Exception as e:
        logger.error(f"[HIST_MAX] 无法读取 CSV: {e}")
        return {}
    
    # Check required columns.
    required = [m49_column, item_column, production_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"[HIST_MAX] CSV 缺少必要列: {missing}")
        return {}
    
    # Normalize M49 codes.
    def _norm_m49(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lstrip("'\"")
        try:
            return str(int(s)) if s.isdigit() else s
        except:
            return s
    
    df[m49_column] = df[m49_column].apply(_norm_m49_code)
    df = df[df[m49_column].notna()]
    
    # Map M49 codes to regions or countries.
    group_col = 'region' if use_regional_aggregation else 'country'
    m49_to_country: Dict[str, str] = {}
    if not use_regional_aggregation:
        try:
            dict_v3 = pd.read_excel(dict_v3_path, sheet_name='region')
            for _, row in dict_v3.iterrows():
                m49 = str(row.get('M49_Country_Code', '')).strip().lstrip("'\"")
                country = str(row.get('Region_label_new', '')).strip()
                if m49 and country and country != 'no':
                    m49_to_country[m49] = country
        except Exception:
            m49_to_country = {}
    if use_regional_aggregation:
        df[group_col] = df[m49_column].apply(lambda m: _REGION_BY_M49.get(m, 'OTHER'))
    else:
        df[group_col] = df[m49_column]
    df = df[df[group_col].notna() & (df[group_col] != '') & (df[group_col] != 'OTHER')]
    
    # Ensure production is numeric.
    df[production_column] = pd.to_numeric(df[production_column], errors='coerce').fillna(0.0)
    # FBS production is reported in 1000 t; scale to tons to match model units
    if 'Production_file_source' in df.columns:
        fbs_mask = df['Production_file_source'].astype(str).str.contains('FoodBalanceSheets', case=False, na=False)
        if fbs_mask.any():
            df.loc[fbs_mask, production_column] = df.loc[fbs_mask, production_column] * 1000.0
    df = df[df[production_column] > 0]  # Exclude zero production.
    
    # Aggregate by region or country, summing within each group-commodity pair.
    agg = df.groupby([group_col, item_column])[production_column].sum().reset_index()
    
    # Build the return dictionary.
    result: Dict[Tuple[str, str], float] = {}
    for _, row in agg.iterrows():
        key = (str(row[group_col]), str(row[item_column]))
        result[key] = float(row[production_column])
    
    scope_label = "区域" if use_regional_aggregation else "国家"
    logger.info(f"[HIST_MAX] 已加载 {len(result)} 个 ({scope_label}, 商品) 历史最大产量")
    logger.info(f"[HIST_MAX] 覆盖{scope_label}: {len(set(k[0] for k in result))} 个")
    logger.info(f"[HIST_MAX] 覆盖商品: {len(set(k[1] for k in result))} 种")
    
    return result


def load_grassland_coefficients(
    feed_need_xlsx: str,
    grass_ratio_xlsx: str,
    pasture_yield_xlsx: str,
    dict_v3_path: str,
    years: List[int],
    use_regional_aggregation: Optional[bool] = None,
    feed_requirement_scheme: Optional[str] = None,
    verbose: bool = False,
) -> Dict[Tuple[str, str], float]:
    """
    Calculate livestock-to-grassland conversion coefficients at regional level.
    
    Formula: grassland_coef = (kg_DM_per_head / stock_to_production) * grass_ratio / pasture_yield.
    
    Simplification: assume one livestock head corresponds to one tonne of production (approximation).
    Then grassland_coef is approximately kg_DM_per_head * grass_ratio / pasture_yield_kg_per_ha.
    
    Unit: ha per tonne of production.
    
    Args:
        feed_need_xlsx: Path to the feed requirement file.
        grass_ratio_xlsx: Path to the grassland feed-share file.  
        pasture_yield_xlsx: Path to the pasture yield file.
        dict_v3_path: Path to dict_v3 for country-to-region mapping.
        years: List of years.
        
    Returns:
        Dictionary {(region_or_country, livestock_commodity): ha_per_ton}.
    """
    logger = logging.getLogger(__name__)
    
    # Load parameter data using the loading functions in S3_2.
    try:
        from S3_2_feed_demand import _load_total_dm_per_head, _load_grass_ratio, _load_pasture_yield
        
        dm_per_head = _load_total_dm_per_head(feed_need_xlsx, years, scheme=feed_requirement_scheme)
        grass_ratio_df = _load_grass_ratio(grass_ratio_xlsx)
        pasture_yield_df = _load_pasture_yield(pasture_yield_xlsx)
        
        if dm_per_head.empty or grass_ratio_df.empty or pasture_yield_df.empty:
            if verbose:
                logger.warning("[GRASSLAND_COEF] 参数数据加载失败，返回空系数")
            return {}
        
        # Calculate country-level coefficients.
        # dm_per_head: [species, m49_code, year, kg_dm_per_head]
        # grass_ratio: [species, m49_code, grass_ratio]
        # pasture_yield: [m49_code, pasture_yield_kg_per_ha]
        
        # Merge parameters.
        coef_df = dm_per_head.merge(grass_ratio_df, on=['species', 'm49_code'], how='left')
        coef_df = coef_df.merge(pasture_yield_df, on='m49_code', how='left')
        
        # Fill missing values.
        coef_df['grass_ratio'] = coef_df['grass_ratio'].fillna(0.0).clip(0, 1)
        coef_df['pasture_yield_kg_per_ha'] = coef_df['pasture_yield_kg_per_ha'].fillna(3000.0)  # Default: 3000 kg/ha.
        
        # Calculate ha_per_head = (kg_DM_per_head * grass_ratio) / pasture_yield.
        coef_df['ha_per_head'] = (
            coef_df['kg_dm_per_head'] * coef_df['grass_ratio'] / 
            coef_df['pasture_yield_kg_per_ha'].clip(lower=1e-6)
        )
        
        # Aggregate to region or country level; production weighting is simplified here to an arithmetic mean.
        # An m49_code-to-country/region mapping is required.
        from S2_0_load_data import DataPaths
        paths = DataPaths()
        dict_v3 = pd.read_excel(dict_v3_path, sheet_name='region')
        m49_to_region = {}
        m49_to_country = {}
        for _, row in dict_v3.iterrows():
            m49 = _norm_m49_code(row.get('M49_Country_Code', ''))
            country = str(row.get('Region_label_new', '')).strip()
            region_market = str(row.get('Region_market_agg', '')).strip()
            if m49 and region_market and region_market != 'no':
                m49_to_region[m49] = region_market
            if m49 and country and country != 'no':
                m49_to_country[m49] = country

        if use_regional_aggregation is None:
            use_regional_aggregation = _USE_REGIONAL_AGGREGATION

        if use_regional_aggregation:
            coef_df['region'] = coef_df['m49_code'].map(m49_to_region)
            coef_df = coef_df.dropna(subset=['region'])
            grouped = coef_df.groupby(['region', 'species'])['ha_per_head'].mean().reset_index()
            key_cols = ('region', 'species')
        else:
            coef_df['country'] = coef_df['m49_code']
            coef_df = coef_df.dropna(subset=['country'])
            grouped = coef_df.groupby(['country', 'species'])['ha_per_head'].mean().reset_index()
            key_cols = ('country', 'species')

        # Build the return dictionary {(region/country, species): ha_per_head}.
        result = {}
        for _, row in grouped.iterrows():
            key = (str(row[key_cols[0]]), str(row[key_cols[1]]))
            result[key] = float(row['ha_per_head'])
        
        if verbose:
            logger.info(f"[GRASSLAND_COEF] 已计算 {len(result)} 个 (区域, livestock) 草地系数")
            logger.info(f"[GRASSLAND_COEF] 覆盖区域: {len(set(k[0] for k in result))} 个")
            logger.info(f"[GRASSLAND_COEF] 覆盖livestock: {len(set(k[1] for k in result))} 种")
        
        # Print a few examples.
        sample_items = ['Cattle', 'Pigs', 'Poultry', 'Sheep', 'Goats']
        sample_regions = ['China', 'U.S.', 'India', 'Brazil']
        if verbose:
            logger.info(f"[GRASSLAND_COEF] 系数示例（ha/head）:")
            for region in sample_regions:
                for item in sample_items:
                    coef = result.get((region, item))
                    if coef:
                        logger.info(f"  {region:20s} | {item:15s}: {coef:.6f} ha/head")
        
        return result
        
    except Exception as e:
        if verbose:
            logger.error(f"[GRASSLAND_COEF] 计算grassland系数失败: {e}")
            import traceback
            traceback.print_exc()
        return {}



# Data aggregation


# Livestock commodities requiring grassland
LIVESTOCK_COMMODITIES = {
    'Cattle', 'Buffaloes', 'Sheep', 'Goats', 'Pigs', 'Chickens', 
    'Ducks', 'Geese', 'Turkeys', 'Horses', 'Asses', 'Mules',
    'Milk', 'Eggs', 'Meat'
}

def aggregate_nodes_to_regions(
    nodes: List[Any], 
    dict_v3_path: Optional[str] = None,
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    hist_end_year: int = 2020,
) -> pd.DataFrame:
    """
    Aggregate country-level nodes to regional level.
    
    Aggregation rules:
    - Q0, D0: sum.
    - P0: Q0-weighted average.
    - Elasticities: Q0/D0-weighted average.
    - Ymult, Tmult: Q0-weighted average.
    - Population and income: regional sums.
    
    Use the 34 market regions in dict_v3 Region_market_agg.
    
    Complete elasticity list, consistent with S3_0_ds_emis_mc_full.py:
    - Supply: eps_supply, eps_supply_yield (eta_y), eps_supply_temp (eta_temp).
    - Demand: eps_demand, eps_pop_demand, eps_income_demand.
    - Cross-price: epsS_row, epsD_row; not aggregated yet, with simplified handling in the regional model.
    """
    # Ensure regional mappings are loaded.
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)

    # Base-year P0 by country-commodity for aggregate_nodes_to_regions.
    base_p0_by_key: Dict[Tuple[str, str], float] = {}
    for n in nodes:
        if getattr(n, 'year', None) != hist_end_year:
            continue
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        key_country = str(m49).strip() if m49 is not None else str(getattr(n, 'country', '')).strip()
        if not key_country:
            continue
        try:
            p0_val = float(getattr(n, 'P0', 0.0) or 0.0)
        except Exception:
            continue
        if np.isfinite(p0_val) and p0_val > 0:
            base_p0_by_key[(key_country, getattr(n, 'commodity', None))] = p0_val
    
    records = []
    for n in nodes:
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        region = get_region(n.country, m49=m49, dict_v3_path=dict_v3_path)
        key_country = str(m49).strip() if m49 is not None else str(getattr(n, 'country', '')).strip()
        
        # Get population and income data.
        pop_base = 1.0
        pop_t = 1.0
        inc_base = 1.0
        inc_t = 1.0
        if population_by_country_year:
            pop_base = float(population_by_country_year.get((n.country, hist_end_year), 1.0) or 1.0)
            pop_t = float(population_by_country_year.get((n.country, n.year), pop_base) or pop_base)
        if income_mult_by_country_year:
            inc_base = float(income_mult_by_country_year.get((n.country, hist_end_year), 1.0) or 1.0)
            inc_t = float(income_mult_by_country_year.get((n.country, n.year), inc_base) or inc_base)
        
        # Get yield0, the historical average yield, from meta.
        meta = getattr(n, 'meta', {}) or {}
        yield0 = float(meta.get('yield0', 0.0) or 0.0)
        
        # Get grassland coefficients for livestock commodities only, used in approach A.
        # grassland_coef = ha_per_ton: convert production tonnes to grassland hectares.
        # Check meta directly for grassland_coef instead of using the old LIVESTOCK_COMMODITIES membership test.
        # Reason: LIVESTOCK_COMMODITIES={'Cattle', ...} does not match the actual commodity='Cattle, dairy'.
        grassland_coef = float(meta.get('grassland_coef', 0.0) or 0.0)

        p0_val = getattr(n, 'P0', 0.0) or 0.0
        if getattr(n, 'year', None) > hist_end_year:
            base_p0 = base_p0_by_key.get((key_country, getattr(n, 'commodity', None)))
            if base_p0 is not None:
                p0_val = base_p0
        try:
            p0_val = float(p0_val)
        except Exception:
            p0_val = 0.0
        if not np.isfinite(p0_val) or p0_val <= 0:
            base_p0 = base_p0_by_key.get((key_country, getattr(n, 'commodity', None)))
            if base_p0 is not None:
                p0_val = base_p0
            else:
                p0_val = 1.0
        
        records.append({
            'region': region,
            'country': n.country,
            'commodity': n.commodity,
            'year': n.year,
            # Basic quantities
            'Q0': getattr(n, 'Q0', 0.0) or 0.0,
            'D0': getattr(n, 'D0', 0.0) or 0.0,
            'P0': p0_val,
            # Yield for land constraints
            'yield0': yield0,
            'grassland_coef': grassland_coef,
                'livestock_feed_account': bool(meta.get('livestock_feed_account', False)),
                'feed_only_entry': bool(meta.get('feed_only_entry', False)),
                'livestock_background_only': bool(meta.get('livestock_background_only', False)),
                'livestock_feed_unidentified': bool(meta.get('livestock_feed_unidentified', False)),
                'seed_coefficient': float(meta.get('seed_coefficient', 0.0)),
                'seed_reference_t': float(meta.get('seed_reference_t', 0.0)),
                'seed_coefficient_unidentified': bool(meta.get('seed_coefficient_unidentified', False)),
            # Supply elasticities
            'eps_supply': getattr(n, 'eps_supply', 0.0) or 0.0,
            'eps_supply_yield': getattr(n, 'eps_supply_yield', 0.0) or 0.0,  # η_y
            'eps_supply_temp': getattr(n, 'eps_supply_temp', 0.0) or 0.0,    # η_temp
            # Supply factors
            'Ymult': getattr(n, 'Ymult', 1.0) or 1.0,  # Yield multiplier
            'Tmult': getattr(n, 'Tmult', 1.0) or 1.0,  # Temperature multiplier
            # Demand elasticities
            'eps_demand': getattr(n, 'eps_demand', 0.0) or 0.0,
            'eps_pop_demand': getattr(n, 'eps_pop_demand', 0.0) or 0.0,
            'eps_income_demand': getattr(n, 'eps_income_demand', 0.0) or 0.0,
            # Cross-price elasticities: preserve original dictionaries and use Q0/D0 weights during aggregation.
            'epsS_row': dict(getattr(n, 'epsS_row', {}) or {}),  # Supply-side cross-price elasticities
            'epsD_row': dict(getattr(n, 'epsD_row', {}) or {}),  # Demand-side cross-price elasticities
            # Population and income for the demand equation
            'pop_base': pop_base,
            'pop_t': pop_t,
            'inc_base': inc_base,
            'inc_t': inc_t,
        })
    
    df = pd.DataFrame(records)
    
    # Aggregate by region-commodity-year.
    agg_funcs = {
        # Sum quantities.
        'Q0': 'sum',
        'D0': 'sum',
        # Weighted average price
        'P0': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        # Yield: Q0-weighted average
        'yield0': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)) if x.sum() > 0 else 0.0,
        # Grassland coefficients: Q0-weighted average, populated only for livestock commodities
        'grassland_coef': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)) if x.sum() > 0 else 0.0,
        'livestock_feed_account': 'max', 'feed_only_entry': 'max',
        'livestock_background_only': 'min',
        'livestock_feed_unidentified': 'min',
        'seed_coefficient': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        'seed_reference_t': 'sum',
        'seed_coefficient_unidentified': 'max',
        # Supply elasticities weighted by Q0
        'eps_supply': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        'eps_supply_yield': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        'eps_supply_temp': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        # Supply factors weighted by Q0
        'Ymult': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        'Tmult': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        # Demand elasticities weighted by D0
        'eps_demand': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),
        'eps_pop_demand': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),
        'eps_income_demand': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),
        # Population and income totals
        'pop_base': 'sum',
        'pop_t': 'sum',
        'inc_base': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),  # Use a weighted average for income.
        'inc_t': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),
    }
    
    regional_df = df.groupby(['region', 'commodity', 'year']).agg(agg_funcs).reset_index()

    # Force future-year P0 to base-year (regional) when available.
    if int(hist_end_year) in set(pd.to_numeric(regional_df['year'], errors='coerce').dropna().astype(int).unique()):
        base_p0 = regional_df[regional_df['year'] == hist_end_year][['region', 'commodity', 'P0']].rename(columns={'P0': 'P0_base'})
        regional_df = regional_df.merge(base_p0, on=['region', 'commodity'], how='left')
        mask_future = pd.to_numeric(regional_df['year'], errors='coerce') > hist_end_year
        has_base = regional_df['P0_base'].notna()
        regional_df.loc[mask_future & has_base, 'P0'] = regional_df.loc[mask_future & has_base, 'P0_base']
        regional_df.drop(columns=['P0_base'], inplace=True)
    
    # Aggregate cross-price elasticities separately, weighting each commodity's elasticity by Q0/D0.
    cross_price_data: Dict[Tuple[str, str, int], Dict[str, Dict[str, float]]] = {}
    for _, row in df.iterrows():
        key = (row['region'], row['commodity'], row['year'])
        if key not in cross_price_data:
            cross_price_data[key] = {'epsS_sum': {}, 'epsD_sum': {}, 'Q0_total': 0.0, 'D0_total': 0.0}
        
        Q0 = max(1e-9, float(row['Q0']))
        D0 = max(1e-9, float(row['D0']))
        cross_price_data[key]['Q0_total'] += Q0
        cross_price_data[key]['D0_total'] += D0
        
        # Supply-side cross-price elasticities weighted by Q0
        for comm, eps_val in row.get('epsS_row', {}).items():
            if comm not in cross_price_data[key]['epsS_sum']:
                cross_price_data[key]['epsS_sum'][comm] = 0.0
            cross_price_data[key]['epsS_sum'][comm] += float(eps_val) * Q0
        
        # Demand-side cross-price elasticities weighted by D0
        for comm, eps_val in row.get('epsD_row', {}).items():
            if comm not in cross_price_data[key]['epsD_sum']:
                cross_price_data[key]['epsD_sum'][comm] = 0.0
            cross_price_data[key]['epsD_sum'][comm] += float(eps_val) * D0
    
    # Calculate weighted averages and add them to regional_df.
    epsS_row_col = []
    epsD_row_col = []
    for _, row in regional_df.iterrows():
        key = (row['region'], row['commodity'], row['year'])
        cpd = cross_price_data.get(key, {})
        
        Q0_total = max(1e-9, cpd.get('Q0_total', 1.0))
        D0_total = max(1e-9, cpd.get('D0_total', 1.0))
        
        epsS_avg = {c: v / Q0_total for c, v in cpd.get('epsS_sum', {}).items()}
        epsD_avg = {c: v / D0_total for c, v in cpd.get('epsD_sum', {}).items()}
        
        epsS_row_col.append(epsS_avg)
        epsD_row_col.append(epsD_avg)
    
    regional_df['epsS_row'] = epsS_row_col
    regional_df['epsD_row'] = epsD_row_col
    
    return regional_df


def aggregate_emissions_to_regions(
    nodes: List[Any], 
    dict_v3_path: Optional[str] = None
) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """
    Aggregate country-level emissions intensities e0_by_proc to regional level.
    
    Aggregation rule: Q0-weighted average.
    
    Returns: {(region, commodity, year): {process: e0_intensity}}.
    """
    # Ensure regional mappings are loaded.
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    
    # Collect (region, commodity, year) -> [(e0_by_proc, Q0), ...].
    data_by_key: Dict[Tuple[str, str, int], List[Tuple[Dict[str, float], float]]] = {}
    
    for n in nodes:
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        region = get_region(n.country, m49=m49, dict_v3_path=dict_v3_path)
        key = (region, n.commodity, n.year)
        
        e0_map = getattr(n, 'e0_by_proc', {}) or {}
        Q0 = float(getattr(n, 'Q0', 0.0) or 0.0)
        
        if key not in data_by_key:
            data_by_key[key] = []
        data_by_key[key].append((dict(e0_map), Q0))
    
    # Weighted average
    result: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for key, entries in data_by_key.items():
        all_procs = set()
        for e0_map, _ in entries:
            all_procs.update(e0_map.keys())
        
        total_Q0 = sum(Q0 for _, Q0 in entries)
        use_equal_weights = total_Q0 < 1e-9
        denom = float(len(entries)) if use_equal_weights else float(total_Q0)
        
        agg_e0: Dict[str, float] = {}
        for proc in all_procs:
            if use_equal_weights:
                weighted_sum = sum(float(e0_map.get(proc, 0.0) or 0.0) for e0_map, _ in entries)
            else:
                weighted_sum = sum(float(e0_map.get(proc, 0.0) or 0.0) * Q0 for e0_map, Q0 in entries)
            agg_e0[proc] = weighted_sum / denom if denom > 0 else 0.0
        
        result[key] = agg_e0
    
    return result


def _scale_cross_terms(
    eps_row: Dict[str, float],
    scale: Optional[float],
) -> Dict[str, float]:
    """Scale cross-price elasticities by a factor; None or 1 keeps original."""
    if not eps_row:
        return {}
    cleaned: Dict[str, float] = {}
    for k, v in eps_row.items():
        try:
            cleaned[k] = float(v)
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return {}
    if scale is None:
        return cleaned
    try:
        s = float(scale)
    except (TypeError, ValueError):
        return cleaned
    if s == 1.0:
        return cleaned
    if s == 0.0:
        return {}
    return {k: v * s for k, v in cleaned.items()}

CROSS_COEF_RATIO_TOL = 10.0
CROSS_COEF_FIX_LOG_LIMIT = 20


def _sanitize_cross_coef(
    b_cross: float,
    *,
    base_qty: float,
    cross_eps: float,
    use_relative_price: bool,
    ratio_tol: float,
    logger: Optional[logging.Logger] = None,
    log_state: Optional[Dict[str, Any]] = None,
    region: Any = None,
    commodity: Any = None,
    year: Any = None,
    other_comm: Any = None,
    tag: str = '',
) -> float:
    if not use_relative_price:
        return b_cross
    try:
        base = float(base_qty)
        eps = float(cross_eps)
    except Exception:
        return b_cross
    ref = base * eps
    if not np.isfinite(ref) or abs(ref) < 1e-12:
        if not np.isfinite(b_cross) or abs(b_cross) > 1e-6:
            if logger and log_state is not None:
                cnt = int(log_state.get('count', 0) or 0) + 1
                log_state['count'] = cnt
                if cnt <= int(log_state.get('limit', CROSS_COEF_FIX_LOG_LIMIT) or CROSS_COEF_FIX_LOG_LIMIT):
                    logger.warning(
                        "[LINEAR] cross coef override(%s) r=%s j=%s t=%s other=%s b=%.6g ref=0 ratio=inf",
                        tag, region, commodity, year, other_comm, b_cross,
                    )
            return 0.0
        return b_cross
    ratio = abs(b_cross) / abs(ref)
    if not np.isfinite(ratio) or ratio > ratio_tol or ratio < 1.0 / ratio_tol:
        if logger and log_state is not None:
            cnt = int(log_state.get('count', 0) or 0) + 1
            log_state['count'] = cnt
            if cnt <= int(log_state.get('limit', CROSS_COEF_FIX_LOG_LIMIT) or CROSS_COEF_FIX_LOG_LIMIT):
                logger.warning(
                    "[LINEAR] cross coef override(%s) r=%s j=%s t=%s other=%s b=%.6g ref=%.6g ratio=%.3g",
                    tag, region, commodity, year, other_comm, b_cross, ref, ratio,
                )
        return ref
    return b_cross


def _compute_price_ref_by_comm(
    idx: Dict[Tuple[str, str, int], Dict[str, Any]],
    commodities: List[str],
    regions: List[str],
    hist_end_year: int,
) -> Dict[str, float]:
    """Compute base price reference per commodity (weighted by Q0 at hist_end_year)."""
    ref: Dict[str, float] = {}
    for j in commodities:
        w_sum = 0.0
        wp_sum = 0.0
        p_list: List[float] = []
        for r in regions:
            key = (r, j, hist_end_year)
            data = idx.get(key)
            if not data:
                continue
            p0 = float(data.get('P0', 0.0) or 0.0)
            if not np.isfinite(p0) or p0 <= 0:
                continue
            q0 = float(data.get('Q0', 0.0) or 0.0)
            if q0 > 0:
                w_sum += q0
                wp_sum += q0 * p0
            else:
                p_list.append(p0)
        if w_sum > 0:
            ref[j] = wp_sum / w_sum
        elif p_list:
            ref[j] = float(np.mean(p_list))
        else:
            ref[j] = 1.0
    return ref


def _normalize_price_bounds_mode(raw: Optional[str]) -> str:
    if raw is None:
        return 'absolute'
    val = str(raw).strip().lower()
    if val in {'p0', 'p0_mult', 'p0-based', 'p0_based', 'p0mult'}:
        return 'p0_mult'
    return 'absolute'


def _normalize_positive_bounds(
    raw: Optional[Tuple[float, float]],
    default: Tuple[float, float],
) -> Tuple[float, float]:
    try:
        lo, hi = raw
    except Exception:
        return default
    try:
        lo = float(lo)
        hi = float(hi)
    except Exception:
        return default
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= lo:
        return default
    return lo, hi


def _normalize_nonnegative_bounds(
    raw: Optional[Tuple[float, float]],
    default: Tuple[float, float],
) -> Tuple[float, float]:
    try:
        lo, hi = raw
    except Exception:
        return default
    try:
        lo = float(lo)
        hi = float(hi)
    except Exception:
        return default
    if not np.isfinite(lo) or not np.isfinite(hi) or lo < 0 or hi <= lo:
        return default
    return lo, hi


def _compute_price_bounds_by_comm(
    *,
    idx: Dict[Tuple[str, str, int], Dict[str, Any]],
    commodities: List[str],
    regions: List[str],
    hist_end_year: int,
    price_bounds: Tuple[float, float],
    use_relative_price: bool,
    relative_price_bounds: Tuple[float, float],
    price_bounds_mode: Optional[str],
    price_bounds_p0_mult: Optional[Tuple[float, float]],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float], Tuple[float, float], str]:
    bounds_by_comm: Dict[str, Tuple[float, float]] = {}
    price_ref_by_comm: Dict[str, float] = {}
    abs_min, abs_max = _normalize_positive_bounds(price_bounds, (1e-6, 1e6))
    mode = _normalize_price_bounds_mode(price_bounds_mode)

    if use_relative_price:
        rel_min, rel_max = _normalize_positive_bounds(relative_price_bounds, (0.1, 10.0))
        price_ref_by_comm = _compute_price_ref_by_comm(idx, commodities, regions, hist_end_year)
        for j in commodities:
            ref = price_ref_by_comm.get(j, 1.0)
            if ref is None or not np.isfinite(ref) or ref <= 0:
                price_ref_by_comm[j] = 1.0
            bounds_by_comm[j] = (rel_min, rel_max)
        return bounds_by_comm, price_ref_by_comm, (rel_min, rel_max), mode

    if mode == 'p0_mult':
        mult_min, mult_max = _normalize_nonnegative_bounds(price_bounds_p0_mult, (0.1, 10.0))
        price_ref_by_comm = _compute_price_ref_by_comm(idx, commodities, regions, hist_end_year)
        for j in commodities:
            ref = float(price_ref_by_comm.get(j, 1.0) or 1.0)
            if not np.isfinite(ref) or ref <= 0:
                ref = 1.0
            pmin = ref * mult_min
            pmax = ref * mult_max
            if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin < 0 or pmax <= pmin:
                pmin, pmax = abs_min, abs_max
            bounds_by_comm[j] = (pmin, pmax)
        global_min = min(v[0] for v in bounds_by_comm.values()) if bounds_by_comm else abs_min
        global_max = max(v[1] for v in bounds_by_comm.values()) if bounds_by_comm else abs_max
        return bounds_by_comm, price_ref_by_comm, (global_min, global_max), mode

    for j in commodities:
        bounds_by_comm[j] = (abs_min, abs_max)
    return bounds_by_comm, price_ref_by_comm, (abs_min, abs_max), mode


def _normalize_price_wedge_rcy(
    raw: Optional[Dict[Tuple[Any, Any, Any], Any]],
) -> Dict[Tuple[str, str, int], float]:
    if not raw:
        return {}
    out: Dict[Tuple[str, str, int], float] = {}
    for key, val in raw.items():
        try:
            r, j, t = key
        except Exception:
            continue
        try:
            t_int = int(t)
        except Exception:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        out[(str(r), str(j), t_int)] = v
    return out


def _normalize_price_wedge_rc(
    raw: Optional[Dict[Tuple[Any, Any], Any]],
) -> Dict[Tuple[str, str], float]:
    if not raw:
        return {}
    out: Dict[Tuple[str, str], float] = {}
    for key, val in raw.items():
        try:
            r, j = key
        except Exception:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        out[(str(r), str(j))] = v
    return out


def _normalize_price_wedge_r(
    raw: Optional[Dict[Any, Any]],
) -> Dict[str, float]:
    if not raw:
        return {}
    out: Dict[str, float] = {}
    for r, val in raw.items():
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        out[str(r)] = v
    return out


def _get_price_wedge(
    region: Any,
    commodity: Any,
    year: Any,
    *,
    by_rcy: Optional[Dict[Tuple[str, str, int], float]] = None,
    by_rc: Optional[Dict[Tuple[str, str], float]] = None,
    by_r: Optional[Dict[str, float]] = None,
) -> float:
    r_key = str(region)
    j_key = str(commodity)
    try:
        t_key = int(year)
    except Exception:
        t_key = year
    if by_rcy:
        val = by_rcy.get((r_key, j_key, t_key))
        if val is not None and np.isfinite(val):
            return float(val)
    if by_rc:
        val = by_rc.get((r_key, j_key))
        if val is not None and np.isfinite(val):
            return float(val)
    if by_r:
        val = by_r.get(r_key)
        if val is not None and np.isfinite(val):
            return float(val)
    return 0.0


def _limit_cross_terms(
    eps_row: Dict[str, float],
    top_n: Optional[int],
) -> Dict[str, float]:
    """Keep top-N cross-price elasticities by abs value; None keeps all."""
    if not eps_row:
        return {}
    cleaned: Dict[str, float] = {}
    for k, v in eps_row.items():
        try:
            cleaned[k] = float(v)
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return {}
    if top_n is None:
        return cleaned
    try:
        n = int(top_n)
    except (TypeError, ValueError):
        return cleaned
    if n <= 0:
        return {}
    if len(cleaned) <= n:
        return cleaned
    items = sorted(cleaned.items(), key=lambda kv: (-abs(kv[1]), str(kv[0])))
    return dict(items[:n])


def _normalize_cross_eps(
    eps_row: Dict[str, float],
    eps_d: float,
) -> Tuple[Dict[str, float], float, float]:
    """Scale demand cross-price elasticities so sum <= 1 - eps_d."""
    if not eps_row:
        return {}, 0.0, 1.0
    sum_cross = sum(float(v) for v in eps_row.values())
    max_sum = max(0.0, 1.0 - float(eps_d))
    if sum_cross > max_sum and sum_cross > 0:
        scale = max_sum / sum_cross
        scaled = {k: float(v) * scale for k, v in eps_row.items()}
        return scaled, max_sum, scale
    return {k: float(v) for k, v in eps_row.items()}, sum_cross, 1.0


def _write_eps_d_distribution(
    eps_d_map: Dict[Tuple[str, str, int], float],
    output_dir: Optional[str],
    hist_end_year: Optional[int] = None,
) -> Optional[Path]:
    if not output_dir:
        return None
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    out_path = out_dir / "eps_d_distribution.log"

    def _format_stats(label: str, arr: np.ndarray) -> List[str]:
        n = int(arr.size)
        if n == 0:
            return [f"{label}: count=0"]
        neg = int((arr < -1e-12).sum())
        pos = int((arr > 1e-12).sum())
        zero = n - neg - pos
        qs = np.quantile(arr, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        mean = float(arr.mean())
        return [
            (
                f"{label}: count={n} neg={neg} zero={zero} pos={pos} "
                f"min={qs[0]:.6g} p05={qs[1]:.6g} p25={qs[2]:.6g} "
                f"p50={qs[3]:.6g} mean={mean:.6g} p75={qs[4]:.6g} "
                f"p95={qs[5]:.6g} max={qs[6]:.6g}"
            )
        ]

    try:
        values = []
        year_values: Dict[int, List[float]] = {}
        for (r, j, t), v in eps_d_map.items():
            if v is None:
                continue
            try:
                val = float(v)
            except Exception:
                continue
            if math.isnan(val):
                continue
            values.append(val)
            year_values.setdefault(int(t), []).append(val)

        lines: List[str] = []
        lines.append("eps_d distribution summary")
        lines.append(f"total_keys={len(eps_d_map)}")
        arr_all = np.array(values, dtype=float) if values else np.array([], dtype=float)
        lines.extend(_format_stats("all", arr_all))
        if hist_end_year is not None:
            hist_vals = [v for (r, j, t), v in eps_d_map.items() if int(t) <= int(hist_end_year)]
            fut_vals = [v for (r, j, t), v in eps_d_map.items() if int(t) > int(hist_end_year)]
            arr_hist = np.array(hist_vals, dtype=float) if hist_vals else np.array([], dtype=float)
            arr_fut = np.array(fut_vals, dtype=float) if fut_vals else np.array([], dtype=float)
            lines.extend(_format_stats(f"history(t<={hist_end_year})", arr_hist))
            lines.extend(_format_stats(f"future(t>{hist_end_year})", arr_fut))
        if year_values:
            lines.append("by_year")
            for t in sorted(year_values.keys()):
                arr_t = np.array(year_values[t], dtype=float)
                lines.extend(_format_stats(f"year={t}", arr_t))

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path
    except Exception:
        return None


def _init_pc_range_record(pmin: float, pmax: float) -> Dict[str, Any]:
    return {
        'pc_lb': float(pmin),
        'pc_ub': float(pmax),
        'lb_src': None,
        'ub_src': None,
        'min_width': float('inf'),
        'min_width_src': None,
        'n_constraints': 0,
        'n_bounds': 0,
        'n_infeasible': 0,
        'n_unbounded': 0,
    }


def _update_pc_range_diag(
    diag: Dict[Tuple[str, int], Dict[str, Any]],
    *,
    commodity: str,
    year: int,
    region: str,
    constr_type: str,
    constr_name: str,
    a: float,
    b: float,
    cross_min: float,
    cross_max: float,
    q_lb: float,
    q_ub: float,
    pmin: float,
    pmax: float,
) -> None:
    info = diag.setdefault((commodity, int(year)), _init_pc_range_record(pmin, pmax))
    info['n_constraints'] += 1

    try:
        b_val = float(b)
        a_val = float(a)
        c_min = float(cross_min)
        c_max = float(cross_max)
    except Exception:
        info['n_infeasible'] += 1
        return

    if not np.isfinite(b_val) or abs(b_val) < 1e-12:
        q_min = a_val + c_min
        q_max = a_val + c_max
        if q_lb <= q_max and q_ub >= q_min:
            info['n_unbounded'] += 1
        else:
            info['n_infeasible'] += 1
        return

    if c_min > c_max:
        c_min, c_max = c_max, c_min
    if q_lb > q_ub:
        q_lb, q_ub = q_ub, q_lb

    if b_val > 0:
        lb = (q_lb - a_val - c_max) / b_val
        ub = (q_ub - a_val - c_min) / b_val
    else:
        lb = (q_ub - a_val - c_min) / b_val
        ub = (q_lb - a_val - c_max) / b_val

    if lb > ub:
        info['n_infeasible'] += 1
        return

    lb = max(lb, pmin)
    ub = min(ub, pmax)
    if lb > ub:
        info['n_infeasible'] += 1
        return

    info['n_bounds'] += 1
    src = (constr_type, region, commodity, constr_name)
    if lb > info['pc_lb'] + 1e-12:
        info['pc_lb'] = lb
        info['lb_src'] = src
    if ub < info['pc_ub'] - 1e-12:
        info['pc_ub'] = ub
        info['ub_src'] = src
    width = ub - lb
    if width < info['min_width']:
        info['min_width'] = width
        info['min_width_src'] = src


def _write_pc_range_diagnosis(
    diag: Dict[Tuple[str, int], Dict[str, Any]],
    output_dir: Optional[str],
    *,
    use_relative_price: bool,
    price_ref_by_comm: Optional[Dict[str, float]] = None,
) -> Optional[Path]:
    if not output_dir or not diag:
        return None
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    rows: List[Dict[str, Any]] = []
    price_ref_by_comm = price_ref_by_comm or {}
    for (j, t), info in sorted(diag.items(), key=lambda x: (x[0][0], x[0][1])):
        lb = float(info.get('pc_lb', 0.0))
        ub = float(info.get('pc_ub', 0.0))
        feasible = (info.get('n_infeasible', 0) == 0) and (lb <= ub)

        lb_src = info.get('lb_src') or (None, None, None, None)
        ub_src = info.get('ub_src') or (None, None, None, None)
        min_src = info.get('min_width_src') or (None, None, None, None)
        min_width = info.get('min_width')
        if min_width is not None and not np.isfinite(min_width):
            min_width = None

        p0_ref = None
        pc_lb_abs = None
        pc_ub_abs = None
        pc_width_abs = None
        if use_relative_price:
            p0_ref = float(price_ref_by_comm.get(j, 1.0) or 1.0)
            pc_lb_abs = lb * p0_ref
            pc_ub_abs = ub * p0_ref
            pc_width_abs = pc_ub_abs - pc_lb_abs

        rows.append({
            'commodity': j,
            'year': int(t),
            'pc_lb': lb,
            'pc_ub': ub,
            'pc_width': ub - lb,
            'lb_src_type': lb_src[0],
            'lb_src_region': lb_src[1],
            'lb_src_item': lb_src[2],
            'lb_src_constraint': lb_src[3],
            'ub_src_type': ub_src[0],
            'ub_src_region': ub_src[1],
            'ub_src_item': ub_src[2],
            'ub_src_constraint': ub_src[3],
            'min_constraint_width': min_width,
            'min_width_src_type': min_src[0],
            'min_width_src_region': min_src[1],
            'min_width_src_item': min_src[2],
            'min_width_src_constraint': min_src[3],
            'n_constraints': info.get('n_constraints', 0),
            'n_bounds': info.get('n_bounds', 0),
            'n_infeasible': info.get('n_infeasible', 0),
            'n_unbounded': info.get('n_unbounded', 0),
            'feasible': bool(feasible),
            'use_relative_price': bool(use_relative_price),
            'p0_ref': p0_ref,
            'pc_lb_abs': pc_lb_abs,
            'pc_ub_abs': pc_ub_abs,
            'pc_width_abs': pc_width_abs,
        })

    out_path = out_dir / "pc_range_diagnosis.csv"
    try:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        return None
    return out_path


def _write_nutrition_feasibility_diagnosis(
    cache: Dict[str, Any],
    output_dir: Optional[str],
    *,
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]],
    nutrient_per_unit_by_comm: Optional[Dict[str, float]],
    feed_reduction_by: Optional[Dict[Tuple[str, str, int], float]],
    nutrition_demand_map: Optional[Dict[Tuple[str, str, int], float]] = None,
    hist_end_year: int,
) -> Optional[Path]:
    if not output_dir or not nutrition_rhs or not nutrient_per_unit_by_comm:
        return None
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    idx = cache.get('idx', {}) or {}
    commodities = cache.get('commodities', []) or []
    price_bounds = cache.get('price_bounds', (1e-6, 1e6))
    Pmin, Pmax = float(price_bounds[0]), float(price_bounds[1])
    price_bounds_by_comm = cache.get('price_bounds_by_comm', {}) or {}
    def _p_bounds(comm: str) -> Tuple[float, float]:
        return price_bounds_by_comm.get(comm, (Pmin, Pmax))
    use_relative_price = bool(cache.get('use_relative_price'))
    price_ref_by_comm = cache.get('price_ref_by_comm', {}) or {}
    cross_terms_top_n = cache.get('cross_terms_top_n')
    cross_terms_scale = cache.get('cross_terms_scale')
    alpha_d_cache = cache.get('alpha_d', {}) or {}
    pop_base_cache = cache.get('pop_base', {}) or {}
    inc_base_cache = cache.get('inc_base', {}) or {}
    qty_scale = float(cache.get('qty_scale', 1.0) or 1.0)
    price_wedge_by_region_comm_year = cache.get('price_wedge_by_region_comm_year', {}) or {}
    price_wedge_by_region_comm = cache.get('price_wedge_by_region_comm', {}) or {}
    price_wedge_by_region = cache.get('price_wedge_by_region', {}) or {}

    def _wedge(region: Any, comm: Any, year: Any) -> float:
        return _get_price_wedge(
            region,
            comm,
            year,
            by_rcy=price_wedge_by_region_comm_year,
            by_rc=price_wedge_by_region_comm,
            by_r=price_wedge_by_region,
        )

    demand_map = nutrition_demand_map or cache.get('nutrition_demand_map', {}) or {}
    if demand_map:
        totals: Dict[Tuple[str, int], float] = defaultdict(float)
        top_by_rt: Dict[Tuple[str, int], List[Tuple[float, str]]] = defaultdict(list)
        for (r, j, t), qty in demand_map.items():
            try:
                t_val = int(t)
            except Exception:
                continue
            if t_val <= hist_end_year:
                continue
            try:
                qty_val = float(qty)
            except Exception:
                continue
            if not np.isfinite(qty_val) or qty_val <= 0:
                continue
            nutrient_val = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
            if nutrient_val <= 0:
                continue
            contrib = nutrient_val * qty_val
            totals[(r, t_val)] += contrib
            top_by_rt[(r, t_val)].append((contrib, j))

        rows: List[Dict[str, Any]] = []
        for (r, t), rhs in nutrition_rhs.items():
            try:
                t_val = int(t)
            except Exception:
                continue
            if t_val <= hist_end_year:
                continue
            try:
                rhs_val = float(rhs)
            except Exception:
                continue
            if not np.isfinite(rhs_val):
                continue
            total_kcal = totals.get((r, t_val), 0.0)
            top_items = top_by_rt.get((r, t_val), [])
            top_items.sort(reverse=True, key=lambda x: x[0])
            top_str = ";".join(f"{j}:{val:.3e}" for val, j in top_items[:8])
            gap = rhs_val - total_kcal
            gap_ratio = gap / rhs_val if rhs_val != 0 else np.nan
            rows.append({
                'region': r,
                'year': t_val,
                'rhs_kcal': rhs_val,
                'max_kcal': total_kcal,
                'min_kcal': total_kcal,
                'gap_to_max': gap,
                'gap_ratio': gap_ratio,
                'coeff_pos_sum': 0.0,
                'coeff_neg_sum': 0.0,
                'top_items': top_str,
            })

        if not rows:
            return None
        out_path = out_dir / "nutrition_feasibility_diagnosis.csv"
        try:
            pd.DataFrame(rows).sort_values(['gap_to_max'], ascending=False).to_csv(
                out_path, index=False, encoding="utf-8-sig"
            )
        except Exception:
            return None
        return out_path

    rows: List[Dict[str, Any]] = []
    for (r, t), rhs in nutrition_rhs.items():
        try:
            t_val = int(t)
        except Exception:
            continue
        if t_val <= hist_end_year:
            continue
        try:
            rhs_val = float(rhs)
        except Exception:
            continue
        if not np.isfinite(rhs_val):
            continue

        coeff_pc: Dict[str, float] = defaultdict(float)
        total_const = 0.0
        demand_terms: Dict[str, Tuple[float, float, Dict[str, float], float]] = {}
        n_terms = 0

        for j in commodities:
            key = (r, j, t_val)
            if key not in idx:
                continue
            nutrient_val = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
            if nutrient_val <= 0:
                continue

            data = idx[key]
            base_key = (r, j, hist_end_year)
            if base_key in idx:
                D0_base_actual = max(1e-6, float(idx[base_key].get('D0', 1e-6) or 1e-6))
                P0_base = max(1e-6, float(idx[base_key].get('P0', 1.0) or 1.0))
            else:
                D0_base_actual = max(1e-6, float(data.get('D0', 1e-6) or 1e-6))
                P0_base = max(1e-6, float(data.get('P0', 1.0) or 1.0))
            D0 = D0_base_actual / max(1e-12, qty_scale)
            P0 = P0_base

            eps_d = float(data.get('eps_demand', 0.0) or 0.0)
            eps_pop = float(data.get('eps_pop_demand', 0.0) or 0.0)
            eps_inc = float(data.get('eps_income_demand', 0.0) or 0.0)

            pop_base = float(pop_base_cache.get(key, 1.0) or 1.0)
            inc_base = float(inc_base_cache.get(key, 1.0) or 1.0)
            pop_t = float(data.get('pop_t', pop_base) or pop_base)
            inc_t = float(data.get('inc_t', inc_base) or inc_base)

            pop_ratio = pop_t / max(1e-9, pop_base)
            inc_ratio = inc_t / max(1e-9, inc_base)
            pop_effect = pop_ratio ** eps_pop if eps_pop != 0 else 1.0
            inc_effect = inc_ratio ** eps_inc if eps_inc != 0 else 1.0
            D0_adjusted = D0 * pop_effect * inc_effect

            if feed_reduction_by:
                rate = float(feed_reduction_by.get(key, 0.0) or 0.0)
                rate = max(-1.0, min(1.0, rate))
                D0_adjusted *= (1.0 + rate)

            epsD_row_raw = {k: v for k, v in (data.get('epsD_row', {}) or {}).items() if k in commodities}
            epsD_row_scaled = _scale_cross_terms(epsD_row_raw, cross_terms_scale)
            epsD_row_limited = _limit_cross_terms(epsD_row_scaled, cross_terms_top_n)
            epsD_row, sum_cross_eps, _ = _normalize_cross_eps(epsD_row_limited, eps_d)

            a_d = D0_adjusted * (1.0 - eps_d)
            b_d_abs = D0_adjusted * eps_d / max(1e-6, P0)
            b_d = b_d_abs
            if use_relative_price:
                price_ref_self = float(price_ref_by_comm.get(j, 1.0) or 1.0)
                if not np.isfinite(price_ref_self) or price_ref_self <= 0:
                    price_ref_self = 1.0
                b_d = b_d_abs * price_ref_self

            cross_terms: Dict[str, float] = {}
            cross_const_offset_d = 0.0
            for other_comm, cross_eps in epsD_row.items():
                if other_comm == j:
                    continue
                other_base_key = (r, other_comm, hist_end_year)
                P0_other = idx.get(other_base_key, {}).get('P0')
                if P0_other is None:
                    other_key = (r, other_comm, t_val)
                    P0_other = idx.get(other_key, {}).get('P0', P0) or P0
                P0_other = max(1e-6, float(P0_other))
                b_cross_abs = D0_adjusted * float(cross_eps) / P0_other
                b_cross = b_cross_abs
                pc_other_base_for_offset = P0_other
                if use_relative_price:
                    price_ref_other = float(price_ref_by_comm.get(other_comm, 1.0) or 1.0)
                    if not np.isfinite(price_ref_other) or price_ref_other <= 0:
                        price_ref_other = 1.0
                    b_cross = b_cross_abs * price_ref_other
                    pc_other_base_for_offset = P0_other / price_ref_other
                b_cross = _sanitize_cross_coef(
                    b_cross,
                    base_qty=D0_adjusted,
                    cross_eps=float(cross_eps),
                    use_relative_price=use_relative_price,
                    ratio_tol=CROSS_COEF_RATIO_TOL,
                )
                if abs(b_cross) < 1e-12:
                    continue
                cross_terms[other_comm] = b_cross
                cross_const_offset_d += b_cross * pc_other_base_for_offset

            a_d = a_d - cross_const_offset_d
            a_d_eff = float(alpha_d_cache.get(key, a_d) or a_d)
            if key not in alpha_d_cache:
                wedge_self = _wedge(r, j, t_val)
                if wedge_self:
                    a_d_eff += b_d * wedge_self
                for other_comm, b_cross in cross_terms.items():
                    wedge_other = _wedge(r, other_comm, t_val)
                    if wedge_other:
                        a_d_eff += b_cross * wedge_other

            mult = nutrient_val * qty_scale
            total_const += mult * a_d_eff
            coeff_pc[j] += mult * b_d
            for other_comm, b_cross in cross_terms.items():
                coeff_pc[other_comm] += mult * b_cross

            demand_terms[j] = (a_d_eff, b_d, cross_terms, mult)
            n_terms += 1

        if n_terms <= 0:
            continue

        pc_choice_max: Dict[str, float] = {}
        pc_choice_min: Dict[str, float] = {}
        pos_sum = 0.0
        neg_sum = 0.0
        for j in commodities:
            coeff = float(coeff_pc.get(j, 0.0) or 0.0)
            pmin_j, pmax_j = _p_bounds(j)
            if coeff >= 0:
                pc_choice_max[j] = pmax_j
                pc_choice_min[j] = pmin_j
                pos_sum += coeff
            else:
                pc_choice_max[j] = pmin_j
                pc_choice_min[j] = pmax_j
                neg_sum += coeff

        max_kcal = total_const
        min_kcal = total_const
        for j in commodities:
            coeff = float(coeff_pc.get(j, 0.0) or 0.0)
            max_kcal += coeff * pc_choice_max[j]
            min_kcal += coeff * pc_choice_min[j]

        top_items: List[Tuple[float, str]] = []
        for j, (a_d_eff, b_d, cross_terms, mult) in demand_terms.items():
            qd = a_d_eff + b_d * pc_choice_max.get(j, _p_bounds(j)[0])
            for other_comm, b_cross in cross_terms.items():
                qd += b_cross * pc_choice_max.get(other_comm, _p_bounds(other_comm)[0])
            contrib = mult * qd
            top_items.append((contrib, j))
        top_items.sort(reverse=True, key=lambda x: x[0])
        top_str = ";".join(f"{j}:{val:.3e}" for val, j in top_items[:8])

        gap = rhs_val - max_kcal
        gap_ratio = gap / rhs_val if rhs_val != 0 else np.nan
        rows.append({
            'region': r,
            'year': t_val,
            'rhs_kcal': rhs_val,
            'max_kcal': max_kcal,
            'min_kcal': min_kcal,
            'gap_to_max': gap,
            'gap_ratio': gap_ratio,
            'coeff_pos_sum': pos_sum,
            'coeff_neg_sum': neg_sum,
            'top_items': top_str,
        })

    if not rows:
        return None
    out_path = out_dir / "nutrition_feasibility_diagnosis.csv"
    try:
        pd.DataFrame(rows).sort_values(['gap_to_max'], ascending=False).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
    except Exception:
        return None
    return out_path


def _write_nutrition_gap_by_model_coverage(
    cache: Dict[str, Any],
    output_dir: Optional[str],
    *,
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]],
    nutrient_per_unit_by_comm: Optional[Dict[str, float]],
    nutrition_demand_map: Optional[Dict[Tuple[str, str, int], float]] = None,
    hist_end_year: int,
    top_n: int = 8,
) -> Optional[Path]:
    if not output_dir or not nutrition_rhs or not nutrient_per_unit_by_comm:
        return None
    demand_map = nutrition_demand_map or cache.get('nutrition_demand_map', {}) or {}
    if not demand_map:
        return None
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    idx = cache.get('idx', {}) or {}
    nonfood = set(cache.get('nonfood_commodities', []) or [])

    model_comms_by_rt: Dict[Tuple[str, int], set] = defaultdict(set)
    for (r, j, t) in idx.keys():
        try:
            t_val = int(t)
        except Exception:
            continue
        if t_val <= hist_end_year:
            continue
        if nonfood and j in nonfood:
            continue
        model_comms_by_rt[(r, t_val)].add(j)

    demand_by_rt: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(dict)
    for (r, j, t), qty in demand_map.items():
        try:
            t_val = int(t)
        except Exception:
            continue
        if t_val <= hist_end_year:
            continue
        try:
            qty_val = float(qty)
        except Exception:
            continue
        if not np.isfinite(qty_val) or qty_val <= 0:
            continue
        demand_by_rt[(r, t_val)][j] = qty_val

    def _fmt_pairs(pairs: List[Tuple[float, str]]) -> str:
        return ";".join(f"{j}:{val:.3e}" for val, j in pairs[:top_n])

    rows: List[Dict[str, Any]] = []
    for (r, t), rhs in nutrition_rhs.items():
        try:
            t_val = int(t)
        except Exception:
            continue
        if t_val <= hist_end_year:
            continue
        try:
            rhs_val = float(rhs)
        except Exception:
            continue
        if not np.isfinite(rhs_val):
            continue

        comms_model = model_comms_by_rt.get((r, t_val), set())
        comms_demand = demand_by_rt.get((r, t_val), {})
        comms_demand_set = set(comms_demand.keys())

        total_model = 0.0
        total_profile = 0.0
        missing_in_model: List[Tuple[float, str]] = []
        top_model: List[Tuple[float, str]] = []
        missing_nutrient: List[str] = []

        for j, qty_val in comms_demand.items():
            nutrient_val = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
            if nutrient_val <= 0:
                missing_nutrient.append(j)
                continue
            kcal = nutrient_val * qty_val
            total_profile += kcal
            if j in comms_model:
                total_model += kcal
                top_model.append((kcal, j))
            else:
                missing_in_model.append((kcal, j))

        missing_in_model.sort(reverse=True, key=lambda x: x[0])
        top_model.sort(reverse=True, key=lambda x: x[0])
        missing_in_map = sorted(comms_model - comms_demand_set)
        missing_nutrient = sorted(set(missing_nutrient))

        gap_to_rhs = rhs_val - total_model
        rows.append({
            'region': r,
            'year': t_val,
            'rhs_kcal': rhs_val,
            'kcal_model_total': total_model,
            'kcal_profile_total': total_profile,
            'gap_to_rhs': gap_to_rhs,
            'gap_ratio': gap_to_rhs / rhs_val if rhs_val != 0 else np.nan,
            'profile_gap_to_rhs': rhs_val - total_profile,
            'model_coverage_ratio': total_model / total_profile if total_profile > 0 else np.nan,
            'n_model_comms': len(comms_model),
            'n_profile_comms': len(comms_demand_set),
            'n_overlap_comms': len(comms_model & comms_demand_set),
            'n_missing_in_model': len(missing_in_model),
            'missing_in_model_kcal': sum(val for val, _ in missing_in_model),
            'missing_in_model_items': _fmt_pairs(missing_in_model),
            'n_missing_in_map': len(missing_in_map),
            'missing_in_map_items': ";".join(missing_in_map[:top_n]),
            'n_missing_nutrient': len(missing_nutrient),
            'missing_nutrient_items': ";".join(missing_nutrient[:top_n]),
            'top_model_items': _fmt_pairs(top_model),
        })

    if not rows:
        return None
    out_path = out_dir / "nutrition_gap_by_model.csv"
    try:
        pd.DataFrame(rows).sort_values(['gap_to_rhs'], ascending=False).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
    except Exception:
        return None
    return out_path


def _write_forest_nonneg_feasibility_diagnosis(
    cache: Dict[str, Any],
    output_dir: Optional[str],
    *,
    hist_end_year: int,
) -> Optional[Path]:
    if not output_dir:
        return None
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    idx = cache.get('idx', {}) or {}
    commodities = cache.get('commodities', []) or []
    regions = cache.get('regions', []) or []
    years = cache.get('years', []) or []
    price_bounds = cache.get('price_bounds', (1e-6, 1e6))
    Pmin, Pmax = float(price_bounds[0]), float(price_bounds[1])
    price_bounds_by_comm = cache.get('price_bounds_by_comm', {}) or {}
    def _p_bounds(comm: str) -> Tuple[float, float]:
        return price_bounds_by_comm.get(comm, (Pmin, Pmax))
    use_relative_price = bool(cache.get('use_relative_price'))
    price_ref_by_comm = cache.get('price_ref_by_comm', {}) or {}
    cross_terms_top_n = cache.get('cross_terms_top_n')
    cross_terms_scale = cache.get('cross_terms_scale')
    alpha_s_cache = cache.get('alpha_s', {}) or {}
    qty_scale = float(cache.get('qty_scale', 1.0) or 1.0)
    grassland_method = str(cache.get('grassland_method', 'dynamic') or 'dynamic')
    base_cropland_area = cache.get('base_cropland_area', {}) or {}
    base_grassland_area = cache.get('base_grassland_area', {}) or {}
    base_forest_area = cache.get('base_forest_area', {}) or {}
    base_forest_area_scaled = cache.get('base_forest_area_scaled', {}) or {}
    grass_area_by_region_year = cache.get('grass_area_by_region_year', {}) or {}
    forest_area_by_region_year = cache.get('forest_area_by_region_year', {}) or {}
    price_wedge_by_region_comm_year = cache.get('price_wedge_by_region_comm_year', {}) or {}
    price_wedge_by_region_comm = cache.get('price_wedge_by_region_comm', {}) or {}
    price_wedge_by_region = cache.get('price_wedge_by_region', {}) or {}

    def _wedge(region: Any, comm: Any, year: Any) -> float:
        return _get_price_wedge(
            region,
            comm,
            year,
            by_rcy=price_wedge_by_region_comm_year,
            by_rc=price_wedge_by_region_comm,
            by_r=price_wedge_by_region,
        )

    rows: List[Dict[str, Any]] = []
    for r in regions:
        base_crop = float(base_cropland_area.get(r, 0.0) or 0.0)
        base_grass = float(base_grassland_area.get(r, 0.0) or 0.0)
        base_forest = float(base_forest_area_scaled.get(r, base_forest_area.get(r, 0.0)) or 0.0)
        if base_forest <= 0 and base_crop <= 0 and base_grass <= 0:
            continue
        base_total = base_crop + base_grass + base_forest

        for t in years:
            if int(t) <= hist_end_year:
                continue
            grass_fixed = 0.0
            if grassland_method != 'dynamic':
                grass_fixed = float(grass_area_by_region_year.get((r, t), 0.0) or 0.0)
            forest_fixed = 0.0
            if forest_area_by_region_year:
                forest_fixed = float(forest_area_by_region_year.get((r, t), 0.0) or 0.0)
            capacity = base_total - grass_fixed if grassland_method != 'dynamic' else base_total

            coeff_pc: Dict[str, float] = defaultdict(float)
            total_const = 0.0
            supply_terms: Dict[str, Tuple[float, float, Dict[str, float], float]] = {}
            n_terms = 0

            for j in commodities:
                if j == "Fish, Seafood":
                    continue
                key = (r, j, int(t))
                if key not in idx:
                    continue
                data = idx[key]
                base_key = (r, j, hist_end_year)
                if base_key in idx:
                    base_node = idx[base_key]
                    try:
                        Q0_base_raw = float(base_node.get('Q0', 0.0) or 0.0)
                    except Exception:
                        Q0_base_raw = 0.0
                    Q0_base_actual = max(1e-6, Q0_base_raw)
                    P0_base = max(1e-6, float(idx[base_key].get('P0', 1.0) or 1.0))
                else:
                    base_node = data
                    try:
                        Q0_base_raw = float(data.get('Q0', 0.0) or 0.0)
                    except Exception:
                        Q0_base_raw = 0.0
                    Q0_base_actual = max(1e-6, Q0_base_raw)
                    P0_base = max(1e-6, float(data.get('P0', 1.0) or 1.0))
                Q0 = Q0_base_actual / max(1e-12, qty_scale)
                P0 = P0_base

                eps_s = float(data.get('eps_supply', 0.0) or 0.0)
                eta_y = float(data.get('eps_supply_yield', 0.0) or 0.0)
                eta_temp = float(data.get('eps_supply_temp', 0.0) or 0.0)
                Ymult = float(data.get('Ymult', 1.0) or 1.0)
                Tmult = float(data.get('Tmult', 1.0) or 1.0)

                try:
                    yield_check = float(data.get('yield0', 0.0) or 0.0)
                except Exception:
                    yield_check = 0.0
                if (not np.isfinite(yield_check) or yield_check <= 0.0) and Q0_base_raw <= 1e-3:
                    continue
                yield_used = _cropland_yield0(
                    data,
                    region=r,
                    commodity=j,
                    year=int(t),
                    context="forest_nonneg_diag",
                )
                coef_crop = qty_scale / yield_used
                coef_grass = 0.0
                if grassland_method == 'dynamic':
                    coef_grass = float(data.get('grassland_coef', 0.0) or 0.0)
                land_coef = coef_crop + coef_grass

                epsS_row_raw = {k: v for k, v in (data.get('epsS_row', {}) or {}).items() if k in commodities}
                epsS_row_scaled = _scale_cross_terms(epsS_row_raw, cross_terms_scale)
                epsS_row = _limit_cross_terms(epsS_row_scaled, cross_terms_top_n)
                sum_cross_eps = sum(float(v) for v in epsS_row.values())

                yield_adj = eta_y * (Ymult - 1.0)
                temp_adj = eta_temp * (Tmult - 1.0)
                a_s = Q0 * (1.0 + yield_adj + temp_adj - eps_s - sum_cross_eps)

                b_s_abs = Q0 * eps_s / max(1e-6, P0)
                b_s = b_s_abs
                if use_relative_price:
                    price_ref_self = float(price_ref_by_comm.get(j, 1.0) or 1.0)
                    if not np.isfinite(price_ref_self) or price_ref_self <= 0:
                        price_ref_self = 1.0
                    b_s = b_s_abs * price_ref_self

                cross_terms: Dict[str, float] = {}
                for other_comm, cross_eps in epsS_row.items():
                    if other_comm == j:
                        continue
                    other_base_key = (r, other_comm, hist_end_year)
                    P0_other = idx.get(other_base_key, {}).get('P0')
                    if P0_other is None:
                        other_key = (r, other_comm, int(t))
                        P0_other = idx.get(other_key, {}).get('P0', P0) or P0
                    P0_other = max(1e-6, float(P0_other))
                    b_cross_abs = Q0 * float(cross_eps) / P0_other
                    b_cross = b_cross_abs
                    if use_relative_price:
                        price_ref_other = float(price_ref_by_comm.get(other_comm, 1.0) or 1.0)
                        if not np.isfinite(price_ref_other) or price_ref_other <= 0:
                            price_ref_other = 1.0
                        b_cross = b_cross_abs * price_ref_other
                    b_cross = _sanitize_cross_coef(
                        b_cross,
                        base_qty=Q0,
                        cross_eps=float(cross_eps),
                        use_relative_price=use_relative_price,
                        ratio_tol=CROSS_COEF_RATIO_TOL,
                    )
                    if abs(b_cross) < 1e-12:
                        continue
                    cross_terms[other_comm] = b_cross

                a_s_eff = float(alpha_s_cache.get(key, a_s) or a_s)
                if key not in alpha_s_cache:
                    wedge_self = _wedge(r, j, int(t))
                    if wedge_self:
                        a_s_eff += b_s * wedge_self
                    for other_comm, b_cross in cross_terms.items():
                        wedge_other = _wedge(r, other_comm, int(t))
                        if wedge_other:
                            a_s_eff += b_cross * wedge_other

                total_const += land_coef * a_s_eff
                coeff_pc[j] += land_coef * b_s
                for other_comm, b_cross in cross_terms.items():
                    coeff_pc[other_comm] += land_coef * b_cross

                supply_terms[j] = (a_s_eff, b_s, cross_terms, land_coef)
                n_terms += 1

            if n_terms <= 0:
                continue

            pc_choice_min: Dict[str, float] = {}
            pc_choice_max: Dict[str, float] = {}
            pos_sum = 0.0
            neg_sum = 0.0
            for j in commodities:
                coeff = float(coeff_pc.get(j, 0.0) or 0.0)
                pmin_j, pmax_j = _p_bounds(j)
                if coeff >= 0:
                    pc_choice_min[j] = pmin_j
                    pc_choice_max[j] = pmax_j
                    pos_sum += coeff
                else:
                    pc_choice_min[j] = pmax_j
                    pc_choice_max[j] = pmin_j
                    neg_sum += coeff

            min_land = total_const
            max_land = total_const
            for j in commodities:
                coeff = float(coeff_pc.get(j, 0.0) or 0.0)
                min_land += coeff * pc_choice_min[j]
                max_land += coeff * pc_choice_max[j]

            top_items: List[Tuple[float, str]] = []
            for j, (a_s_eff, b_s, cross_terms, land_coef) in supply_terms.items():
                qs = a_s_eff + b_s * pc_choice_min.get(j, _p_bounds(j)[0])
                for other_comm, b_cross in cross_terms.items():
                    qs += b_cross * pc_choice_min.get(other_comm, _p_bounds(other_comm)[0])
                land_use = land_coef * qs
                top_items.append((land_use, j))
            top_items.sort(reverse=True, key=lambda x: x[0])
            top_str = ";".join(f"{j}:{val:.3e}" for val, j in top_items[:8])

            min_gap = min_land - capacity
            rows.append({
                'region': r,
                'year': int(t),
                'base_total_ha': base_total,
                'base_forest_ha': base_forest,
                'base_cropland_ha': base_crop,
                'base_grassland_ha': base_grass,
                'grassland_fixed_ha': grass_fixed,
                'forest_fixed_ha': forest_fixed,
                'capacity_cropland_ha': capacity,
                'min_land_demand_ha': min_land,
                'max_land_demand_ha': max_land,
                'min_gap_ha': min_gap,
                'coeff_pos_sum': pos_sum,
                'coeff_neg_sum': neg_sum,
                'top_items': top_str,
            })

    if not rows:
        return None
    out_path = out_dir / "forest_nonneg_feasibility_diagnosis.csv"
    try:
        pd.DataFrame(rows).sort_values(['min_gap_ha'], ascending=False).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
    except Exception:
        return None
    return out_path


def _parse_bracket_args(raw: str,
                        *,
                        expect_region: Optional[bool] = None) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    s = raw.strip()
    if not s.startswith('[') or not s.endswith(']'):
        return None, None, None
    inner = s[1:-1].strip()
    if not inner:
        return None, None, None
    parts = inner.rsplit(',', 1)
    if len(parts) != 2:
        return None, None, None
    left, year_str = parts[0].strip(), parts[1].strip()
    try:
        year = int(year_str)
    except Exception:
        year = None
    if expect_region is True:
        if ',' in left:
            region, commodity = left.split(',', 1)
            return region.strip(), commodity.strip(), year
        return left.strip(), None, year
    if expect_region is False:
        return None, left.strip(), year
    if left.startswith("'") and len(left) >= 4 and left[1:4].isdigit():
        if ',' in left:
            region, commodity = left.split(',', 1)
            return region.strip(), commodity.strip(), year
        return left.strip(), None, year
    return None, left.strip(), year


def _write_constraint_residuals(model: gp.Model,
                                output_dir: Optional[str]) -> Optional[Path]:
    if not output_dir or model is None:
        return None
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    rows: List[Dict[str, Any]] = []
    for c in model.getConstrs():
        name = c.ConstrName
        ctype = name.split('[', 1)[0] if '[' in name else name
        region = None
        commodity = None
        year = None
        if '[' in name and name.endswith(']'):
            inside = name[name.find('['):]
            expect_region = True
            if ctype == 'clear':
                expect_region = False
            region, commodity, year = _parse_bracket_args(inside, expect_region=expect_region)
            if ctype in {'nutri', 'forest_nonneg', 'land', 'land_limit'}:
                if commodity and region is None:
                    region = commodity
                    commodity = None

        row = model.getRow(c)
        lhs = 0.0
        for k in range(row.size()):
            v = row.getVar(k)
            coef = row.getCoeff(k)
            try:
                val = float(v.X)
            except Exception:
                val = 0.0
            lhs += coef * val
        rhs = float(c.RHS)
        sense = c.Sense
        if sense == '=':
            violation = abs(lhs - rhs)
        elif sense == '<':
            violation = max(lhs - rhs, 0.0)
        elif sense == '>':
            violation = max(rhs - lhs, 0.0)
        else:
            violation = 0.0

        rows.append({
            'name': name,
            'type': ctype,
            'region': region,
            'commodity': commodity,
            'year': year,
            'sense': sense,
            'lhs': lhs,
            'rhs': rhs,
            'slack': float(c.Slack),
            'violation': violation,
        })

    if not rows:
        return None
    out_path = out_dir / "constraint_residuals.csv"
    try:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        return None
    return out_path


def _write_demand_equation_check(model: gp.Model,
                                 output_dir: Optional[str],
                                 *,
                                 years_filter: Optional[List[int]] = None) -> Optional[Path]:
    if not output_dir or model is None:
        return None
    cache = getattr(model, '_nzf_cache', {}) or {}
    pc_by_region = bool(cache.get('pc_by_region', False))
    constr_demand = cache.get('constr_demand', {}) or {}
    Pc = cache.get('Pc', {}) or {}
    Qd = cache.get('Qd', {}) or {}
    price_ref_by_comm = cache.get('price_ref_by_comm', {}) or {}
    eps_d_map = cache.get('eps_d', {}) or {}
    D0_map = cache.get('D0', {}) or {}
    P0_map = cache.get('P0', {}) or {}
    qty_scale = float(cache.get('qty_scale', 1.0) or 1.0)
    use_relative_price = bool(cache.get('use_relative_price'))

    rows: List[Dict[str, Any]] = []
    for (r, j, t), c in constr_demand.items():
        if years_filter and int(t) not in years_filter:
            continue
        qd_var = Qd.get((r, j, t))
        if qd_var is None:
            continue
        try:
            qd_val = float(qd_var.X)
        except Exception:
            continue

        row = model.getRow(c)
        a_d_eff = float(c.RHS)
        qd_implied = a_d_eff
        b_self = 0.0
        cross_n = 0
        cross_sum = 0.0
        qs_terms_n = 0
        qs_terms_sum = 0.0
        for k in range(row.size()):
            v = row.getVar(k)
            coef = row.getCoeff(k)
            vname = v.VarName
            if vname.startswith("Pc[") and vname.endswith("]"):
                inside = vname[vname.find('['):]
                region, comm, yr = _parse_bracket_args(inside, expect_region=pc_by_region)
                if comm is None:
                    continue
                if pc_by_region and region is not None and str(region) != str(r):
                    continue
                try:
                    pc_val = float(v.X)
                except Exception:
                    pc_val = 0.0
                term = -coef * pc_val
                qd_implied += term
                if comm == j and int(yr or -1) == int(t):
                    b_self = -coef
                else:
                    cross_n += 1
                    cross_sum += -coef
            elif vname.startswith("Qs[") and vname.endswith("]"):
                inside = vname[vname.find('['):]
                _, comm, yr = _parse_bracket_args(inside, expect_region=True)
                if comm is None:
                    continue
                try:
                    qs_val = float(v.X)
                except Exception:
                    qs_val = 0.0
                term = -coef * qs_val
                qd_implied += term
                qs_terms_n += 1
                qs_terms_sum += -coef

        pc_self_var = Pc.get((r, j, t)) if pc_by_region else Pc.get((j, t))
        pc_val = float(pc_self_var.X) if pc_self_var is not None else float('nan')
        pc_abs = pc_val
        if use_relative_price:
            ref = float(price_ref_by_comm.get(j, 1.0) or 1.0)
            if not np.isfinite(ref) or ref <= 0:
                ref = 1.0
            pc_abs = pc_val * ref

        resid = qd_val - qd_implied
        rows.append({
            'region': r,
            'commodity': j,
            'year': int(t),
            'Pc_model': pc_val,
            'Pc_abs': pc_abs,
            'Qd_model_scaled': qd_val,
            'Qd_model': qd_val * qty_scale,
            'Qd_implied_scaled': qd_implied,
            'Qd_implied': qd_implied * qty_scale,
            'residual_scaled': resid,
            'residual': resid * qty_scale,
            'a_d_eff': a_d_eff,
            'b_d': b_self,
            'cross_terms_n': cross_n,
            'cross_coef_sum': cross_sum,
            'qs_terms_n': qs_terms_n,
            'qs_coef_sum': qs_terms_sum,
            'eps_d': float(eps_d_map.get((r, j, t), 0.0) or 0.0),
            'D0_scaled': float(D0_map.get((r, j, t), 0.0) or 0.0),
            'P0': float(P0_map.get((r, j, t), 0.0) or 0.0),
        })

    if not rows:
        return None
    out_path = Path(output_dir) / "demand_equation_check.csv"
    try:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        return None
    return out_path


def _write_supply_equation_check(model: gp.Model,
                                 output_dir: Optional[str],
                                 *,
                                 years_filter: Optional[List[int]] = None) -> Optional[Path]:
    if not output_dir or model is None:
        return None
    cache = getattr(model, '_nzf_cache', {}) or {}
    constr_supply = cache.get('constr_supply', {}) or {}
    Pc = cache.get('Pc', {}) or {}
    Qs = cache.get('Qs', {}) or {}
    price_ref_by_comm = cache.get('price_ref_by_comm', {}) or {}
    eps_s_map = cache.get('eps_s', {}) or {}
    Q0_map = cache.get('Q0', {}) or {}
    P0_map = cache.get('P0', {}) or {}
    qty_scale = float(cache.get('qty_scale', 1.0) or 1.0)
    use_relative_price = bool(cache.get('use_relative_price'))
    pc_by_region = bool(cache.get('pc_by_region', False))
    pc_by_region = bool(cache.get('pc_by_region', False))

    rows: List[Dict[str, Any]] = []
    for (r, j, t), c in constr_supply.items():
        if years_filter and int(t) not in years_filter:
            continue
        qs_var = Qs.get((r, j, t))
        if qs_var is None:
            continue
        try:
            qs_val = float(qs_var.X)
        except Exception:
            continue

        row = model.getRow(c)
        a_s_eff = float(c.RHS)
        qs_implied = a_s_eff
        b_self = 0.0
        cross_n = 0
        cross_sum = 0.0
        for k in range(row.size()):
            v = row.getVar(k)
            coef = row.getCoeff(k)
            vname = v.VarName
            if vname.startswith("Pc[") and vname.endswith("]"):
                inside = vname[vname.find('['):]
                region, comm, yr = _parse_bracket_args(inside, expect_region=pc_by_region)
                if comm is None:
                    continue
                if pc_by_region and region is not None and str(region) != str(r):
                    continue
                try:
                    pc_val = float(v.X)
                except Exception:
                    pc_val = 0.0
                term = -coef * pc_val
                qs_implied += term
                if comm == j and int(yr or -1) == int(t):
                    b_self = -coef
                else:
                    cross_n += 1
                    cross_sum += -coef

        pc_self_var = Pc.get((r, j, t)) if pc_by_region else Pc.get((j, t))
        pc_val = float(pc_self_var.X) if pc_self_var is not None else float('nan')
        pc_abs = pc_val
        if use_relative_price:
            ref = float(price_ref_by_comm.get(j, 1.0) or 1.0)
            if not np.isfinite(ref) or ref <= 0:
                ref = 1.0
            pc_abs = pc_val * ref

        resid = qs_val - qs_implied
        cross_contrib = qs_implied - a_s_eff - b_self * pc_val
        rows.append({
            'region': r,
            'commodity': j,
            'year': int(t),
            'Pc_model': pc_val,
            'Pc_abs': pc_abs,
            'Qs_model_scaled': qs_val,
            'Qs_model': qs_val * qty_scale,
            'Qs_implied_scaled': qs_implied,
            'Qs_implied': qs_implied * qty_scale,
            'residual_scaled': resid,
            'residual': resid * qty_scale,
            'a_s_eff': a_s_eff,
            'b_s': b_self,
            'cross_terms_n': cross_n,
            'cross_coef_sum': cross_sum,
            'cross_contrib': cross_contrib,
            'eps_s': float(eps_s_map.get((r, j, t), 0.0) or 0.0),
            'Q0_scaled': float(Q0_map.get((r, j, t), 0.0) or 0.0),
            'P0': float(P0_map.get((r, j, t), 0.0) or 0.0),
        })

    if not rows:
        return None
    out_path = Path(output_dir) / "supply_equation_check.csv"
    try:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        return None
    return out_path


def _write_balance_clear_diagnosis(model: gp.Model,
                                   output_dir: Optional[str],
                                   *,
                                   top_n: int = 50,
                                   years_filter: Optional[List[int]] = None) -> Optional[Path]:
    if not output_dir or model is None:
        return None
    cache = getattr(model, '_nzf_cache', {}) or {}
    Pc = cache.get('Pc', {}) or {}
    Qs = cache.get('Qs', {}) or {}
    Qd = cache.get('Qd', {}) or {}
    net_import = cache.get('net_import', {}) or {}
    excess = cache.get('excess', {}) or {}
    shortage = cache.get('shortage', {}) or {}
    constr_supply = cache.get('constr_supply', {}) or {}
    constr_demand = cache.get('constr_demand', {}) or {}
    eps_s_map = cache.get('eps_s', {}) or {}
    eps_d_map = cache.get('eps_d', {}) or {}
    Q0_map = cache.get('Q0', {}) or {}
    D0_map = cache.get('D0', {}) or {}
    P0_map = cache.get('P0', {}) or {}
    price_ref_by_comm = cache.get('price_ref_by_comm', {}) or {}
    qty_scale = float(cache.get('qty_scale', 1.0) or 1.0)
    use_relative_price = bool(cache.get('use_relative_price'))
    pc_by_region = bool(cache.get('pc_by_region', False))

    def _eq_terms(constr: gp.Constr, target_region: str, target_comm: str, target_year: int) -> Tuple[float, float, float, float, int]:
        row = model.getRow(constr)
        a_eff = float(constr.RHS)
        implied = a_eff
        b_self = 0.0
        cross_sum = 0.0
        cross_n = 0
        for k in range(row.size()):
            v = row.getVar(k)
            coef = row.getCoeff(k)
            vname = v.VarName
            if not (vname.startswith("Pc[") and vname.endswith("]")):
                continue
            inside = vname[vname.find('['):]
            region, comm, yr = _parse_bracket_args(inside, expect_region=pc_by_region)
            if comm is None or yr is None or int(yr) != int(target_year):
                continue
            if pc_by_region and region is not None and str(region) != str(target_region):
                continue
            try:
                pc_val = float(v.X)
            except Exception:
                pc_val = 0.0
            term = -coef * pc_val
            implied += term
            if comm == target_comm:
                b_self = -coef
            else:
                cross_sum += -coef
                cross_n += 1
        return implied, a_eff, b_self, cross_sum, cross_n

    rows: List[Dict[str, Any]] = []

    # Collect balance & clear constraints
    balance_records: List[Tuple[float, str, str, str, int, float]] = []
    clear_records: List[Tuple[float, str, str, int, float]] = []
    for c in model.getConstrs():
        name = c.ConstrName
        if name.startswith("balance["):
            region, comm, year = _parse_bracket_args(name[name.find('['):], expect_region=True)
            if region is None or comm is None or year is None:
                continue
            if years_filter and int(year) not in years_filter:
                continue
            violation = max(abs(float(c.Slack)), 0.0)
            balance_records.append((violation, name, region, comm, int(year), float(c.Slack)))
        elif name.startswith("clear["):
            region, comm, year = _parse_bracket_args(name[name.find('['):], expect_region=False)
            if comm is None or year is None:
                continue
            if years_filter and int(year) not in years_filter:
                continue
            violation = max(abs(float(c.Slack)), 0.0)
            clear_records.append((violation, name, comm, int(year), float(c.Slack)))

    balance_records.sort(key=lambda x: x[0], reverse=True)
    clear_records.sort(key=lambda x: x[0], reverse=True)

    for violation, name, r, j, t, slack in balance_records[:top_n]:
        key = (r, j, t)
        qs = float(Qs.get(key).X) if key in Qs else 0.0
        qd = float(Qd.get(key).X) if key in Qd else 0.0
        mi = float(net_import.get(key).X) if key in net_import else 0.0
        pc_var = Pc.get((r, j, t)) if pc_by_region else Pc.get((j, t))
        pc_val = float(pc_var.X) if pc_var is not None else float('nan')
        pc_abs = pc_val
        if use_relative_price:
            ref = float(price_ref_by_comm.get(j, 1.0) or 1.0)
            if not np.isfinite(ref) or ref <= 0:
                ref = 1.0
            pc_abs = pc_val * ref

        qs_imp = qs
        qd_imp = qd
        a_s_eff = b_s = a_d_eff = b_d = cross_s = cross_d = 0.0
        cross_s_n = cross_d_n = 0
        if key in constr_supply:
            qs_imp, a_s_eff, b_s, cross_s, cross_s_n = _eq_terms(constr_supply[key], r, j, t)
        if key in constr_demand:
            qd_imp, a_d_eff, b_d, cross_d, cross_d_n = _eq_terms(constr_demand[key], r, j, t)

        rows.append({
            'type': 'balance',
            'name': name,
            'region': r,
            'commodity': j,
            'year': t,
            'violation': violation,
            'slack': slack,
            'Qs_model': qs * qty_scale,
            'Qd_model': qd * qty_scale,
            'net_import': mi * qty_scale,
            'balance_lhs': (qs + mi - qd) * qty_scale,
            'Pc_model': pc_val,
            'Pc_abs': pc_abs,
            'Qs_implied': qs_imp * qty_scale,
            'Qd_implied': qd_imp * qty_scale,
            'a_s_eff': a_s_eff,
            'b_s': b_s,
            'cross_s_sum': cross_s,
            'cross_s_n': cross_s_n,
            'a_d_eff': a_d_eff,
            'b_d': b_d,
            'cross_d_sum': cross_d,
            'cross_d_n': cross_d_n,
            'eps_s': float(eps_s_map.get(key, 0.0) or 0.0),
            'eps_d': float(eps_d_map.get(key, 0.0) or 0.0),
            'Q0': float(Q0_map.get(key, 0.0) or 0.0),
            'D0': float(D0_map.get(key, 0.0) or 0.0),
            'P0': float(P0_map.get(key, 0.0) or 0.0),
        })

    for violation, name, j, t, slack in clear_records[:top_n]:
        ex = float(excess.get((j, t)).X) if (j, t) in excess else 0.0
        sh = float(shortage.get((j, t)).X) if (j, t) in shortage else 0.0
        net_vals = []
        for (r, jj, tt), v in net_import.items():
            if jj == j and tt == t:
                try:
                    net_vals.append((r, float(v.X)))
                except Exception:
                    continue
        net_sum = sum(v for _, v in net_vals)
        net_vals.sort(key=lambda x: x[1], reverse=True)
        top_pos = ";".join(f"{r}:{v*qty_scale:.3e}" for r, v in net_vals[:5])
        top_neg = ";".join(f"{r}:{v*qty_scale:.3e}" for r, v in net_vals[-5:])

        rows.append({
            'type': 'clear',
            'name': name,
            'region': None,
            'commodity': j,
            'year': t,
            'violation': violation,
            'slack': slack,
            'net_import_sum': net_sum * qty_scale,
            'excess': ex * qty_scale,
            'shortage': sh * qty_scale,
            'clear_lhs': (net_sum + ex - sh) * qty_scale,
            'top_net_import_pos': top_pos,
            'top_net_import_neg': top_neg,
        })

    # Net import bound violations
    for (r, j, t), v in net_import.items():
        if years_filter and int(t) not in years_filter:
            continue
        try:
            x = float(v.X)
        except Exception:
            continue
        lb = float(v.LB)
        ub = float(v.UB)
        viol = 0.0
        sense = ''
        if x < lb:
            viol = lb - x
            sense = 'LB'
        elif x > ub:
            viol = x - ub
            sense = 'UB'
        if viol <= 0:
            continue
        key = (r, j, t)
        qs = float(Qs.get(key).X) if key in Qs else 0.0
        qd = float(Qd.get(key).X) if key in Qd else 0.0
        rows.append({
            'type': f'net_import_{sense}',
            'name': f"net_import[{r},{j},{t}]",
            'region': r,
            'commodity': j,
            'year': int(t),
            'violation': viol * qty_scale,
            'slack': None,
            'net_import': x * qty_scale,
            'net_import_lb': lb * qty_scale,
            'net_import_ub': ub * qty_scale,
            'Qs_model': qs * qty_scale,
            'Qd_model': qd * qty_scale,
        })

    if not rows:
        return None
    out_path = Path(output_dir) / "balance_clear_diagnosis.csv"
    try:
        pd.DataFrame(rows).sort_values(['type', 'violation'], ascending=[True, False]).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
    except Exception:
        return None
    return out_path


def _write_supply_cross_terms_diagnosis(model: gp.Model,
                                        output_dir: Optional[str],
                                        *,
                                        years_filter: Optional[List[int]] = None,
                                        top_n: int = 50,
                                        per_key_terms: int = 8) -> Optional[Path]:
    if not output_dir or model is None:
        return None
    cache = getattr(model, '_nzf_cache', {}) or {}
    constr_supply = cache.get('constr_supply', {}) or {}
    Pc = cache.get('Pc', {}) or {}
    Qs = cache.get('Qs', {}) or {}
    price_ref_by_comm = cache.get('price_ref_by_comm', {}) or {}
    qty_scale = float(cache.get('qty_scale', 1.0) or 1.0)
    use_relative_price = bool(cache.get('use_relative_price'))
    pc_by_region = bool(cache.get('pc_by_region', False))

    key_stats = []
    for (r, j, t), c in constr_supply.items():
        if years_filter and int(t) not in years_filter:
            continue
        qs_var = Qs.get((r, j, t))
        if qs_var is None:
            continue
        try:
            qs_val = float(qs_var.X)
        except Exception:
            continue
        row = model.getRow(c)
        a_s_eff = float(c.RHS)
        qs_implied = a_s_eff
        b_self = 0.0
        for k in range(row.size()):
            v = row.getVar(k)
            coef = row.getCoeff(k)
            vname = v.VarName
            if vname.startswith("Pc[") and vname.endswith("]"):
                inside = vname[vname.find('['):]
                region, comm, yr = _parse_bracket_args(inside, expect_region=pc_by_region)
                if comm is None or yr is None or int(yr) != int(t):
                    continue
                if pc_by_region and region is not None and str(region) != str(r):
                    continue
                try:
                    pc_val = float(v.X)
                except Exception:
                    pc_val = 0.0
                term = -coef * pc_val
                qs_implied += term
                if comm == j:
                    b_self = -coef
        resid = qs_val - qs_implied
        key_stats.append((qs_implied, abs(resid), r, j, t, qs_val, a_s_eff, b_self))

    key_stats.sort(key=lambda x: (x[0], -x[1]))
    top_keys = key_stats[:top_n]

    rows: List[Dict[str, Any]] = []
    for qs_implied, resid_abs, r, j, t, qs_val, a_s_eff, b_self in top_keys:
        c = constr_supply.get((r, j, t))
        if c is None:
            continue
        pc_var = Pc.get((r, j, t)) if pc_by_region else Pc.get((j, t))
        pc_val = float(pc_var.X) if pc_var is not None else float('nan')
        pc_abs = pc_val
        if use_relative_price:
            ref = float(price_ref_by_comm.get(j, 1.0) or 1.0)
            if not np.isfinite(ref) or ref <= 0:
                ref = 1.0
            pc_abs = pc_val * ref

        row = model.getRow(c)
        terms = []
        for k in range(row.size()):
            v = row.getVar(k)
            coef = row.getCoeff(k)
            vname = v.VarName
            if vname.startswith("Pc[") and vname.endswith("]"):
                inside = vname[vname.find('['):]
                region, comm, yr = _parse_bracket_args(inside, expect_region=pc_by_region)
                if comm is None or yr is None or int(yr) != int(t):
                    continue
                if pc_by_region and region is not None and str(region) != str(r):
                    continue
                try:
                    pc_val_term = float(v.X)
                except Exception:
                    pc_val_term = 0.0
                term_coef = -coef
                term_val = term_coef * pc_val_term
                terms.append((term_val, comm, term_coef, pc_val_term))

        terms.sort(key=lambda x: x[0])
        neg_terms = [x for x in terms if x[0] < 0 and x[1] != j][:per_key_terms]
        for term_val, comm, term_coef, pc_val_term in neg_terms:
            rows.append({
                'region': r,
                'commodity': j,
                'year': int(t),
                'term_comm': comm,
                'term_coef': term_coef,
                'Pc_term': pc_val_term,
                'term_contrib': term_val * qty_scale,
                'Qs_model': qs_val * qty_scale,
                'Qs_implied': qs_implied * qty_scale,
                'residual': (qs_val - qs_implied) * qty_scale,
                'a_s_eff': a_s_eff,
                'b_s': b_self,
                'Pc_model': pc_val,
                'Pc_abs': pc_abs,
            })

    if not rows:
        return None
    out_path = Path(output_dir) / "supply_cross_terms_diagnosis.csv"
    try:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        return None
    return out_path


def _write_epsS_row_compare(
    nodes: List[Any],
    regional_df: pd.DataFrame,
    output_dir: Optional[str],
    *,
    dict_v3_path: Optional[str] = None,
    use_regional_agg: bool = False,
    cross_terms_scale: Optional[float] = None,
    cross_terms_top_n: Optional[int] = None,
    hist_end_year: int = 2020,
) -> Optional[Path]:
    if not output_dir or regional_df is None or regional_df.empty:
        return None

    if use_regional_agg and not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)

    input_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for n in nodes:
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        region = get_region(n.country, m49=m49, dict_v3_path=dict_v3_path) if use_regional_agg else str(n.country).strip()
        key = (region, str(getattr(n, 'commodity', '')).strip(), int(getattr(n, 'year', 0) or 0))
        Q0 = float(getattr(n, 'Q0', 0.0) or 0.0)
        Q0_w = max(1e-9, Q0)
        rec = input_map.setdefault(key, {'Q0_total': 0.0, 'n_nodes': 0, 'eps_sum': {}})
        rec['Q0_total'] += Q0_w
        rec['n_nodes'] += 1
        eps_row = getattr(n, 'epsS_row', {}) or {}
        for comm, eps in eps_row.items():
            try:
                val = float(eps)
            except Exception:
                continue
            rec['eps_sum'][comm] = rec['eps_sum'].get(comm, 0.0) + val * Q0_w

    for rec in input_map.values():
        q0 = max(1e-9, float(rec.get('Q0_total', 0.0) or 0.0))
        eps_avg = {comm: v / q0 for comm, v in (rec.get('eps_sum', {}) or {}).items()}
        rec['eps_avg'] = eps_avg

    years = {int(hist_end_year)}
    if int(hist_end_year) not in set(pd.to_numeric(regional_df['year'], errors='coerce').dropna().astype(int).unique()):
        try:
            fallback_year = int(pd.to_numeric(regional_df['year'], errors='coerce').dropna().min())
            years = {fallback_year}
        except Exception:
            years = set()

    rows: List[Dict[str, Any]] = []
    for row in regional_df.itertuples(index=False):
        try:
            year = int(getattr(row, 'year'))
        except Exception:
            continue
        if years and year not in years:
            continue
        region = str(getattr(row, 'region'))
        commodity = str(getattr(row, 'commodity'))
        key = (region, commodity, year)
        input_rec = input_map.get(key, {})
        eps_input = dict(input_rec.get('eps_avg', {}) or {})
        eps_model_raw = dict(getattr(row, 'epsS_row', {}) or {})
        eps_model_scaled = _scale_cross_terms(eps_model_raw, cross_terms_scale)
        eps_model_used = _limit_cross_terms(eps_model_scaled, cross_terms_top_n)
        keys = set(eps_input) | set(eps_model_raw) | set(eps_model_scaled) | set(eps_model_used)
        for term_comm in sorted(keys):
            in_val = eps_input.get(term_comm)
            raw_val = eps_model_raw.get(term_comm)
            scaled_val = eps_model_scaled.get(term_comm)
            used_val = eps_model_used.get(term_comm)
            dropped = term_comm in eps_model_scaled and term_comm not in eps_model_used
            ratio_raw = None
            if in_val is not None and raw_val is not None and abs(in_val) > 0:
                ratio_raw = raw_val / in_val
            rows.append({
                'region': region,
                'commodity': commodity,
                'year': year,
                'term_comm': term_comm,
                'eps_input_avg': in_val,
                'eps_model_raw': raw_val,
                'eps_model_scaled': scaled_val,
                'eps_model_used': used_val,
                'ratio_raw_to_input': ratio_raw,
                'dropped_by_topn': bool(dropped),
                'Q0_total': float(input_rec.get('Q0_total', 0.0) or 0.0),
                'n_nodes': int(input_rec.get('n_nodes', 0) or 0),
                'cross_terms_scale': cross_terms_scale,
                'cross_terms_top_n': cross_terms_top_n,
            })

    if not rows:
        return None
    out_path = Path(output_dir) / "epsS_row_compare.csv"
    try:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        return None
    return out_path

# Linear elasticity model with full emissions, MACC, and constraints


def build_linear_regional_model(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    price_bounds: Tuple[float, float] = (1e-6, 1e6),  # A price lower bound of 1e-6 is sufficiently small.
    use_relative_price: bool = False,
    relative_price_bounds: Tuple[float, float] = (0.1, 10.0),
    price_bounds_mode: str = 'absolute',  # 'absolute' | 'p0_mult'
    price_bounds_p0_mult: Tuple[float, float] = (0.1, 10.0),
    price_wedge_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    price_wedge_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    price_wedge_by_region: Optional[Dict[str, float]] = None,
    market_clearing_mode: str = 'country_trade',
    armington_sigma_by_comm: Optional[Dict[Any, float]] = None,
    trade_base_net_import: Optional[Dict[Tuple[str, str], float]] = None,
    trade_base_volume: Optional[Dict[Tuple[str, str], float]] = None,
    trade_cap_region_volume: Optional[Dict[Tuple[str, str], float]] = None,
    trade_cap_region_map: Optional[Dict[Any, str]] = None,
    trade_cap_ratio: Any = None,
    trade_cap_exempt_pairs: Optional[set] = None,
    armington_trade_scale: Optional[float] = None,
    armington_trade_slack_penalty: Optional[float] = None,
    supply_curtailment_enabled: bool = False,
    supply_curtailment_penalty: Optional[float] = None,
    zero_price_shutdown_enabled: bool = False,
    zero_demand_production_shutdown: bool = False,
    qty_scale: float = 1.0,
    land_scale: float = 1.0,
    qty_bounds: Tuple[float, float] = (1e-6, 1e12),
    dict_v3_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    gurobi_log_path: Optional[str] = None,
    solver_method: Optional[int] = None,
    solver_threads: Optional[int] = None,
    # Population and income
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    # Emissions and abatement parameters
    
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    # LUC in optimization
    luc_opt_mode: str = 'none',  # 'none' | 'explicit' | 'iterative'
    luc_params: Optional[Dict[str, Any]] = None,
    luc_shift_area_mode: str = 'abs',
    luc_penalty_by_region_year: Optional[Dict[Tuple[str, int], Dict[str, float]]] = None,
    
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    land_soft_constraints_enabled: bool = False,
    land_slack_max_rate: Optional[float] = None,
    land_slack_penalty: Optional[float] = None,
    land_delta_anchor_to_available_stock: bool = False,
    grass_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,  # Grassland area {(region, year): ha}
    forest_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,  # Forest area {(region, year): ha}
    forest_global_target_slack_enabled: bool = False,
    forest_global_target_slack_penalty: Optional[float] = None,
    forest_global_target_slack_max_rate: Optional[float] = None,
    forest_nonneg_ratio: float = 1.0,
    cropland_nonforest_expand_ratio: float = 1.0,
    pasture_nonforest_expand_ratio: float = 1.0,
    base_cropland_by_region: Optional[Dict[str, float]] = None,  # LUH2 base-period cropland area {(region): ha}
    base_grassland_by_region: Optional[Dict[str, float]] = None,  # LUH2 base-period grassland area {(region): ha}
    base_forest_by_region: Optional[Dict[str, float]] = None,  # LUH2/FAO base-period forest area {(region): ha}
    background_forest_to_cropland_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_forest_to_grassland_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_cropland_to_forest_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_grassland_to_forest_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    land_demand_calibration_mode: str = 'none',
    yield_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,  # Target-year yield {(region/country, commodity, year): t/ha}
    yield_t_per_ha_default: float = 3.0,
    grassland_method: str = 'dynamic',  # 'dynamic' (approach A: optimization variables) or 'static' (approach B: iteration)
    grassland_conversion_penalty: float = 0.0,
    grassland_to_cropland_cost_mode: str = 'per_ha_cost',
    cropland_to_grassland_penalty: float = 0.0,
    land_conversion_allocation_mode: str = 'priority_nonforest_pasture_forest',
    land_conversion_priority_penalty_per_ha: float = 1e6,
    land_priority_weight_grassland_to_cropland: float = 1.0,
    land_priority_weight_forest_to_cropland: float = 100.0,
    land_priority_weight_forest_to_grassland: float = 100.0,
    luc_direct_carbon_price: bool = False,
    limit_reforestation_to_released_ag_land: bool = True,
    prevent_land_conversion_cycles: bool = True,
    reforestation_physical_cap_enabled: bool = True,
    reforestation_max_forest_increase_ratio: Optional[float] = 0.30,
    # Growth constraints
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    hist_max_production: Optional[Dict[Tuple[str, str], float]] = None,
    future_last_only: bool = True,
    hist_max_small_prod_exempt_t: Optional[float] = None,
    hist_max_small_prod_floor_t: Optional[float] = None,
    # Scenario parameters (Phase 2)
    tax_unit_adder: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    waste_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    losses_ratio_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_mode: Optional[str] = None,
    feed_crop_link_base: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_coeff: Optional[Dict[Tuple[str, str, str, int], float]] = None,
    feed_background_crop: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_background_pasture: Optional[Dict[Tuple[str, int], float]] = None,
    feed_crop_link_livestock: Optional[List[str]] = None,
    feed_crop_link_override: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_credit: Optional[Dict[Tuple[str, str, int], float]] = None,
    ruminant_intake_cap: Optional[Dict[Tuple[str, int], float]] = None,
    ruminant_commodities: Optional[List[str]] = None,
    # Market imbalance limits
    max_slack_rate: Optional[float] = 0.1,  # Annual shortage/surplus stock cap as a fraction of total energy demand (e.g. 0.1=10%; None=unlimited).
    max_shortage_slack_rate: Any = "inherit",
    max_excess_slack_rate: Any = "inherit",
    slack_penalty: Optional[float] = 1e6,
    disable_production_cost_term: bool = True,  # True: disable the production cost term (default).
    production_cost_weight: float = 1.0,  # Optional production cost weight
    # Cross-elasticity term clipping
    cross_terms_top_n: Optional[int] = None,
    cross_terms_scale: Optional[float] = None,
    # Unit-cost method parameters
    cost_calculation_method: str = 'MACC',  # 'MACC' (default) or 'unit_cost'
    unit_cost_data: Optional[Dict[Tuple[str, str], float]] = None,  # {(region/country, process): USD/tCO2e}
    process_cost_mapping: Optional[Dict[str, str]] = None,  # {pipeline_process: cost_process_name}
    baseline_scenario_result: Optional[Dict[str, Any]] = None,  # BASE scenario results {'Qs': {(region, comm, year): value}}
    active_strategy_cost_keys: Optional[Sequence[str]] = None,
    strategy_cost_regions: Optional[Sequence[str]] = None,
    cost_database_metadata: Optional[Mapping[str, Any]] = None,
    cost_strategy_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    # Demand projection method
    demand_method: str = 'elasticity',  # 'elasticity' | 'nutrition' | 'nutrition_band' | 'nutrition_anchor'
    nutrition_profile_xlsx: Optional[str] = None,
    nutrition_profile_sheet: Any = 0,
    nutrition_indicator: str = 'energy',
    nutrition_use_baseyear_for_future: bool = True,
    nutrition_band_epsilon: float = 0.1,
    nutrition_feed_t_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    nutrition_residual_demand_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    bioenergy_crop_demand_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    energy_crop_land_requirement_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    enable_output_diagnostics: bool = False,
    enable_verbose_logging: bool = True,
) -> gp.Model:
    """
    Build the full linear elasticity regional model.
    
    Features:
    1. Linear elasticity equations without log transformations or PWL, for fast solving.
    2. Aggregation into 34 regions using dict_v3 Region_market_agg.
    3. Global market clearing.
    
    Full elasticities, consistent with S3_0_ds_emis_mc_full.py:
    - Supply: epsilon_supply (price), eta_yield (yield), eta_temp (temperature).
    - Demand: epsilon_demand (price), epsilon_pop (population), epsilon_income (income).
    
    Full functionality:
    - Emissions as e0_by_proc * Qs with abatement decisions per process driven by MACC
    - Land carbon price objective term for LULUCF processes
    - Optional nutrition and land constraints
    - Scenario support: tax_unit, feed_reduction, ruminant_intake_cap
    - Monte Carlo simulation support via LinearModelCache
    
    Parameters:
        nodes: List of country-level nodes.
        commodities: List of commodities.
        years: List of years.
        price_bounds: Price bounds (min, max).
        use_relative_price: use Pc_rel = Pc / P0_ref (per-commodity)
        relative_price_bounds: bounds for Pc_rel when use_relative_price is True
        price_wedge_by_region_comm_year: regional price wedge by (region, commodity, year)
        price_wedge_by_region_comm: regional price wedge by (region, commodity)
        price_wedge_by_region: regional price wedge by region
        market_clearing_mode: 'country_trade' (default) | 'regional_armington'
        armington_sigma_by_comm: Armington elasticity by commodity or (region, commodity)
        trade_base_net_import: base-year net import {(region, commodity): value}
        trade_base_volume: base-year trade volume {(region, commodity): value}
        armington_trade_scale: optional scale factor for Armington trade slope
        armington_trade_slack_penalty: penalty for Armington deviation slack (None -> use slack_penalty)
        qty_scale: ???????Qs/Qd ?????? qty_scale?
        qty_bounds: Quantity bounds (min, max).
        dict_v3_path: Path to dict_v3.xlsx.
        macc_path: Path to MACC data (pickle file).
        land_carbon_price_by_year: Land carbon price {year: price_per_tCO2e}.
        nutrition_rhs: Nutrition constraint RHS {(region, year): min_calories}.
        nutrient_per_unit_by_comm: Nutrition per commodity unit {commodity: value}.
        land_area_limits: Land area caps {(region, year): max_ha}.
        yield_t_per_ha_default: Default yield (tonnes/hectare).
        grassland_method: Grassland handling method.
            - 'dynamic' (approach A, default): grassland is an optimization expression, grassland_ha = sum(coef * Qs).
            - 'static' (approach B): grassland is exogenous and updated iteratively until convergence.
        max_growth_rate_per_period: Maximum growth rate.
        max_decline_rate_per_period: Maximum decline rate.
        hist_end_year: Last historical year.
        hist_max_production: Historical production anchor {(region, commodity): max_t}, limiting future production.
        tax_unit_adder: Unit tax {(region, commodity, year): $/t}, affecting supply-side net prices.
        feed_reduction_by: Feed intensity change {(region, commodity, year): rate in [-1, 1]}.
        ruminant_intake_cap: Ruminant demand cap {(region, year): cap_in_t}.
        ruminant_commodities: Ruminant commodities; defaults to beef and sheep meat.
    """
    logger = logging.getLogger(__name__)
    if supply_curtailment_enabled:
        logger.warning("[LINEAR] supply_curtailment_enabled is globally disabled; ignore requested True")
        supply_curtailment_enabled = False
    cross_fix_supply = {'count': 0, 'limit': CROSS_COEF_FIX_LOG_LIMIT}
    cross_fix_demand = {'count': 0, 'limit': CROSS_COEF_FIX_LOG_LIMIT}

    yield_override_map: Dict[Tuple[str, str, int], float] = {}
    if yield_by_region_comm_year:
        for key, val in yield_by_region_comm_year.items():
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            region_raw, comm_raw, year_raw = key
            try:
                year_i = int(year_raw)
                y_val = float(val)
            except Exception:
                continue
            if not np.isfinite(y_val) or y_val <= 0:
                continue
            region_s = str(region_raw).strip()
            comm_s = str(comm_raw).strip()
            if not region_s or not comm_s:
                continue
            yield_override_map[(region_s, comm_s, year_i)] = y_val
            m49_s = _norm_m49_code(region_s)
            if m49_s:
                yield_override_map[(m49_s, comm_s, year_i)] = y_val

    yield_override_hits = 0

    def _target_year_yield(region_raw: Any, commodity_raw: Any, year_raw: Any, fallback: float) -> float:
        nonlocal yield_override_hits
        if not yield_override_map:
            return fallback
        try:
            year_i = int(year_raw)
        except Exception:
            return fallback
        region_s = str(region_raw).strip()
        comm_s = str(commodity_raw).strip()
        candidates = []
        if region_s:
            candidates.append(region_s)
        m49_s = _norm_m49_code(region_s)
        if m49_s and m49_s not in candidates:
            candidates.append(m49_s)
        for region_key in candidates:
            y_val = yield_override_map.get((region_key, comm_s, year_i))
            if y_val is not None:
                yield_override_hits += 1
                return float(y_val)
        return fallback
    
    # Build model input table. Skip aggregation when regional aggregation is disabled.
    use_regional_agg = is_region_aggregation_enabled()
    if use_regional_agg:
        regional_df = aggregate_nodes_to_regions(
            nodes,
            dict_v3_path=dict_v3_path,
            population_by_country_year=population_by_country_year,
            income_mult_by_country_year=income_mult_by_country_year,
            hist_end_year=hist_end_year,
        )
    else:
        base_p0_by_country_comm: Dict[Tuple[str, str], float] = {}
        for n in nodes:
            if getattr(n, 'year', None) != hist_end_year:
                continue
            key_country = str(getattr(n, 'country', '')).strip()
            if not key_country:
                continue
            try:
                p0_val = float(getattr(n, 'P0', 0.0) or 0.0)
            except Exception:
                continue
            if np.isfinite(p0_val) and p0_val > 0:
                base_p0_by_country_comm[(key_country, getattr(n, 'commodity', None))] = p0_val
        records = []
        for n in nodes:
            pop_base = 1.0
            pop_t = 1.0
            inc_base = 1.0
            inc_t = 1.0
            if population_by_country_year:
                pop_base = float(population_by_country_year.get((n.country, hist_end_year), 1.0) or 1.0)
                pop_t = float(population_by_country_year.get((n.country, n.year), pop_base) or pop_base)
            if income_mult_by_country_year:
                inc_base = float(income_mult_by_country_year.get((n.country, hist_end_year), 1.0) or 1.0)
                inc_t = float(income_mult_by_country_year.get((n.country, n.year), inc_base) or inc_base)

            meta = getattr(n, 'meta', {}) or {}
            yield0 = float(meta.get('yield0', 0.0) or 0.0)
            yield0 = _target_year_yield(n.country, n.commodity, n.year, yield0)
            grassland_coef = float(meta.get('grassland_coef', 0.0) or 0.0)

            key_country = str(getattr(n, 'country', '')).strip()
            p0_val = getattr(n, 'P0', 0.0) or 0.0
            if getattr(n, 'year', None) > hist_end_year:
                base_p0 = base_p0_by_country_comm.get((key_country, getattr(n, 'commodity', None)))
                if base_p0 is not None:
                    p0_val = base_p0
            try:
                p0_val = float(p0_val)
            except Exception:
                p0_val = 0.0
            if not np.isfinite(p0_val) or p0_val <= 0:
                base_p0 = base_p0_by_country_comm.get((key_country, getattr(n, 'commodity', None)))
                if base_p0 is not None:
                    p0_val = base_p0
                else:
                    p0_val = 1.0

            records.append({
                'region': str(n.country).strip(),
                'commodity': n.commodity,
                'year': n.year,
                'Q0': getattr(n, 'Q0', 0.0) or 0.0,
                'D0': getattr(n, 'D0', 0.0) or 0.0,
                'P0': p0_val,
                'yield0': yield0,
                'grassland_coef': grassland_coef,
                'livestock_feed_account': bool(meta.get('livestock_feed_account', False)),
                'feed_only_entry': bool(meta.get('feed_only_entry', False)),
                'livestock_background_only': bool(meta.get('livestock_background_only', False)),
                'livestock_feed_unidentified': bool(meta.get('livestock_feed_unidentified', False)),
                'seed_coefficient': float(meta.get('seed_coefficient', 0.0)),
                'seed_reference_t': float(meta.get('seed_reference_t', 0.0)),
                'seed_coefficient_unidentified': bool(meta.get('seed_coefficient_unidentified', False)),
                'eps_supply': getattr(n, 'eps_supply', 0.0) or 0.0,
                'eps_supply_yield': getattr(n, 'eps_supply_yield', 0.0) or 0.0,
                'eps_supply_temp': getattr(n, 'eps_supply_temp', 0.0) or 0.0,
                'Ymult': getattr(n, 'Ymult', 1.0) or 1.0,
                'Tmult': getattr(n, 'Tmult', 1.0) or 1.0,
                'eps_demand': getattr(n, 'eps_demand', 0.0) or 0.0,
                'eps_pop_demand': getattr(n, 'eps_pop_demand', 0.0) or 0.0,
                'eps_income_demand': getattr(n, 'eps_income_demand', 0.0) or 0.0,
                'epsS_row': dict(getattr(n, 'epsS_row', {}) or {}),
                'epsD_row': dict(getattr(n, 'epsD_row', {}) or {}),
                'pop_base': pop_base,
                'pop_t': pop_t,
                'inc_base': inc_base,
                'inc_t': inc_t,
            })
        regional_df = pd.DataFrame(records)
        if yield_override_map:
            logger.info(
                "[LINEAR] target-year yield overrides applied to land coefficients: hits=%d, map_keys=%d",
                yield_override_hits,
                len(yield_override_map),
            )
    
    # Retain only commodities in the commodities list.
    # This removes non-commodity items such as emissions processes and land categories.
    original_count = len(regional_df)
    regional_df = regional_df[regional_df['commodity'].isin(commodities)]
    filtered_count = original_count - len(regional_df)
    if filtered_count > 0:
        logger.info(f"[LINEAR] 已过滤 {filtered_count} 行非商品数据")
    
    # Further filter region-commodity pairs with very small Q0.
    # If both Q0 and D0 for a region-commodity pair are very small in hist_end_year,
    # remove that pair's data for all years.
    MIN_Q0_THRESHOLD = 1e-3  # Minimum production threshold (kt/year)
    
    # Extract base-year data for screening.
    base_year_df = regional_df[regional_df['year'] == hist_end_year].copy()
    base_year_df['tiny_q0'] = (base_year_df['Q0'] < MIN_Q0_THRESHOLD) & (base_year_df['D0'] < MIN_Q0_THRESHOLD)
    
    # Flag region-commodity pairs to remove.
    tiny_pairs = base_year_df[base_year_df['tiny_q0']][['region', 'commodity']].drop_duplicates()
    
    if len(tiny_pairs) > 0:
        logger.info(f"[LINEAR] 发现 {len(tiny_pairs)} 个区域-商品组合在基期({hist_end_year})的Q0和D0都 < {MIN_Q0_THRESHOLD:.0e}")
        
        # Build the (region, commodity) exclusion set, preserving explicit future bioenergy demand.
        # S4 may add these nodes as Q0=D0=0 placeholders; filtering by base-period size
        # would silently remove their demand from regional market-balance constraints.
        tiny_set = {
            (str(region).strip(), str(commodity).strip())
            for region, commodity in tiny_pairs.itertuples(index=False, name=None)
        }
        bioenergy_active_keys = set()
        for key, value in (bioenergy_crop_demand_by_country_comm_year or {}).items():
            try:
                region, commodity, year = key
                value_f = float(value or 0.0)
                year_i = int(year)
            except Exception:
                continue
            if (
                year_i > int(hist_end_year)
                and np.isfinite(value_f)
                and value_f > 0.0
            ):
                bioenergy_active_keys.add(
                    (
                        str(region).strip(),
                        str(commodity).strip(),
                        year_i,
                    )
                )

        feed_required_keys={(str(r),str(c),int(t)) for (r,c,l,t),v in (feed_crop_link_coeff or {}).items() if float(v)>0}
        feed_required_keys.update((str(r),str(c),int(t)) for (r,c,t),v in (feed_background_crop or {}).items() if float(v)>0)
        bioenergy_active_keys.update(feed_required_keys)
        feed_required_pairs={(r,c) for r,c,t in feed_required_keys}
        bioenergy_active_pairs = {
            (region, commodity)
            for region, commodity, _ in bioenergy_active_keys
        }
        protected_pairs = tiny_set.intersection(bioenergy_active_pairs)
        if protected_pairs:
            logger.info(
                "[BIOENERGY] 将仅保留 %d 个基期零规模区域-商品组合中的 "
                "%d 个显式未来需求节点",
                len(protected_pairs),
                sum(
                    1
                    for region, commodity, _ in bioenergy_active_keys
                    if (region, commodity) in protected_pairs
                ),
            )

        if tiny_set:
            normalized_rows = zip(
                regional_df['region'].astype(str).str.strip(),
                regional_df['commodity'].astype(str).str.strip(),
                pd.to_numeric(regional_df['year'], errors='coerce'),
            )
            filter_mask = pd.Series(
                [
                    (region, commodity) in tiny_set
                    and (region, commodity) not in feed_required_pairs
                    and (
                        not np.isfinite(year)
                        or (region, commodity, int(year)) not in bioenergy_active_keys
                    )
                    for region, commodity, year in normalized_rows
                ],
                index=regional_df.index,
                dtype=bool,
            )

            filtered_rows = int(filter_mask.sum())
            logger.info(f"[LINEAR] 将过滤 {filtered_rows} 行数据（跨所有年份）")

            # Record commodities actually removed.
            filtered_comms = sorted({commodity for _, commodity in tiny_set})
            logger.info(f"[LINEAR] 涉及商品: {', '.join(filtered_comms[:10])}")
            if len(filtered_comms) > 10:
                logger.info(f"[LINEAR] ... 以及其他 {len(filtered_comms)-10} 个")

            # Apply the filter.
            regional_df = regional_df[~filter_mask]

    if enable_output_diagnostics:
        eps_compare_path = _write_epsS_row_compare(
            nodes,
            regional_df,
            output_dir,
            dict_v3_path=dict_v3_path,
            use_regional_agg=use_regional_agg,
            cross_terms_scale=cross_terms_scale,
            cross_terms_top_n=cross_terms_top_n,
            hist_end_year=hist_end_year,
        )
        if eps_compare_path:
            logger.info("[LINEAR] epsS_row compare saved to %s", eps_compare_path)

    regions = regional_df['region'].unique().tolist()
    actual_commodities = regional_df['commodity'].unique().tolist()
    comm_set = set(commodities)
    
    # Aggregate emissions intensities.
    e0_by_region = aggregate_emissions_to_regions(nodes, dict_v3_path=dict_v3_path)
    
    # Load MACC data.
    macc_df = _read_macc(macc_path)
    has_macc = not macc_df.empty
    
    logger.info(f"[LINEAR] 区域数: {len(regions)}, 商品数: {len(actual_commodities)}, 年份数: {len(years)}")
    if enable_verbose_logging:
        logger.info(f"[LINEAR] 参与模拟的区域: {regions}")
        logger.info(f"[LINEAR] 参与模拟的商品 ({len(actual_commodities)}个): {', '.join(sorted(actual_commodities))}")
    logger.info(f"[LINEAR] MACC 数据: {'已加载' if has_macc else '无'}")

    market_clearing_mode = str(market_clearing_mode or 'country_trade').strip().lower()
    if market_clearing_mode in {'regional', 'region', 'armington', 'regional_armington'}:
        market_clearing_mode = 'regional_armington'
    elif market_clearing_mode in {'country', 'country_trade', 'national'}:
        market_clearing_mode = 'country_trade'
    else:
        logger.warning("[LINEAR] unknown market_clearing_mode=%s; fallback to country_trade", market_clearing_mode)
        market_clearing_mode = 'country_trade'
    use_regional_price = market_clearing_mode == 'regional_armington'
    if use_regional_price:
        logger.info("[LINEAR] market_clearing_mode=regional_armington (regional prices + Armington trade)")

    armington_sigma_by_comm = armington_sigma_by_comm or {}
    trade_base_net_import = trade_base_net_import or {}
    trade_base_volume = trade_base_volume or {}
    trade_cap_region_volume = trade_cap_region_volume or {}
    trade_cap_region_map = trade_cap_region_map or {}
    trade_cap_ratio_val, trade_cap_ratio_by_key = _normalize_trade_cap_ratio_arg(trade_cap_ratio)
    trade_cap_enabled = trade_cap_ratio_val is not None or bool(trade_cap_ratio_by_key)
    use_region_trade_cap = False
    if trade_cap_enabled:
        if trade_cap_region_volume and trade_cap_region_map:
            use_region_trade_cap = True
            if trade_cap_ratio_by_key:
                logger.info(
                    "[LINEAR] dynamic trade_cap_ratio keys=%d; use grouped trade cap",
                    len(trade_cap_ratio_by_key),
                )
            else:
                logger.info("[LINEAR] trade_cap_ratio=%.6g; use grouped trade cap", trade_cap_ratio_val)
        elif not trade_base_volume:
            logger.warning("[LINEAR] trade_cap_ratio set but trade_base_volume is empty; ignore cap")
            trade_cap_enabled = False
    try:
        armington_trade_scale = float(armington_trade_scale) if armington_trade_scale is not None else 1.0
    except Exception:
        armington_trade_scale = 1.0
    if not np.isfinite(armington_trade_scale) or armington_trade_scale <= 0:
        logger.warning("[LINEAR] armington_trade_scale invalid; fallback to 1.0")
        armington_trade_scale = 1.0

    def _normalize_feed_link_mode(mode: Optional[str]) -> str:
        if mode is None:
            return 'off'
        m = str(mode).strip().lower()
        if m in ('', 'none', 'off', 'false', '0'):
            return 'off'
        if m in ('iter', 'iterative', 'loop', 'iteration'):
            return 'iterative'
        if m in ('dynamic', 'dynamic_constraints', 'constraints', 'constraint'):
            return 'dynamic_constraints'
        return m

    feed_link_mode = _normalize_feed_link_mode(feed_crop_link_mode)

    # Nutrition-driven demand (future years)
    method = str(demand_method).lower()
    strict_nutrition = (method == 'nutrition')
    nutrition_supply_driven = (method == 'nutrition')
    nutrition_demand_map: Dict[Tuple[str, str, int], float] = {}
    nutrition_residual_demand_map: Dict[Tuple[str, str, int], float] = {}
    bioenergy_crop_demand_map: Dict[Tuple[str, str, int], float] = {}
    energy_crop_land_requirement_map: Dict[Tuple[str, int], float] = {}
    nonfood_commodities: set = set()
    crop_land_commodities: set = _load_crop_commodities(dict_v3_path)
    if crop_land_commodities:
        logger.info("[LINEAR] crop land commodities loaded for Qs-LUC cap: %d", len(crop_land_commodities))
    country_by_m49: Dict[str, str] = {}
    if waste_reduction_by or losses_ratio_by:
        country_by_m49 = _load_m49_to_country(dict_v3_path)
    if nutrition_residual_demand_by_country_comm_year:
        for key, val in nutrition_residual_demand_by_country_comm_year.items():
            try:
                r_key, j_key, t_key = key
                t_int = int(t_key)
                val_f = float(val or 0.0)
            except Exception:
                continue
            if not np.isfinite(val_f) or val_f <= 0.0:
                continue
            norm_key = (str(r_key).strip(), str(j_key).strip(), t_int)
            nutrition_residual_demand_map[norm_key] = (
                nutrition_residual_demand_map.get(norm_key, 0.0) + val_f
            )
    if bioenergy_crop_demand_by_country_comm_year:
        for key, val in bioenergy_crop_demand_by_country_comm_year.items():
            try:
                r_key, j_key, t_key = key
                t_int = int(t_key)
                val_f = float(val or 0.0)
            except Exception:
                continue
            if not np.isfinite(val_f) or val_f <= 0.0:
                continue
            norm_key = (str(r_key).strip(), str(j_key).strip(), t_int)
            bioenergy_crop_demand_map[norm_key] = (
                bioenergy_crop_demand_map.get(norm_key, 0.0) + val_f
            )
        logger.info(
            "[BIOENERGY] explicit crop demand loaded: keys=%d total=%.6g t",
            len(bioenergy_crop_demand_map),
            sum(bioenergy_crop_demand_map.values()),
        )
    if energy_crop_land_requirement_by_region_year:
        for key, val in energy_crop_land_requirement_by_region_year.items():
            try:
                r_key, t_key = key
                t_int = int(t_key)
                val_f = float(val or 0.0)
            except Exception:
                continue
            if not np.isfinite(val_f) or val_f <= 0.0:
                continue
            norm_key = (str(r_key).strip(), t_int)
            energy_crop_land_requirement_map[norm_key] = (
                energy_crop_land_requirement_map.get(norm_key, 0.0) + val_f
            )
        logger.info(
            "[BIOENERGY] dedicated energy-crop land loaded: keys=%d total=%.6g ha",
            len(energy_crop_land_requirement_map),
            sum(energy_crop_land_requirement_map.values()),
        )
    if method in {'nutrition', 'nutrition_band', 'nutrition_anchor'}:
        if not population_by_country_year:
            raise ValueError("营养驱动需求需要 population_by_country_year")
        if not nutrition_profile_xlsx:
            try:
                from config_paths import get_input_base
                nutrition_profile_xlsx = str(Path(get_input_base()) / 'Driver' / 'Nutrition' / 'Nutrition_profile_rescaled.xlsx')
            except Exception:
                nutrition_profile_xlsx = None
        nutrition_demand_map = build_nutrition_demand_map(
            nutrition_xlsx=nutrition_profile_xlsx or '',
            dict_v3_path=dict_v3_path,
            indicator=nutrition_indicator,
            years=years,
            population_by_country_year=population_by_country_year,
            nutrition_profile_sheet=nutrition_profile_sheet,
            use_regional_agg=_USE_REGIONAL_AGGREGATION,
            hist_end_year=hist_end_year,
            use_baseyear_for_future=nutrition_use_baseyear_for_future,
            feed_t_by_country_comm_year=nutrition_feed_t_by_country_comm_year,
            waste_reduction_by_country_comm_year=waste_reduction_by,
            losses_ratio_by_country_comm_year=losses_ratio_by,
            country_by_m49=country_by_m49 if country_by_m49 else None,
            separate_nonfood_demand=bool(nutrition_residual_demand_map) or feed_link_mode == 'dynamic_constraints',
        )
        nonfood_commodities = _load_nonfood_commodities(dict_v3_path)
        if not nutrition_demand_map:
            msg = f"[LINEAR] 营养驱动需求为空，demand_method={method}"
            if strict_nutrition:
                raise ValueError(msg)
            logger.warning(msg)
        if nutrition_supply_driven:
            logger.info(
                "[LINEAR] demand_method=nutrition: future supply is demand/land/trade driven; "
                "supply elasticity equations are disabled"
            )

    loss_ratio_by_item: Dict[Tuple[str, str], float] = {}
    demand_item_map: Dict[str, List[str]] = {}
    normalize_comp_item = None
    region_to_m49: Dict[str, str] = {v: k for k, v in country_by_m49.items() if v} if country_by_m49 else {}
    if method == 'elasticity' and (waste_reduction_by or losses_ratio_by):
        try:
            from config_paths import get_input_base
            comp_path = str(Path(get_input_base()) / 'Production_Trade' / 'Demand_composition.xlsx')
            loss_ratio_by_item = _load_demand_composition_losses_ratio(comp_path)
            _, demand_item_map = _load_item_demand_extra_and_map(dict_v3_path)
            normalize_comp_item = _normalize_comp_item_name
        except Exception as exc:
            logger.warning("[LINEAR] 读取Losses比例失败: %s", exc)
            loss_ratio_by_item = {}
            demand_item_map = {}
    
    # Count region-commodity pairs simulated in future years.
    future_df = regional_df[regional_df['year'] > hist_end_year]
    if len(future_df) > 0:
        future_pairs = future_df.groupby(['region', 'commodity']).size().reset_index(name='count')
        logger.info(f"[LINEAR] 未来年份参与模拟: {len(future_pairs)} 个区域-商品组合")
        logger.info(f"[LINEAR] 覆盖 {future_df['year'].nunique()} 个未来年份: {sorted(future_df['year'].unique().tolist())}")
    
    # Create the model.
    m = gp.Model("nzf_linear_regional_full")
    m.setParam('OutputFlag', 1)
    m.setParam('LogToConsole', 1 if enable_verbose_logging else 0)
    # Set the Gurobi log file.
    if gurobi_log_path:
        m.Params.LogFile = str(gurobi_log_path)
    # Improve numerical stability to avoid spurious infeasibility from wide coefficient ranges.
    m.setParam('NumericFocus', 3)  # 0=auto, 1=moderate, 2=aggressive, 3=very aggressive
    m.setParam('ScaleFlag', 2)  # Automatically scale the coefficient matrix to improve its numerical range.
    m.setParam('FeasibilityTol', 1e-6)  # Relax feasibility tolerance (default 1e-6; try 1e-5 or 1e-4).
    m.setParam('OptimalityTol', 1e-6)   # Relax optimality tolerance.
    solver_method_val: Optional[int] = None
    if solver_method is not None:
        try:
            solver_method_val = int(solver_method)
        except Exception:
            logger.warning("[LINEAR] invalid solver_method=%r; ignore", solver_method)
    if solver_method_val is not None:
        try:
            m.setParam('Method', solver_method_val)
            logger.info("[LINEAR] Gurobi Method=%s", solver_method_val)
        except Exception as exc:
            logger.warning("[LINEAR] failed to set Gurobi Method=%r: %s", solver_method_val, exc)
    solver_threads_val: Optional[int] = None
    if solver_threads is not None:
        try:
            solver_threads_val = int(solver_threads)
        except Exception:
            logger.warning("[LINEAR] invalid solver_threads=%r; ignore", solver_threads)
    if solver_threads_val is not None and solver_threads_val > 0:
        try:
            m.setParam('Threads', solver_threads_val)
            logger.info("[LINEAR] Gurobi Threads=%s", solver_threads_val)
        except Exception as exc:
            logger.warning("[LINEAR] failed to set Gurobi Threads=%r: %s", solver_threads_val, exc)
    
    use_relative_price = bool(use_relative_price)
    try:
        qty_scale = float(qty_scale or 1.0)
    except Exception:
        qty_scale = 1.0
    if not np.isfinite(qty_scale) or qty_scale <= 0:
        logger.warning("[LINEAR] qty_scale invalid; fallback to 1.0")
        qty_scale = 1.0
    try:
        land_scale = float(land_scale or 1.0)
    except Exception:
        land_scale = 1.0
    if not np.isfinite(land_scale) or land_scale <= 0:
        logger.warning("[LINEAR] land_scale invalid; fallback to 1.0")
        land_scale = 1.0
    zero_price_shutdown_enabled = bool(zero_price_shutdown_enabled)
    inv_qty_scale = 1.0 / qty_scale
    inv_land_scale = 1.0 / land_scale
    Qmin_raw, Qmax_raw = qty_bounds
    Qmin = float(Qmin_raw) * inv_qty_scale
    Qmax = float(Qmax_raw) * inv_qty_scale
    logger.info(f"[LINEAR] qty_scale={qty_scale} (Qs/Qd scaled by 1/qty_scale)")
    logger.info(f"[LINEAR] land_scale={land_scale} (land/LUC variables scaled by 1/land_scale)")
    band_eps = 0.0
    try:
        band_eps = float(nutrition_band_epsilon or 0.0)
    except Exception:
        band_eps = 0.0
    if not np.isfinite(band_eps) or band_eps < 0:
        band_eps = 0.0
    
    # Indices
    idx = {}
    for _, row in regional_df.iterrows():
        key = (row['region'], row['commodity'], row['year'])
        idx[key] = row.to_dict()
    idx_keys = list(idx.keys())
    future_keys = [key for key in idx_keys if int(key[2]) > int(hist_end_year)]
    future_key_set = {
        (str(region).strip(), str(commodity).strip(), int(year))
        for region, commodity, year in future_keys
    }
    unlinked_bioenergy_demand = {
        key: value
        for key, value in bioenergy_crop_demand_map.items()
        if int(key[2]) > int(hist_end_year)
        and float(value) > 0.0
        and key not in future_key_set
    }
    if unlinked_bioenergy_demand:
        sample = list(unlinked_bioenergy_demand.items())[:5]
        sample_text = "; ".join(
            f"{key}={value:.6g} t" for key, value in sample
        )
        raise ValueError(
            "BIOENERGY_MARKET_LINK_MISSING: "
            f"{len(unlinked_bioenergy_demand)} positive future demand keys have "
            f"no solver market node; sample: {sample_text}"
        )
    region_keys_by_comm_year: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    future_region_keys_by_comm_year: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    future_keys_by_region_group_comm_year: Dict[Tuple[str, str, int], List[Tuple[str, str, int]]] = defaultdict(list)

    def _trade_cap_group_for(region: str, commodity: str) -> Optional[str]:
        if not trade_cap_region_map:
            return None
        val = trade_cap_region_map.get((region, commodity))
        if val is None:
            val = trade_cap_region_map.get(region)
        if val is None:
            return None
        val_s = str(val).strip()
        return val_s or None

    for r, j, t in idx_keys:
        region_keys_by_comm_year[(j, t)].append(r)
        if int(t) > int(hist_end_year):
            future_region_keys_by_comm_year[(j, t)].append(r)
            region_group = _trade_cap_group_for(r, j)
            if region_group:
                future_keys_by_region_group_comm_year[(region_group, j, t)].append((r, j, t))

    price_bounds_by_comm, price_ref_by_comm, (Pmin, Pmax), price_bounds_mode_norm = _compute_price_bounds_by_comm(
        idx=idx,
        commodities=commodities,
        regions=regions,
        hist_end_year=hist_end_year,
        price_bounds=price_bounds,
        use_relative_price=use_relative_price,
        relative_price_bounds=relative_price_bounds,
        price_bounds_mode=price_bounds_mode,
        price_bounds_p0_mult=price_bounds_p0_mult,
    )
    if not use_relative_price:
        if price_bounds_mode_norm == 'p0_mult':
            logger.info(
                "[LINEAR] price bounds mode=p0_mult, mult=%s, global=[%.6g, %.6g]",
                str(price_bounds_p0_mult),
                Pmin,
                Pmax,
            )
        else:
            logger.info(
                "[LINEAR] price bounds mode=absolute, global=[%.6g, %.6g]",
                Pmin,
                Pmax,
            )

    def _price_bounds_for_comm(comm: str) -> Tuple[float, float]:
        return price_bounds_by_comm.get(comm, (Pmin, Pmax))

    price_wedge_by_region_comm_year_norm = _normalize_price_wedge_rcy(price_wedge_by_region_comm_year)
    price_wedge_by_region_comm_norm = _normalize_price_wedge_rc(price_wedge_by_region_comm)
    price_wedge_by_region_norm = _normalize_price_wedge_r(price_wedge_by_region)
    if price_wedge_by_region_comm_year_norm or price_wedge_by_region_comm_norm or price_wedge_by_region_norm:
        logger.info(
            "[LINEAR] price wedge enabled: rcy=%d rc=%d r=%d",
            len(price_wedge_by_region_comm_year_norm),
            len(price_wedge_by_region_comm_norm),
            len(price_wedge_by_region_norm),
        )

    def _price_wedge(region: Any, commodity: Any, year: Any) -> float:
        return _get_price_wedge(
            region,
            commodity,
            year,
            by_rcy=price_wedge_by_region_comm_year_norm,
            by_rc=price_wedge_by_region_comm_norm,
            by_r=price_wedge_by_region_norm,
        )

    def _armington_sigma(region: Any, commodity: Any) -> Optional[float]:
        if not armington_sigma_by_comm:
            return None
        if (region, commodity) in armington_sigma_by_comm:
            return float(armington_sigma_by_comm.get((region, commodity)) or 0.0)
        if commodity in armington_sigma_by_comm:
            return float(armington_sigma_by_comm.get(commodity) or 0.0)
        return None

    feed_base_by_rc: Dict[Tuple[str, str], float] = defaultdict(float)
    feed_coeff_map: Dict[Tuple[str, str, int], float] = dict(feed_crop_link_coeff or {})
    feed_override_map: Dict[Tuple[str, str, int], float] = dict(feed_crop_link_override or {})
    feed_credit_map: Dict[Tuple[str, str, int], float] = {}
    feed_livestock_set: set = set(feed_crop_link_livestock or [])
    if feed_crop_link_credit:
        for key, val in feed_crop_link_credit.items():
            try:
                r_key, j_key, t_key = key
                t_int = int(t_key)
                val_f = float(val or 0.0)
            except Exception:
                continue
            if not np.isfinite(val_f) or val_f <= 0.0:
                continue
            norm_key = (str(r_key).strip(), str(j_key).strip(), t_int)
            feed_credit_map[norm_key] = feed_credit_map.get(norm_key, 0.0) + val_f

    if feed_link_mode != 'off':
        if feed_crop_link_base:
            for (r_key, j_key, y_key), val in feed_crop_link_base.items():
                try:
                    y_int = int(y_key)
                except Exception:
                    continue
                if y_int != hist_end_year:
                    continue
                try:
                    feed_base_by_rc[(r_key, j_key)] += float(val or 0.0)
                except Exception:
                    continue
        if feed_link_mode == 'dynamic_constraints':
            if not feed_livestock_set:
                logger.warning("[LINEAR] feed link dynamic requested but livestock list empty")
                feed_link_mode = 'off'
            else:
                feed_livestock_set = {c for c in feed_livestock_set if c in comm_set}
                if not feed_livestock_set:
                    logger.warning("[LINEAR] feed link dynamic livestock not in commodities")
                    feed_link_mode = 'off'
        if feed_link_mode == 'iterative' and not feed_override_map:
            logger.warning("[LINEAR] feed link iterative requested but override map empty")

    if feed_link_mode == 'off':
        feed_coeff_map = {}
        feed_override_map = {}
        feed_livestock_set = set()
    else:
        logger.info(
            "[LINEAR] feed link: mode=%s base=%d coeff=%d override=%d livestock=%d",
            feed_link_mode,
            len(feed_base_by_rc),
            len(feed_coeff_map),
            len(feed_override_map),
            len(feed_livestock_set),
        )
        if feed_credit_map:
            logger.info(
                "[LINEAR] bioenergy feed credit loaded: keys=%d total=%.6g tDM",
                len(feed_credit_map),
                sum(feed_credit_map.values()),
            )

    def _lookup_feed_base(region: Any, commodity: Any) -> float:
        val = feed_base_by_rc.get((region, commodity))
        if val is None:
            val = feed_base_by_rc.get((str(region).strip(), str(commodity).strip()))
        return float(val or 0.0)

    def _lookup_feed_coef(region: Any, commodity: Any, year: Any) -> float:
        val = feed_coeff_map.get((region, commodity, year))
        if val is None:
            try:
                key = (str(region).strip(), str(commodity).strip(), int(year))
            except Exception:
                key = None
            if key is not None:
                val = feed_coeff_map.get(key)
        return float(val or 0.0)

    def _lookup_feed_override(region: Any, commodity: Any, year: Any) -> float:
        val = feed_override_map.get((region, commodity, year))
        if val is None:
            try:
                key = (str(region).strip(), str(commodity).strip(), int(year))
            except Exception:
                key = None
            if key is not None:
                val = feed_override_map.get(key)
        return float(val or 0.0)

    def _lookup_feed_credit(region: Any, commodity: Any, year: Any) -> float:
        val = feed_credit_map.get((region, commodity, year))
        if val is None:
            try:
                key = (str(region).strip(), str(commodity).strip(), int(year))
            except Exception:
                key = None
            if key is not None:
                val = feed_credit_map.get(key)
        return float(val or 0.0)

    
    # Strict consistency checks for nutrition-driven demand
    
    if method in {'nutrition', 'nutrition_band', 'nutrition_anchor'}:
        if not nutrition_rhs:
            msg = "[LINEAR] 需求采用营养驱动时必须提供 nutrition_rhs"
            if strict_nutrition:
                raise ValueError(msg)
            logger.warning(msg)
        if not nutrient_per_unit_by_comm:
            msg = "[LINEAR] 需求采用营养驱动时必须提供 nutrient_per_unit_by_comm"
            if strict_nutrition:
                raise ValueError(msg)
            logger.warning(msg)
        nutrition_rhs = nutrition_rhs or {}
        nutrient_per_unit_by_comm = nutrient_per_unit_by_comm or {}

        # Ensure every future (region, commodity, year) has nutrition demand
        if nutrition_demand_map:
            missing_keys: List[Tuple[str, str, int]] = []
            for (r, j, t) in idx.keys():
                if t <= hist_end_year or j in nonfood_commodities:
                    continue
                if (r, j, t) not in nutrition_demand_map:
                    missing_keys.append((r, j, t))
            if missing_keys:
                sample = missing_keys[:10]
                report_path = _write_nutrition_missing_report(
                    missing_keys=missing_keys,
                    nutrition_profile_xlsx=nutrition_profile_xlsx,
                    nutrition_profile_sheet=nutrition_profile_sheet,
                    dict_v3_path=dict_v3_path,
                    indicator=nutrition_indicator,
                    population_by_country_year=population_by_country_year,
                    hist_end_year=hist_end_year,
                    use_baseyear_for_future=nutrition_use_baseyear_for_future,
                    output_dir=output_dir,
                    nonfood_commodities=nonfood_commodities,
                )
                if report_path:
                    logger.warning(f"[LINEAR] 营养缺失清单已输出: {report_path}")
                msg = f"[LINEAR] 营养需求缺失 {len(missing_keys)} 条，示例: {sample}"
                if strict_nutrition:
                    raise ValueError(msg)
                logger.warning(msg)

        # Ensure RHS exists for every (region, year) appearing in nutrition demand
        if nutrition_demand_map and nutrition_rhs:
            demand_rt = {(r, t) for (r, j, t) in nutrition_demand_map.keys()
                         if t > hist_end_year and j not in nonfood_commodities}
            rhs_rt = {(r, t) for (r, t) in nutrition_rhs.keys() if t > hist_end_year}
            missing_rhs = sorted(demand_rt - rhs_rt)
            if missing_rhs:
                msg = f"[LINEAR] nutrition_rhs 缺失 {len(missing_rhs)} 个(地区,年份)，示例: {missing_rhs[:10]}"
                if strict_nutrition:
                    raise ValueError(msg)
                logger.warning(msg)

        # Compare nutrition totals derived from profile with RHS
        if nutrition_demand_map and nutrition_rhs and nutrient_per_unit_by_comm:
            totals_by_rt: Dict[Tuple[str, int], float] = {}
            for (r, j, t), demand_val in nutrition_demand_map.items():
                if t <= hist_end_year or j in nonfood_commodities:
                    continue
                kcal_per_ton = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                if kcal_per_ton <= 0:
                    continue
                totals_by_rt[(r, t)] = totals_by_rt.get((r, t), 0.0) + float(demand_val) * kcal_per_ton

            active_region_years = {
                (region, year) for (region, _commodity, year) in idx.keys()
            }
            mismatches = _nutrition_rhs_mismatches_for_active_step(
                nutrition_rhs,
                totals_by_rt,
                active_region_years=active_region_years,
                hist_end_year=hist_end_year,
            )
            if mismatches:
                log_fn = logger.error if strict_nutrition else logger.warning
                for r, t, rhs_val, total in mismatches:
                    nutri_gap = rhs_val - total
                    log_fn(
                        "[NUTRI_GAP] region=%s year=%s nutri_gap=%.6e RHS=%.6e -sum_kcal=%.6e",
                        r, t, nutri_gap, rhs_val, -total
                    )
                sample = mismatches[:10]
                msg = f"[LINEAR] 营养需求总量低于 nutrition_rhs {len(mismatches)} 条，示例: {sample}"
                if strict_nutrition:
                    raise ValueError(msg)
                logger.warning(msg)
    
    
    # Variables
    
    
    # Global or regional price variables
    Pc: Dict[Any, gp.Var] = {}
    Pw: Dict[Tuple[str, int], gp.Var] = {}
    if use_regional_price:
        for (r, j, t) in idx.keys():
            pmin_j, pmax_j = _price_bounds_for_comm(j)
            Pc[(r, j, t)] = m.addVar(lb=pmin_j, ub=pmax_j, name=f"Pc[{r},{j},{t}]")
        for j in commodities:
            for t in years:
                pmin_j, pmax_j = _price_bounds_for_comm(j)
                Pw[(j, t)] = m.addVar(lb=pmin_j, ub=pmax_j, name=f"Pw[{j},{t}]")
    else:
        for j in commodities:
            for t in years:
                pmin_j, pmax_j = _price_bounds_for_comm(j)
                Pc[(j, t)] = m.addVar(lb=pmin_j, ub=pmax_j, name=f"Pc[{j},{t}]")

    def _pc_key(region: Any, commodity: Any, year: Any) -> Tuple[Any, Any, Any]:
        return (region, commodity, year) if use_regional_price else (commodity, year)

    def _pc_var(region: Any, commodity: Any, year: Any) -> Optional[gp.Var]:
        return Pc.get(_pc_key(region, commodity, year))
    
    # Regional supply Qs[r,j,t] and demand Qd[r,j,t]
    Qs: Dict[Tuple[str, str, int], gp.Var] = {}
    Qd: Dict[Tuple[str, str, int], gp.Var] = {}
    Qs_bounds: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
    Qd_bounds: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
    supply_curtailment: Dict[Tuple[str, str, int], gp.Var] = {}
    qs_lb: Dict[Tuple[str, str, int], float] = {}
    qs_ub: Dict[Tuple[str, str, int], float] = {}
    qd_lb: Dict[Tuple[str, str, int], float] = {}
    qd_ub: Dict[Tuple[str, str, int], float] = {}
    for key in idx_keys:
        r, j, t = key
        if t <= hist_end_year:
            Q0_val = idx[key]['Q0']
            D0_val = idx[key]['D0']
            Q0_lb = max(1e-9, Q0_val * 0.1)
            D0_lb = max(1e-9, D0_val * 0.1)
            Q0_ub = max(Q0_val * 1.1, Q0_val * 10, 1e-3)
            D0_ub = max(D0_val * 1.1, D0_val * 10, 1e-3)
        else:
            pmin_j, _ = _price_bounds_for_comm(j)
            Q0_lb = 0.0 if zero_price_shutdown_enabled and pmin_j <= 0.0 else 1e-9
            D0_lb = 0.0
            Q0_ub = 1e12
            D0_ub = 1e12
        Q0_lb_scaled = float(Q0_lb) * inv_qty_scale
        Q0_ub_scaled = float(Q0_ub) * inv_qty_scale
        D0_lb_scaled = float(D0_lb) * inv_qty_scale
        D0_ub_scaled = float(D0_ub) * inv_qty_scale
        qs_lb[key] = float(Q0_lb_scaled)
        qs_ub[key] = float(Q0_ub_scaled)
        qd_lb[key] = float(D0_lb_scaled)
        qd_ub[key] = float(D0_ub_scaled)
        Qs_bounds[key] = (float(Q0_lb_scaled), float(Q0_ub_scaled))
        Qd_bounds[key] = (float(D0_lb_scaled), float(D0_ub_scaled))
    if idx_keys:
        qs_vars = m.addVars(idx_keys, lb=qs_lb, ub=qs_ub, name="Qs")
        qd_vars = m.addVars(idx_keys, lb=qd_lb, ub=qd_ub, name="Qd")
        Qs = {key: qs_vars[key] for key in idx_keys}
        Qd = {key: qd_vars[key] for key in idx_keys}

    if supply_curtailment_enabled and future_keys:
        curtail_vars = m.addVars(future_keys, lb=0.0, name="supply_curtail")
        supply_curtailment = {key: curtail_vars[key] for key in future_keys}
        logger.info(
            "[LINEAR] supply curtailment variables enabled: %d future supply keys",
            len(supply_curtailment),
        )
    if zero_price_shutdown_enabled:
        logger.info("[LINEAR] zero-price shutdown enabled: Pc lower bound 0 rebases future supply curves to Pc=0=>Qs=0")

    feed_demand_expr_by_key = {}
    feed_credit_scaled_by_key = {}
    product_feed_index = defaultdict(list)
    if feed_link_mode == 'dynamic_constraints':
        for key, coefficient in feed_coeff_map.items():
            if len(key) != 4:
                raise ValueError('B04 requires (region, feed crop, livestock product, year) coefficients')
            r, crop, livestock, t = key
            if (r, livestock, t) in Qs:
                product_feed_index[(r,crop,t)].append((float(coefficient), Qs[(r,livestock,t)]))

    if feed_link_mode == 'dynamic_constraints':
        required=set(product_feed_index)|set(feed_background_crop or {})
        missing=required-set(Qd)
        if missing:
            raise ValueError(f'Positive feed accounts have no market-demand variable: {sorted(missing)[:20]}')

    def _build_dynamic_feed_expr(region, commodity, year):
        if feed_link_mode != 'dynamic_constraints':
            return None
        key = (str(region), str(commodity), int(year))
        if key not in product_feed_index and key not in (feed_background_crop or {}):
            return None
        if key not in feed_demand_expr_by_key:
            terms = product_feed_index.get(key,[])
            feed_demand_expr_by_key[key] = gp.LinExpr([v for v,_ in terms], [v for _,v in terms])
            feed_demand_expr_by_key[key] += float((feed_background_crop or {}).get(key,0.)) * inv_qty_scale
        return feed_demand_expr_by_key[key]

    def _actual_feed_credit(region, commodity, year, gross):
        key = (str(region), str(commodity), int(year))
        if key in feed_credit_scaled_by_key:
            return feed_credit_scaled_by_key[key]
        potential = _lookup_feed_credit(*key) * inv_qty_scale
        if potential <= 0 or gross is None:
            return 0.0
        credit = bounded_feed_credit(m, gross, potential, name=f'{region},{commodity},{year}')
        feed_credit_scaled_by_key[key] = credit
        return credit

    world_price_constr: Dict[Tuple[str, int], gp.Constr] = {}
    if use_regional_price:
        for j in commodities:
            for t in years:
                region_keys = region_keys_by_comm_year.get((j, t), [])
                if not region_keys:
                    continue
                avg_expr = gp.quicksum(Pc[(r, j, t)] for r in region_keys) / max(1, len(region_keys))
                world_price_constr[(j, t)] = m.addConstr(
                    Pw[(j, t)] == avg_expr,
                    name=f"world_price[{j},{t}]",
                )
        logger.info("[LINEAR] world price constraints added: %d", len(world_price_constr))

    # Regional net imports M[r,j,t], only for future years
    net_import: Dict[Tuple[str, str, int], gp.Var] = {}
    trade_ub_default = 1e12 * inv_qty_scale
    trade_cap_exempt_pairs = trade_cap_exempt_pairs or set()
    if trade_cap_exempt_pairs:
        logger.info("[LINEAR] trade_cap_exempt_pairs=%d (skip cap for these)", len(trade_cap_exempt_pairs))
    if trade_cap_enabled and not use_region_trade_cap:
        if trade_cap_ratio_by_key:
            logger.info(
                "[LINEAR] dynamic trade_cap_ratio keys=%d; net_import bounds capped by base trade volume",
                len(trade_cap_ratio_by_key),
            )
        else:
            logger.info(
                "[LINEAR] trade_cap_ratio=%.6g; net_import bounds capped by base trade volume",
                trade_cap_ratio_val,
            )
    net_lb: Dict[Tuple[str, str, int], float] = {}
    net_ub: Dict[Tuple[str, str, int], float] = {}
    for key in future_keys:
        r, j, t = key
        ratio_for_pair = _lookup_trade_cap_ratio(r, j, trade_cap_ratio_val, trade_cap_ratio_by_key)
        if ratio_for_pair is not None and trade_cap_enabled and not use_region_trade_cap and (r, j) not in trade_cap_exempt_pairs:
            base_vol = float(trade_base_volume.get((r, j), 0.0) or 0.0)
            trade_ub = max(0.0, ratio_for_pair * base_vol) * inv_qty_scale
        else:
            trade_ub = trade_ub_default
        net_lb[key] = -trade_ub
        net_ub[key] = trade_ub
    if future_keys:
        net_import_vars = m.addVars(future_keys, lb=net_lb, ub=net_ub, name="net_import")
        net_import = {key: net_import_vars[key] for key in future_keys}

    nutrition_import_pos: Dict[Tuple[str, str, int], gp.Var] = {}
    nutrition_export_pos: Dict[Tuple[str, str, int], gp.Var] = {}
    nutrition_trade_abs_constr: Dict[Tuple[str, str, int], gp.Constr] = {}
    if nutrition_supply_driven and net_import:
        nutrition_import_vars = m.addVars(future_keys, lb=0.0, name="nutrition_import_pos")
        nutrition_export_vars = m.addVars(future_keys, lb=0.0, name="nutrition_export_pos")
        nutrition_import_pos = {key: nutrition_import_vars[key] for key in future_keys}
        nutrition_export_pos = {key: nutrition_export_vars[key] for key in future_keys}
        for key, ni_var in net_import.items():
            nutrition_trade_abs_constr[key] = m.addConstr(
                ni_var == nutrition_import_pos[key] - nutrition_export_pos[key],
                name=f"nutrition_trade_split[{key[0]},{key[1]},{key[2]}]",
            )
        logger.info(
            "[LINEAR] nutrition local-supply priority active: trade split variables=%d",
            len(nutrition_trade_abs_constr),
        )

    trade_cap_region_constr: Dict[Tuple[str, str, int, str], gp.Constr] = {}
    if trade_cap_enabled and use_region_trade_cap:
        created = 0
        growth_headroom_constraints = 0
        growth_headroom_total = 0.0
        feed_headroom_constraints = 0
        feed_headroom_terms = 0
        for (region_key, j, t), group_keys_all in future_keys_by_region_group_comm_year.items():
            ratio_for_pair = _lookup_trade_cap_ratio(region_key, j, trade_cap_ratio_val, trade_cap_ratio_by_key)
            if ratio_for_pair is None:
                continue
            base_vol = float(trade_cap_region_volume.get((region_key, j), 0.0) or 0.0)
            group_keys = [
                key for key in group_keys_all
                if (key[0], key[1]) not in trade_cap_exempt_pairs
            ]
            if not group_keys:
                continue
            cap_raw = max(0.0, ratio_for_pair * base_vol)
            if method in {'nutrition', 'nutrition_band', 'nutrition_anchor'}:
                future_demand = 0.0
                base_demand = 0.0
                for r_key, j_key, t_key in group_keys:
                    demand_val = nutrition_demand_map.get((r_key, j_key, t_key))
                    if demand_val is None:
                        demand_val = idx.get((r_key, j_key, t_key), {}).get('D0', 0.0)
                    try:
                        future_demand += max(0.0, float(demand_val or 0.0))
                    except Exception:
                        pass
                    try:
                        future_demand += max(
                            0.0,
                            float(nutrition_residual_demand_map.get((r_key, j_key, t_key), 0.0) or 0.0),
                        )
                    except Exception:
                        pass
                    base_row = idx.get((r_key, j_key, hist_end_year), {})
                    try:
                        base_demand += max(0.0, float(base_row.get('D0', 0.0) or 0.0))
                    except Exception:
                        pass
                growth_headroom = max(0.0, future_demand - base_demand)
                if growth_headroom > 0:
                    cap_raw += growth_headroom
                    growth_headroom_constraints += 1
                    growth_headroom_total += growth_headroom
            cap = cap_raw * inv_qty_scale
            expr = gp.quicksum(net_import[key] for key in group_keys)
            pos_lhs = expr
            if feed_link_mode == 'dynamic_constraints':
                feed_headroom_expr = gp.LinExpr(0.0)
                for r_key, j_key, t_key in group_keys:
                    feed_expr = _build_dynamic_feed_expr(r_key, j_key, t_key)
                    if feed_expr is None:
                        continue
                    feed_headroom_expr += feed_expr
                    feed_headroom_terms += 1
                if feed_headroom_expr.size() > 0 or abs(feed_headroom_expr.getConstant()) > 0:
                    # Dynamic feed is a hard demand term in Qd. The positive import cap must
                    # allow the crop/feed commodity to be sourced for that livestock-driven
                    # demand; otherwise tiny base trade in a feed crop can make BASE infeasible.
                    pos_lhs = expr - feed_headroom_expr
                    feed_headroom_constraints += 1
            trade_cap_region_constr[(region_key, j, t, 'pos')] = m.addConstr(
                pos_lhs <= cap,
                name=f"trade_cap_region_pos[{region_key},{j},{t}]",
            )
            trade_cap_region_constr[(region_key, j, t, 'neg')] = m.addConstr(
                expr >= -cap,
                name=f"trade_cap_region_neg[{region_key},{j},{t}]",
            )
            created += 2
        logger.info("[LINEAR] regional trade cap constraints: %d", created)
        if growth_headroom_constraints:
            logger.info(
                "[LINEAR] grouped trade cap demand-growth headroom: constraints=%d total=%.6g",
                growth_headroom_constraints,
                growth_headroom_total,
            )
        if feed_headroom_constraints:
            logger.info(
                "[LINEAR] grouped trade cap dynamic-feed headroom: constraints=%d terms=%d",
                feed_headroom_constraints,
                feed_headroom_terms,
            )

    # B01: physical gross origins, CES quantity and CES purchaser price.
    # These dictionaries remain empty for the retired trade-deviation penalty.
    armington_slack_pos: Dict[Tuple[str, str, int], gp.Var] = {}
    armington_slack_neg: Dict[Tuple[str, str, int], gp.Var] = {}
    armington_trade_constr: Dict[Tuple[str, str, int], gp.Constr] = {}
    ces_state = {}
    if use_regional_price:
        m.update()
        entry_region_map={}
        if dict_v3_path:
            entry_regions=pd.read_excel(dict_v3_path,sheet_name='region',
                usecols=['M49_Country_Code','Region_market_agg']).dropna().drop_duplicates()
            entry_region_map=dict(zip(entry_regions.M49_Country_Code.astype(str),entry_regions.Region_market_agg.astype(str)))
        ces_state = attach_ces_sources(
            m, external_balance=True, idx=idx, qs=Qs, qd=Qd, net=net_import, pd_vars=Pc, pm_vars=Pw,
            trade_net=trade_base_net_import, trade_volume=trade_base_volume,
            sigma_map=armington_sigma_by_comm, base_year=hist_end_year,
            quantity_scale=qty_scale, price_reference=price_ref_by_comm,
            relative_price=use_relative_price, bioenergy=bioenergy_crop_demand_map,
            output_dir=output_dir, entry_region_map=entry_region_map,
        )
        for key, source in ces_state['records'].items():
            if key in nutrition_import_pos:
                old_split=m.getConstrByName(f'nutrition_trade_split[{key[0]},{key[1]},{key[2]}]')
                if old_split is not None:
                    m.remove(old_split)
                m.addConstr(nutrition_import_pos[key] == source['imports'], name=f"ces_import_cost{key}")
                m.addConstr(nutrition_export_pos[key] == source['exports'], name=f"ces_export_cost{key}")

    def _consumer_price_var(region, commodity, year):
        source = ces_state.get('records', {}).get((region, commodity, year))
        return source['price'] if source is not None else _pc_var(region, commodity, year)

    # slack
    excess: Dict[Tuple[str, int], gp.Var] = {}
    shortage: Dict[Tuple[str, int], gp.Var] = {}
    for j in commodities:
        for t in years:
            excess[j, t] = m.addVar(lb=0, name=f"excess[{j},{t}]")
            shortage[j, t] = m.addVar(lb=0, name=f"shortage[{j},{t}]")
    
    
    # Emissions variables
    
    
    # Regional emissions E[r,j,t] and abatement costs C[r,j,t]
    Eij: Dict[Tuple[str, str, int], gp.Var] = {}
    Cij: Dict[Tuple[str, str, int], gp.Var] = {}
    
    # MACC abatement variables a[r,j,t,proc,seg]: abatement per process and segment
    abatement_vars: Dict[Tuple[str, str, int, str, int], gp.Var] = {}
    abatement_caps: Dict[Tuple[str, str, int, str, int], gp.Constr] = {}
    abatement_cost_vars: Dict[Tuple[str, str, int, str, int], gp.Var] = {}
    abatement_cost_caps: Dict[Tuple[str, str, int, str, int], gp.Constr] = {}
    abatement_req_vars: Dict[Tuple[str, str, int, str], gp.Var] = {}
    abatement_req_constr: Dict[Tuple[str, str, int, str], gp.Constr] = {}
    abatement_costs: Dict[Tuple[str, str, int, str, int], float] = {}  # Marginal costs
    abatement_database_keys: Dict[Tuple[Any, ...], str] = {}
    zero_cost_abatement_specs: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}
    no_opportunity_abatement_specs: Dict[Tuple[Any, ...], float] = {}
    strategy_abatement_vars: Dict[Tuple[str, int, str], gp.Var] = {}
    strategy_abatement_delta_vars: Dict[Tuple[str, int, str], gp.Var] = {}
    strategy_abatement_costs: Dict[Tuple[str, int, str], float] = {}
    zero_cost_strategy_abatement_specs: Dict[Tuple[str, int, str], Tuple[float, Any]] = {}
    
    for key in Qs.keys():
        r, j, t = key
        Eij[key] = m.addVar(lb=0.0, name=f"E[{r},{j},{t}]")
        Cij[key] = m.addVar(lb=0.0, name=f"C[{r},{j},{t}]")

    # Precompute land demand expressions for cropland/grassland
    cropland_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    grassland_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    luc_cropland_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    cropland_actual_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    grassland_actual_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    forest_actual_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    cropland_luc_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    grassland_luc_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    land_expr_missing_yield_skipped = 0
    land_expr_missing_yield_skipped_pairs = set()
    missing_yield_export_block_constr: Dict[Tuple[str, str, int], gp.Constr] = {}
    missing_yield_export_block_samples: List[Tuple[str, str, int, float, Any]] = []
    for (r, j, t), var in Qs.items():
        if j == "Fish, Seafood":
            continue
        node_data = idx.get((r, j, t)) if idx else None
        q0_for_land = 0.0
        yield_raw = None
        if node_data is not None:
            try:
                q0_for_land = float(node_data.get('Q0', 0.0) or 0.0)
            except Exception:
                q0_for_land = 0.0
            yield_raw = node_data.get('yield0')
        try:
            yield_check = float(yield_raw)
        except Exception:
            yield_check = 0.0
        if (not node_data.get('livestock_feed_account', False)) and (not np.isfinite(yield_check) or yield_check <= 0.0) and q0_for_land <= 1e-3:
            land_expr_missing_yield_skipped += 1
            if len(land_expr_missing_yield_skipped_pairs) < 20:
                land_expr_missing_yield_skipped_pairs.add((r, j))
            if t > hist_end_year:
                ni_var = net_import.get((r, j, t))
                missing_yield_export_block_constr[(r, j, t)] = m.addConstr(
                    var <= 0.0,
                    name=f"no_supply_missing_land_yield[{r},{j},{t}]",
                )
                if ni_var is not None:
                    m.addConstr(
                        ni_var >= 0.0,
                        name=f"no_export_missing_land_yield[{r},{j},{t}]",
                    )
                if len(missing_yield_export_block_samples) < 10:
                    missing_yield_export_block_samples.append((r, j, int(t), q0_for_land, yield_raw))
            continue
        yield_j = _cropland_yield0(
            node_data,
            region=r,
            commodity=j,
            year=t,
            context="cropland_expr",
        )
        # Qs is in quantity_model_units (= tonnes / qty_scale).  Land
        # expressions are in land_model_units (= ha / land_scale).
        coef_crop = qty_scale / yield_j * inv_land_scale
        cropland_expr = cropland_expr_by_region_year.get((r, t))
        if cropland_expr is None:
            cropland_expr = gp.LinExpr(0.0)
            cropland_expr_by_region_year[(r, t)] = cropland_expr
        cropland_expr += coef_crop * var
        if (not crop_land_commodities) or (j in crop_land_commodities):
            luc_cropland_expr = luc_cropland_expr_by_region_year.get((r, t))
            if luc_cropland_expr is None:
                luc_cropland_expr = gp.LinExpr(0.0)
                luc_cropland_expr_by_region_year[(r, t)] = luc_cropland_expr
            luc_cropland_expr += coef_crop * var

        if grassland_method == 'dynamic':
            coef_grass = 0.0
            if node_data is not None:
                coef_grass = float(node_data.get('grassland_coef', 0.0) or 0.0)
            if coef_grass > 0:
                grassland_expr = grassland_expr_by_region_year.get((r, t))
                if grassland_expr is None:
                    grassland_expr = gp.LinExpr(0.0)
                    grassland_expr_by_region_year[(r, t)] = grassland_expr
                # grassland_coef is ha/ton; Qs is scaled by 1/qty_scale and
                # land expressions are scaled by 1/land_scale.
                grassland_expr += coef_grass * qty_scale * inv_land_scale * var
    # B04 background herds have no output denominator. Their physical pasture
    # remains in the same finite land budget, separately from product Qs.
    for (r,t), area in (feed_background_pasture or {}).items():
        if (r,t) not in grassland_expr_by_region_year:
            grassland_expr_by_region_year[(r,t)] = gp.LinExpr(0.)
        grassland_expr_by_region_year[(r,t)] += float(area) * inv_land_scale
    for key,var in Qs.items():
        if key[2] > hist_end_year and idx[key].get('livestock_background_only',False):
            m.addConstr(var==0.,name=f'background_herd_no_product[{key[0]},{key[1]},{key[2]}]')
        if key[2] > hist_end_year and idx[key].get('livestock_feed_unidentified',False):
            m.addConstr(var==0.,name=f'missing_feed_reference_no_product[{key[0]},{key[1]},{key[2]}]')

    if land_expr_missing_yield_skipped:
        logger.info(
            "[LINEAR] missing-yield tiny-base land terms: rows=%d sample_pairs=%s",
            land_expr_missing_yield_skipped,
            sorted(land_expr_missing_yield_skipped_pairs)[:10],
        )
    if missing_yield_export_block_constr:
        logger.warning(
            "[LINEAR] blocked future supply/export nodes with missing land yield and tiny base Q0: "
            "constraints=%d sample=%s",
            len(missing_yield_export_block_constr),
            missing_yield_export_block_samples,
        )

    # Base demand (hist_end_year) for cropland/grassland deltas
    base_cropland_demand = defaultdict(float)
    base_luc_cropland_demand = defaultdict(float)
    base_grassland_demand = defaultdict(float)
    if idx:
        for (r, j, t), node_data in idx.items():
            if t != hist_end_year:
                continue
            if j == "Fish, Seafood":
                continue
            q0_val = float(node_data.get('Q0', 0.0) or 0.0)
            if q0_val <= 0:
                continue
            yield_j = _cropland_yield0(
                node_data,
                region=r,
                commodity=j,
                year=t,
                context="base_land_demand",
            )
            base_cropland_demand[r] += q0_val / yield_j
            if (not crop_land_commodities) or (j in crop_land_commodities):
                base_luc_cropland_demand[r] += q0_val / yield_j

            coef_grass = float(node_data.get('grassland_coef', 0.0) or 0.0)
            if coef_grass > 0:
                base_grassland_demand[r] += coef_grass * q0_val

    for (r,t),area in (feed_background_pasture or {}).items():
        if int(t)==hist_end_year:
            base_grassland_demand[r] += float(area)
    if grassland_method == 'static' and grass_area_by_region_year:
        for (r, t), val in grass_area_by_region_year.items():
            if int(t) == hist_end_year:
                base_grassland_demand[r] = float(val or 0.0)

    # Base land areas (hist_end_year) for land cover accounting
    base_cropland_area = defaultdict(float)
    base_grassland_area = defaultdict(float)
    base_forest_area = defaultdict(float)
    base_forest_area_scaled = defaultdict(float)
    if base_cropland_by_region:
        for r, val in base_cropland_by_region.items():
            base_cropland_area[r] = float(val or 0.0)
    if base_grassland_by_region:
        for r, val in base_grassland_by_region.items():
            base_grassland_area[r] = float(val or 0.0)
    if land_area_limits and (not base_cropland_by_region or not base_grassland_by_region):
        raise ValueError("LUH2基期cropland/grassland缺失：土地/森林约束必须使用LUH2基期面积，禁止回退Q0")
    if land_area_limits and base_cropland_by_region and base_grassland_by_region:
        missing_base = [
            r for r in regions
            if (r not in base_cropland_by_region) or (r not in base_grassland_by_region)
        ]
        if missing_base:
            raise ValueError(f"LUH2基期cropland/grassland覆盖不足，缺失区域: {sorted(missing_base)}")
    if forest_area_by_region_year:
        for r in regions:
            val = forest_area_by_region_year.get((r, hist_end_year))
            if val is None:
                continue
            base_forest_area[r] = float(val or 0.0)
    if base_forest_by_region:
        for r, val in base_forest_by_region.items():
            base_forest_area[str(r).strip()] = float(val or 0.0)
    base_forest_area_luc_raw = dict(base_forest_area)
    base_cropland_demand_raw = dict(base_cropland_demand)
    base_luc_cropland_demand_raw = dict(base_luc_cropland_demand)
    base_grassland_demand_raw = dict(base_grassland_demand)
    def _normalize_background_luc_flow_map(
        raw_map: Optional[Dict[Tuple[str, int], float]],
    ) -> Dict[Tuple[str, int], float]:
        out: Dict[Tuple[str, int], float] = {}
        if not raw_map:
            return out
        for key, val in raw_map.items():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            region_raw, year_raw = key[0], key[1]
            region_key = str(region_raw).strip()
            if not region_key:
                continue
            try:
                year_i = int(year_raw)
                area_val = float(val or 0.0)
            except Exception:
                continue
            if not np.isfinite(area_val) or area_val <= 0.0:
                continue
            out[(region_key, year_i)] = out.get((region_key, year_i), 0.0) + area_val
        return out

    background_forest_to_cropland = _normalize_background_luc_flow_map(
        background_forest_to_cropland_by_region_year
    )
    background_forest_to_grassland = _normalize_background_luc_flow_map(
        background_forest_to_grassland_by_region_year
    )
    background_cropland_to_forest = _normalize_background_luc_flow_map(
        background_cropland_to_forest_by_region_year
    )
    background_grassland_to_forest = _normalize_background_luc_flow_map(
        background_grassland_to_forest_by_region_year
    )
    if (
        background_forest_to_cropland
        or background_forest_to_grassland
        or background_cropland_to_forest
        or background_grassland_to_forest
    ):
        logger.info(
            "[LINEAR][LUC-BG] background gross flows active: "
            "forest_to_crop rows=%d area=%.6g ha; forest_to_grass rows=%d area=%.6g ha; "
            "crop_to_forest rows=%d area=%.6g ha; grass_to_forest rows=%d area=%.6g ha",
            len(background_forest_to_cropland),
            sum(background_forest_to_cropland.values()),
            len(background_forest_to_grassland),
            sum(background_forest_to_grassland.values()),
            len(background_cropland_to_forest),
            sum(background_cropland_to_forest.values()),
            len(background_grassland_to_forest),
            sum(background_grassland_to_forest.values()),
        )
    land_demand_mode = str(land_demand_calibration_mode or 'none').strip().lower()
    if land_demand_mode in {'stock_ratio', 'base-stock-ratio', 'base_stock', 'calibrated'}:
        land_demand_mode = 'base_stock_ratio'
    elif land_demand_mode in {'crop_stock_ratio', 'crop-base-stock-ratio', 'crop_base_stock', 'crop_calibrated'}:
        land_demand_mode = 'crop_base_stock_ratio'
    elif land_demand_mode in {'base_stock_capacity', 'stock_capacity', 'pasture_capacity'}:
        land_demand_mode = 'base_stock_capacity'
    elif land_demand_mode in {'off', 'false', '0', 'legacy'}:
        land_demand_mode = 'none'
    land_demand_crop_scale_by_region: Dict[str, float] = {}
    land_demand_grass_scale_by_region: Dict[str, float] = {}
    land_demand_calibration_rows: List[Tuple[str, str, float, float, float]] = []
    if land_demand_mode in {'base_stock_ratio', 'crop_base_stock_ratio', 'base_stock_capacity'}:
        eps_demand = 1e-9
        for r in regions:
            crop_raw = max(0.0, float(base_cropland_demand_raw.get(r, 0.0) or 0.0))
            grass_raw = max(0.0, float(base_grassland_demand_raw.get(r, 0.0) or 0.0))
            crop_stock = max(0.0, float(base_cropland_area.get(r, 0.0) or 0.0))
            grass_stock = max(0.0, float(base_grassland_area.get(r, 0.0) or 0.0))

            crop_scale = 1.0
            if crop_raw > eps_demand and crop_stock > 0.0:
                crop_scale = crop_stock / crop_raw
                base_cropland_demand[r] = crop_stock
                base_luc_cropland_demand[r] = max(
                    0.0,
                    float(base_luc_cropland_demand_raw.get(r, 0.0) or 0.0) * crop_scale,
                )
                if crop_scale < 0.5 or crop_scale > 2.0:
                    land_demand_calibration_rows.append((r, 'crop', crop_raw, crop_stock, crop_scale))
            land_demand_crop_scale_by_region[r] = crop_scale

            grass_scale = 1.0
            if land_demand_mode == 'base_stock_ratio' and grass_raw > eps_demand and grass_stock > 0.0:
                grass_scale = grass_stock / grass_raw
                base_grassland_demand[r] = grass_stock
                if grass_scale < 0.5 or grass_scale > 2.0:
                    land_demand_calibration_rows.append((r, 'grass', grass_raw, grass_stock, grass_scale))
            land_demand_grass_scale_by_region[r] = grass_scale

        crop_scaled = sum(1 for v in land_demand_crop_scale_by_region.values() if abs(v - 1.0) > 1e-9)
        grass_scaled = sum(1 for v in land_demand_grass_scale_by_region.values() if abs(v - 1.0) > 1e-9)
        logger.info(
            "[LINEAR] land demand calibration mode=%s: "
            "crop_scaled_regions=%d grass_scaled_regions=%d; "
            "calibrated land classes use 2020 LUH2 stock and future demand uses the same ratio",
            land_demand_mode,
            crop_scaled,
            grass_scaled,
        )
        if land_demand_calibration_rows:
            logger.info(
                "[LINEAR] land demand calibration large scale sample=%s",
                land_demand_calibration_rows[:12],
            )
    else:
        land_demand_mode = 'none'
        logger.info("[LINEAR] land demand calibration mode=none")
    try:
        forest_nonneg_ratio = float(forest_nonneg_ratio)
    except Exception:
        forest_nonneg_ratio = 1.0
    if not np.isfinite(forest_nonneg_ratio) or forest_nonneg_ratio <= 0:
        logger.warning("[LINEAR] forest_nonneg_ratio invalid; fallback to 1.0")
        forest_nonneg_ratio = 1.0
    def _sanitize_nonforest_expand_ratio(raw_val: Any, label: str) -> float:
        try:
            out = float(raw_val)
        except Exception:
            out = 1.0
        if not np.isfinite(out) or out <= 0:
            logger.warning("[LINEAR] %s invalid; fallback to 1.0", label)
            out = 1.0
        return out
    cropland_nonforest_expand_ratio = _sanitize_nonforest_expand_ratio(
        cropland_nonforest_expand_ratio,
        'cropland_nonforest_expand_ratio',
    )
    pasture_nonforest_expand_ratio = _sanitize_nonforest_expand_ratio(
        pasture_nonforest_expand_ratio,
        'pasture_nonforest_expand_ratio',
    )
    cropland_nonforest_extra_ratio = max(0.0, float(cropland_nonforest_expand_ratio) - 1.0)
    pasture_nonforest_extra_ratio = max(0.0, float(pasture_nonforest_expand_ratio) - 1.0)
    if not land_area_limits and (not base_cropland_by_region or not base_grassland_by_region):
        if idx:
            for (r, j, t), node_data in idx.items():
                if t != hist_end_year:
                    continue
                q0_val = float(node_data.get('Q0', 0.0) or 0.0)
                if q0_val <= 0:
                    continue
                yield_j = _cropland_yield0(
                    node_data,
                    region=r,
                    commodity=j,
                    year=t,
                    context="base_landcover_fallback",
                )
                base_cropland_area[r] += q0_val / yield_j

                coef_grass = float(node_data.get('grassland_coef', 0.0) or 0.0)
                if coef_grass > 0:
                    base_grassland_area[r] += coef_grass * q0_val

        if grassland_method == 'static' and grass_area_by_region_year:
            for (r, t), val in grass_area_by_region_year.items():
                if int(t) == hist_end_year:
                    base_grassland_area[r] = float(val or 0.0)

    base_forest_extra_area = defaultdict(float)
    forest_extra_ratio = max(0.0, float(forest_nonneg_ratio) - 1.0)
    if base_forest_area and forest_extra_ratio > 0.0:
        for r in list(base_forest_area.keys()):
            base_forest_extra_area[r] = float(base_forest_area[r]) * forest_extra_ratio

    def _scale_land_defaultdict(mapping: Mapping[Any, Any]) -> defaultdict:
        out = defaultdict(float)
        for key, val in (mapping or {}).items():
            try:
                out[key] = float(val or 0.0) * inv_land_scale
            except Exception:
                out[key] = 0.0
        return out

    def _scale_land_dict(mapping: Optional[Mapping[Any, Any]]) -> Dict[Any, float]:
        out: Dict[Any, float] = {}
        for key, val in (mapping or {}).items():
            try:
                out[key] = float(val or 0.0) * inv_land_scale
            except Exception:
                continue
        return out

    if abs(land_scale - 1.0) > 1e-12:
        base_cropland_demand = _scale_land_defaultdict(base_cropland_demand)
        base_luc_cropland_demand = _scale_land_defaultdict(base_luc_cropland_demand)
        base_grassland_demand = _scale_land_defaultdict(base_grassland_demand)
        base_cropland_area = _scale_land_defaultdict(base_cropland_area)
        base_grassland_area = _scale_land_defaultdict(base_grassland_area)
        base_forest_area = _scale_land_defaultdict(base_forest_area)
        base_forest_area_luc_raw = _scale_land_dict(base_forest_area_luc_raw)
        base_cropland_demand_raw = _scale_land_dict(base_cropland_demand_raw)
        base_luc_cropland_demand_raw = _scale_land_dict(base_luc_cropland_demand_raw)
        base_grassland_demand_raw = _scale_land_dict(base_grassland_demand_raw)
        base_forest_extra_area = _scale_land_defaultdict(base_forest_extra_area)
        background_forest_to_cropland = _scale_land_dict(background_forest_to_cropland)
        background_forest_to_grassland = _scale_land_dict(background_forest_to_grassland)
        background_cropland_to_forest = _scale_land_dict(background_cropland_to_forest)
        background_grassland_to_forest = _scale_land_dict(background_grassland_to_forest)
        land_area_limits = _scale_land_dict(land_area_limits)
        grass_area_by_region_year = _scale_land_dict(grass_area_by_region_year)
        forest_area_by_region_year = _scale_land_dict(forest_area_by_region_year)
        energy_crop_land_requirement_map = _scale_land_dict(energy_crop_land_requirement_map)
        logger.info(
            "[LINEAR] land-side inputs converted to solver units: land_scale=%.6g; "
            "reported outputs are converted back to ha",
            land_scale,
        )

    if bool(land_delta_anchor_to_available_stock):
        logger.info(
            "[LINEAR] skip base land stock reconcile because land_delta_anchor_to_available_stock=True; "
            "land delta anchors will be clipped to available base stock"
        )
    elif base_cropland_area or base_grassland_area or base_forest_extra_area:
        # First use only the forest_nonneg_ratio extra reserve for base land
        # calibration. Actual forest stock is kept for forest accounting unless
        # the extra reserve is insufficient.
        base_cropland_area, base_grassland_area, base_forest_extra_area = _reconcile_base_land_stock(
            base_cropland_area,
            base_grassland_area,
            base_forest_extra_area,
            base_cropland_demand,
            base_grassland_demand,
            logger=logger,
            context=(
                f"hist_end_year={hist_end_year}, extra_forest_reserve, "
                f"future_last_only={bool(future_last_only)}, "
                f"forest_nonneg_ratio={forest_nonneg_ratio:g}"
            ),
            log_unmet=False,
        )

    if (not bool(land_delta_anchor_to_available_stock)) and (base_cropland_area or base_grassland_area or base_forest_area):
        # Then fall back to real forest stock. This preserves the old reconcile
        # behavior while allowing the ratio-expanded reserve to absorb most
        # base stock mismatches before real forest is moved.
        base_cropland_area, base_grassland_area, base_forest_area = _reconcile_base_land_stock(
            base_cropland_area,
            base_grassland_area,
            base_forest_area,
            base_cropland_demand,
            base_grassland_demand,
            logger=logger,
            context=(
                f"hist_end_year={hist_end_year}, actual_forest_stock, "
                f"future_last_only={bool(future_last_only)}, "
                f"forest_nonneg_ratio={forest_nonneg_ratio:g}"
            ),
        )

    base_cropland_delta_anchor = dict(base_cropland_demand)
    base_grassland_delta_anchor = dict(base_grassland_demand)
    land_anchor_clip_cropland_by_region: Dict[str, float] = defaultdict(float)
    land_anchor_clip_grassland_by_region: Dict[str, float] = defaultdict(float)
    if bool(land_delta_anchor_to_available_stock):
        crop_clip_n = 0
        grass_clip_n = 0
        crop_clip_ha = 0.0
        grass_clip_ha = 0.0
        for r in set(base_cropland_delta_anchor) | set(base_cropland_area):
            demand_val = max(0.0, float(base_cropland_delta_anchor.get(r, 0.0) or 0.0))
            stock_val = max(0.0, float(base_cropland_area.get(r, 0.0) or 0.0))
            if demand_val > stock_val:
                clip_val = demand_val - stock_val
                base_cropland_delta_anchor[r] = stock_val
                land_anchor_clip_cropland_by_region[r] += clip_val
                crop_clip_n += 1
                crop_clip_ha += clip_val
        for r in set(base_grassland_delta_anchor) | set(base_grassland_area):
            demand_val = max(0.0, float(base_grassland_delta_anchor.get(r, 0.0) or 0.0))
            stock_val = max(0.0, float(base_grassland_area.get(r, 0.0) or 0.0))
            if demand_val > stock_val:
                clip_val = demand_val - stock_val
                base_grassland_delta_anchor[r] = stock_val
                land_anchor_clip_grassland_by_region[r] += clip_val
                grass_clip_n += 1
                grass_clip_ha += clip_val
        if crop_clip_n or grass_clip_n:
            logger.info(
                "[LINEAR] land delta anchor clipped to available base stock: "
                "crop_regions=%d crop_gap=%.6g ha; grass_regions=%d grass_gap=%.6g ha",
                crop_clip_n,
                crop_clip_ha,
                grass_clip_n,
                grass_clip_ha,
            )

    for r in set(base_forest_area.keys()) | set(base_forest_extra_area.keys()):
        base_forest_area_scaled[r] = (
            float(base_forest_area.get(r, 0.0) or 0.0)
            + float(base_forest_extra_area.get(r, 0.0) or 0.0)
        )
    # Keep the land-delta anchor fix separate from the forest conversion stock.
    # The anchor option only clips demand-derived base anchors. The
    # forest_nonneg_ratio option defines the forest stock available to
    # forest_to_crop/forest_to_grass conversion.
    forest_conversion_stock_uses_scaled = bool(base_forest_area_scaled)

    # Land conversion variables (future years only)
    grassland_to_cropland: Dict[Tuple[str, int], gp.Var] = {}
    grassland_to_forest: Dict[Tuple[str, int], gp.Var] = {}
    grassland_to_othernat: Dict[Tuple[str, int], gp.Var] = {}
    cropland_to_grassland: Dict[Tuple[str, int], gp.Var] = {}
    cropland_to_forest: Dict[Tuple[str, int], gp.Var] = {}
    cropland_to_othernat: Dict[Tuple[str, int], gp.Var] = {}
    forest_to_cropland: Dict[Tuple[str, int], gp.Var] = {}
    forest_to_grassland: Dict[Tuple[str, int], gp.Var] = {}
    nonforest_to_cropland: Dict[Tuple[str, int], gp.Var] = {}
    nonforest_to_grassland: Dict[Tuple[str, int], gp.Var] = {}
    nonforest_cropland_extra_cap_by_region: Dict[str, float] = {}
    nonforest_grassland_extra_cap_by_region: Dict[str, float] = {}
    if forest_conversion_stock_uses_scaled and base_forest_area_scaled:
        logger.info(
            "[LINEAR] forest_nonneg_ratio applies to forest conversion stock: "
            "base_forest=%.6g ha scaled_forest=%.6g ha extra=%.6g ha",
            sum(float(v or 0.0) for v in base_forest_area.values()),
            sum(float(v or 0.0) for v in base_forest_area_scaled.values()),
            sum(float(base_forest_area_scaled.get(r, 0.0) or 0.0) - float(base_forest_area.get(r, 0.0) or 0.0)
                for r in set(base_forest_area_scaled) | set(base_forest_area)),
        )

    for r in regions:
        base_crop_ub = max(0.0, float(base_cropland_area.get(r, 0.0) or 0.0))
        base_grass_ub = max(0.0, float(base_grassland_area.get(r, 0.0) or 0.0))
        if forest_conversion_stock_uses_scaled:
            base_forest_ub = max(0.0, float(base_forest_area_scaled.get(r, base_forest_area.get(r, 0.0)) or 0.0))
        else:
            base_forest_ub = max(0.0, float(base_forest_area.get(r, 0.0) or 0.0))
        crop_extra_ub = max(0.0, base_crop_ub * cropland_nonforest_extra_ratio)
        grass_extra_ub = max(0.0, base_grass_ub * pasture_nonforest_extra_ratio)
        nonforest_cropland_extra_cap_by_region[r] = crop_extra_ub
        nonforest_grassland_extra_cap_by_region[r] = grass_extra_ub
        for t in years:
            if t <= hist_end_year:
                continue
            grassland_to_cropland[(r, t)] = m.addVar(lb=0.0, ub=base_grass_ub, name=f"grass_to_crop[{r},{t}]")
            grassland_to_forest[(r, t)] = m.addVar(lb=0.0, ub=base_grass_ub, name=f"grass_to_forest[{r},{t}]")
            grassland_to_othernat[(r, t)] = m.addVar(lb=0.0, ub=base_grass_ub, name=f"grass_to_othernat[{r},{t}]")
            cropland_to_grassland[(r, t)] = m.addVar(lb=0.0, ub=base_crop_ub, name=f"crop_to_grass[{r},{t}]")
            cropland_to_forest[(r, t)] = m.addVar(lb=0.0, ub=base_crop_ub, name=f"crop_to_forest[{r},{t}]")
            cropland_to_othernat[(r, t)] = m.addVar(lb=0.0, ub=base_crop_ub, name=f"crop_to_othernat[{r},{t}]")
            forest_to_cropland[(r, t)] = m.addVar(lb=0.0, ub=base_forest_ub, name=f"forest_to_crop[{r},{t}]")
            forest_to_grassland[(r, t)] = m.addVar(lb=0.0, ub=base_forest_ub, name=f"forest_to_grass[{r},{t}]")
            if crop_extra_ub > 0.0:
                nonforest_to_cropland[(r, t)] = m.addVar(
                    lb=0.0,
                    ub=crop_extra_ub,
                    name=f"other_to_crop[{r},{t}]",
                )
            if grass_extra_ub > 0.0:
                nonforest_to_grassland[(r, t)] = m.addVar(
                    lb=0.0,
                    ub=grass_extra_ub,
                    name=f"other_to_grass[{r},{t}]",
                )
    if cropland_nonforest_extra_ratio > 0.0 or pasture_nonforest_extra_ratio > 0.0:
        logger.info(
            "[LINEAR] nonforest land expansion: cropland_ratio=%.6g pasture_ratio=%.6g "
            "crop_extra_cap=%.6g ha pasture_extra_cap=%.6g ha",
            cropland_nonforest_expand_ratio,
            pasture_nonforest_expand_ratio,
            sum(nonforest_cropland_extra_cap_by_region.values()),
            sum(nonforest_grassland_extra_cap_by_region.values()),
        )
    
    
    # Constraints
    
    
    # Constraint reference dictionaries for MC updates
    constr_supply: Dict[Tuple[str, str, int], gp.Constr] = {}
    constr_demand: Dict[Tuple[str, str, int], gp.Constr] = {}
    constr_Edef: Dict[Tuple[str, str, int], gp.Constr] = {}
    
    # Calibration parameter dictionaries for MC updates
    alpha_s_cache: Dict[Tuple[str, str, int], float] = {}
    alpha_d_cache: Dict[Tuple[str, str, int], float] = {}
    eps_s_cache: Dict[Tuple[str, str, int], float] = {}
    eps_d_cache: Dict[Tuple[str, str, int], float] = {}
    eps_pop_cache: Dict[Tuple[str, str, int], float] = {}
    eps_inc_cache: Dict[Tuple[str, str, int], float] = {}
    eta_y_cache: Dict[Tuple[str, str, int], float] = {}
    eta_temp_cache: Dict[Tuple[str, str, int], float] = {}
    Q0_cache: Dict[Tuple[str, str, int], float] = {}
    D0_cache: Dict[Tuple[str, str, int], float] = {}
    P0_cache: Dict[Tuple[str, str, int], float] = {}
    Ymult0_cache: Dict[Tuple[str, str, int], float] = {}
    Tmult0_cache: Dict[Tuple[str, str, int], float] = {}
    pop_base_cache: Dict[Tuple[str, str, int], float] = {}
    inc_base_cache: Dict[Tuple[str, str, int], float] = {}
    proc_cap_basecoeff: Dict[Tuple[str, str, int, str, int], Optional[float]] = {}
    pc_range_diag: Dict[Tuple[str, int], Dict[str, Any]] = {}
    pc_nonneg_lb: Dict[Tuple[Any, ...], float] = {}
    pc_nonneg_ub: Dict[Tuple[Any, ...], float] = {}
    pc_nonneg_lb_src: Dict[Tuple[Any, ...], Tuple[str, str, int, float]] = {}
    pc_nonneg_ub_src: Dict[Tuple[Any, ...], Tuple[str, str, int, float]] = {}
    pc_nonneg_impossible: List[Tuple[Any, ...]] = []
    
    
    # 1. Supply equation: linearized log-log elasticity
    
    # Full log-log form: addition in log space equals multiplication in original space.
    # ln(Qs) = α_s + ε_s·ln(Pnet) + η_y·ln(Ymult) + η_temp·ln(Tmult) + Σ(ε_sj·ln(Pj))
    
    # Equivalent expression in original space:
    # Qs = A · Pnet^ε_s · Ymult^η_y · Tmult^η_temp · ∏(Pj^ε_sj)
    
    # Linearize using a first-order Taylor expansion around the base-period point:
    # ln(Qs) ≈ ln(Q0) + ε_s·[(P-P0)/P0] + η_y·[(Y-1)/1] + η_temp·[(T-1)/1] + Σ(ε_sj·[(Pj-P0j)/P0j])
    
    # Rearrange to obtain a linear approximation of the multiplicative form:
    # Qs ≈ Q0 · exp(ε_s·(P-P0)/P0 + η_y·(Ymult-1) + η_temp·(Tmult-1) + Σ(ε_sj·(Pj-P0j)/P0j))
    
    # Further linearize exp(x) as 1 + x for small x:
    # Qs ≈ Q0 · [1 + ε_s·(P-P0)/P0 + η_y·(Ymult-1) + η_temp·(Tmult-1) + Σ(ε_sj·(Pj-P0j)/P0j)]
    
    # All effects here are additive, not multiplicative!
    
    
    # Count historical/future constraints.
    n_hist_skipped = 0
    n_future_supply = 0
    n_nutrition_supply_driven = 0
    n_tiny_supply_floor = 0
    n_zero_price_shutdown_rebased = 0
    n_zero_price_shutdown_cross_dropped = 0
    TINY_Q0_THRESHOLD = 1e-3  # kt/year, original units
    tiny_q0_scaled = TINY_Q0_THRESHOLD * inv_qty_scale
    
    for key in Qs.keys():
        r, j, t = key
        data = idx[key]
        Q0_actual = max(1e-6, data['Q0'])
        P0 = max(1e-6, data['P0'])
        Q0 = Q0_actual * inv_qty_scale
        
        # Historical years: fix Qs = Q0 without an elasticity equation.
        if t <= hist_end_year:
            # Fix supply to historical observations.
            # Still cache parameters for other uses.
            alpha_s_cache[key] = Q0
            eps_s_cache[key] = 0.0
            eta_y_cache[key] = 0.0
            eta_temp_cache[key] = 0.0
            Q0_cache[key] = Q0
            P0_cache[key] = P0
            Ymult0_cache[key] = 1.0
            Tmult0_cache[key] = 1.0
            constr_supply[key] = m.addConstr(
                Qs[key] == Q0,
                name=f"supply_hist[{r},{j},{t}]",
            )
            n_hist_skipped += 1
            continue
        
        # Future years: apply the elasticity equation.
        
        # Future years use Q0 from hist_end_year as the baseline.
        # This keeps the supply equation consistent with fixed historical values and growth constraints.
        base_key = (r, j, hist_end_year)
        if base_key in idx:
            Q0_base_actual = max(1e-6, idx[base_key].get('Q0', 1e-6))
            P0_base = max(1e-6, idx[base_key].get('P0', 1.0))
        else:
            # Fall back to current-year data if historical base-period data are unavailable.
            Q0_base_actual = max(1e-6, data['Q0'])
            P0_base = max(1e-6, data['P0'])
        
        # Use base-period Q0 and P0.
        Q0 = Q0_base_actual * inv_qty_scale
        P0 = P0_base

        if Q0 <= tiny_q0_scaled:
            # Allow tiny base supply to expand; use a small floor to avoid degenerate coefficients.
            Q0 = tiny_q0_scaled
            n_tiny_supply_floor += 1

        n_future_supply += 1

        if nutrition_supply_driven:
            # In exact nutrition mode Qd already represents total demand
            # (food profile + feed + residual + losses). Future Qs is therefore
            # chosen by balance, land capacity, trade caps, and the cost
            # objective instead of a price-elastic supply equation.
            alpha_s_cache[key] = 0.0
            eps_s_cache[key] = 0.0
            eta_y_cache[key] = 0.0
            eta_temp_cache[key] = 0.0
            Q0_cache[key] = Q0
            P0_cache[key] = P0
            Ymult0_cache[key] = float(data.get('Ymult', 1.0) or 1.0)
            Tmult0_cache[key] = float(data.get('Tmult', 1.0) or 1.0)
            n_nutrition_supply_driven += 1
            continue
        
        # Supply elasticities
        eps_s = data.get('eps_supply', 0.0) or 0.0
        eta_y = data.get('eps_supply_yield', 0.0) or 0.0      # Yield elasticity
        eta_temp = data.get('eps_supply_temp', 0.0) or 0.0    # Temperature elasticity
        
        # Yield and temperature multipliers are exogenous scenario factors.
        Ymult = data.get('Ymult', 1.0) or 1.0
        Tmult = data.get('Tmult', 1.0) or 1.0
        
        # Cross-price elasticities
        epsS_row_raw = {k: v for k, v in (data.get('epsS_row', {}) or {}).items() if k in comm_set}
        epsS_row_scaled = _scale_cross_terms(epsS_row_raw, cross_terms_scale)
        epsS_row = _limit_cross_terms(epsS_row_scaled, cross_terms_top_n)
        pmin_j, pmax_j = _price_bounds_for_comm(j)
        zero_price_shutdown_mode = bool(zero_price_shutdown_enabled and pmin_j <= 0.0 and pmax_j > 0.0)
        if zero_price_shutdown_mode and epsS_row:
            n_zero_price_shutdown_cross_dropped += 1
            epsS_row = {}
        
        # Calculate the intercept from non-price effects.
        # These are known before solving and adjust Q0.
        yield_adj = eta_y * (Ymult - 1.0)      # Yield effect
        temp_adj = eta_temp * (Tmult - 1.0)    # Temperature effect
        
        # Adjusted Q0 baseline, including yield and temperature effects
        # Qs = Q0 * [1 + yield_adj + temp_adj + price_effects]
        Q0_const = Q0 * (1.0 + yield_adj + temp_adj)
        
        # Own-price effect: epsilon_s * Q0 * (Pc - P0) / P0 = epsilon_s * Q0 / P0 * Pc - epsilon_s * Q0.
        # Rearranging: Qs = Q0_const + epsilon_s * Q0 * (Pc - P0) / P0 + sum(epsilon_sj * Q0 * (Pj - P0j) / P0j).
        # Q0_const - ε_s * Q0 + ε_s * Q0 / P0 * Pc + Σ(...)
        # [Q0_const - ε_s * Q0 - Σ(ε_sj * Q0)] + ε_s * Q0 / P0 * Pc + Σ(ε_sj * Q0 / P0j * Pj)
        
        # Initially subtract only the own-price baseline offset. Missing variables or numerical thresholds may exclude cross terms;
        # subtract a cross term's baseline offset below only when that term enters the equation.
        sum_cross_eps = sum(float(v) for v in epsS_row.values())
        a_s = Q0 * (1.0 + yield_adj + temp_adj - eps_s)
        
        # Own-price coefficient: b_s = Q0 * epsilon_s / P0 (absolute-price form).
        b_s_abs = Q0 * eps_s / P0
        b_s = b_s_abs
        if use_relative_price:
            price_ref_self = float(price_ref_by_comm.get(j, 1.0) or 1.0)
            if not np.isfinite(price_ref_self) or price_ref_self <= 0:
                price_ref_self = 1.0
            b_s = b_s_abs * price_ref_self
        
        # Filter tiny coefficients to avoid numerical instability.
        COEFF_THRESHOLD = 1e-6 * inv_qty_scale
        if abs(b_s) < COEFF_THRESHOLD:
            b_s = 0.0
            b_s_abs = 0.0
        
        # Phase 2: unit tax tax_unit_adder
        # Supply responds to net price Pnet = Pc - tau, so:
        # Qs = a_s + b_s * (Pc - tau) = (a_s - b_s * tau) + b_s * Pc
        # Adjust intercept a_s for the tax effect.
        tau = 0.0
        if tax_unit_adder:
            tau = float(tax_unit_adder.get(key, 0.0) or 0.0)
            a_s = a_s - b_s_abs * tau
        
        price_wedge_self = _price_wedge(r, j, t)

        # Construct cross-price terms: sum(epsilon_sj * Q0 / P0j * Pc_j).
        cross_terms_s = gp.LinExpr(0.0)
        cross_min = 0.0
        cross_max = 0.0
        cross_wedge_shift = 0.0
        cross_const_offset_s = 0.0
        for other_comm, cross_eps in epsS_row.items():
            if other_comm == j:  # Skip the commodity itself.
                continue
            pc_other = _pc_var(r, other_comm, t)
            if pc_other is not None:
                other_base_key = (r, other_comm, hist_end_year)
                P0_other = idx.get(other_base_key, {}).get('P0')
                if P0_other is None:
                    other_key = (r, other_comm, t)
                    P0_other = idx.get(other_key, {}).get('P0', P0) or P0
                P0_other = max(1e-6, P0_other)
                b_cross_abs = Q0 * float(cross_eps) / P0_other
                b_cross = b_cross_abs
                pc_other_base_for_offset = P0_other
                if use_relative_price:
                    price_ref_other = float(price_ref_by_comm.get(other_comm, 1.0) or 1.0)
                    if not np.isfinite(price_ref_other) or price_ref_other <= 0:
                        price_ref_other = 1.0
                    b_cross = b_cross_abs * price_ref_other
                    pc_other_base_for_offset = P0_other / price_ref_other
                # Filter tiny cross-elasticity coefficients.
                b_cross = _sanitize_cross_coef(
                    b_cross,
                    base_qty=Q0,
                    cross_eps=float(cross_eps),
                    use_relative_price=use_relative_price,
                    ratio_tol=CROSS_COEF_RATIO_TOL,
                    logger=logger,
                    log_state=cross_fix_supply,
                    region=r,
                    commodity=j,
                    year=t,
                    other_comm=other_comm,
                    tag='supply',
                )
                if abs(b_cross) < COEFF_THRESHOLD:
                    continue
                cross_terms_s += b_cross * pc_other
                cross_const_offset_s += b_cross * pc_other_base_for_offset
                other_wedge = _price_wedge(r, other_comm, t)
                if other_wedge:
                    cross_wedge_shift += b_cross * other_wedge
                pmin_o, pmax_o = _price_bounds_for_comm(other_comm)
                if b_cross >= 0:
                    cross_min += b_cross * pmin_o
                    cross_max += b_cross * pmax_o
                else:
                    cross_min += b_cross * pmax_o
                    cross_max += b_cross * pmin_o

        if j == 'Roundwood' and str(r).lstrip("'") == "203" and int(t) == 2080:
            logger = logging.getLogger(__name__)
            logger.info(
                f"[DEBUG-RW] supply params r={r} j={j} t={t} "
                f"Q0={Q0:.6g} P0={P0:.6g} eps_s={eps_s:.6g} "
                f"yield_adj={yield_adj:.6g} temp_adj={temp_adj:.6g} "
                f"sum_cross_eps={sum_cross_eps:.6g} tau={tau:.6g} "
                f"a_s={a_s:.6g} b_s={b_s:.6g}"
            )
        
        a_s = a_s - cross_const_offset_s
        a_s_eff = a_s + b_s * price_wedge_self + cross_wedge_shift
        a_s_eff = max(0.0, a_s_eff)
        pc_self = _pc_var(r, j, t)
        if pc_self is None:
            raise ValueError(f"[LINEAR] missing Pc for supply key {r},{j},{t}")
        curtail = supply_curtailment.get(key)
        if zero_price_shutdown_mode and a_s_eff > 0.0:
            pc_ref_shutdown = 1.0 if use_relative_price else max(P0, 1e-6)
            b_s += a_s_eff / pc_ref_shutdown
            a_s_eff = 0.0
            n_zero_price_shutdown_rebased += 1
        if curtail is None:
            cs = m.addConstr(Qs[key] == a_s_eff + b_s * pc_self + cross_terms_s, name=f"supply[{r},{j},{t}]")
        else:
            cs = m.addConstr(
                Qs[key] + curtail == a_s_eff + b_s * pc_self + cross_terms_s,
                name=f"supply[{r},{j},{t}]",
            )
        constr_supply[key] = cs
        _update_pc_range_diag(
            pc_range_diag,
            commodity=j,
            year=t,
            region=r,
            constr_type='supply',
            constr_name=f"supply[{r},{j},{t}]",
            a=a_s_eff,
            b=b_s,
            cross_min=cross_min,
            cross_max=cross_max,
            q_lb=float(Qs_bounds.get(key, (Qmin, Qmax))[0]),
            q_ub=float(Qs_bounds.get(key, (Qmin, Qmax))[1]),
            pmin=pmin_j,
            pmax=pmax_j,
        )
        if str(r).lstrip("'") == "356" and int(t) == 2080:
            eps_s = data.get('eps_supply', 0.0) or 0.0
            nz_s = {k: v for k, v in epsS_row.items() if abs(v) > 1e-12}
            logger.info(
                "[SUPPLY-DIAG] region=%s year=%s item=%s Q0=%.6g P0=%.6g eps_s=%.6g "
                "a_s=%.6g b_s=%.6g cross_terms=%d",
                r, t, j, Q0, P0, eps_s, a_s_eff, b_s, len(nz_s)
            )
        
        # Cache calibration parameters.
        alpha_s_cache[key] = a_s_eff
        eps_s_cache[key] = eps_s
        eta_y_cache[key] = eta_y
        eta_temp_cache[key] = eta_temp
        Q0_cache[key] = Q0
        P0_cache[key] = P0
        Ymult0_cache[key] = Ymult
        Tmult0_cache[key] = Tmult
    
    
    # 2. Demand equation: linearized log-log elasticity plus population/income effects
    
    # Full log-log form: addition in log space equals multiplication in original space.
    # ln(Qd) = α_d + ε_d·ln(Pc) + ε_pop·ln(Pop/Pop0) + ε_inc·ln(Inc/Inc0) + Σ(ε_dj·ln(Pj))
    
    # Equivalent expression in original space:
    # Qd = A · Pc^ε_d · (Pop/Pop0)^ε_pop · (Inc/Inc0)^ε_inc · ∏(Pj^ε_dj)
    
    # Linearize using a first-order Taylor expansion:
    # ln(Qd) ≈ ln(D0) + ε_d·(Pc-P0)/P0 + ε_pop·ln(Pop/Pop0) + ε_inc·ln(Inc/Inc0) + Σ(ε_dj·(Pj-P0j)/P0j)
    
    # Population and income log terms are constants known before solving; they need no linearization.
    # Price terms do require linearization.
    
    # Rearranging gives:
    # Qd ≈ D0 · exp(ε_pop·ln(Pop/Pop0) + ε_inc·ln(Inc/Inc0)) · [1 + ε_d·(Pc-P0)/P0 + Σ(ε_dj·(Pj-P0j)/P0j)]
    # D0 · (Pop/Pop0)^ε_pop · (Inc/Inc0)^ε_inc · [1 + price_effects]
    
    # Population/income effects are multiplicative; apply powers directly because these inputs are constants.
    # Price effects are linearized additively.
    
    
    n_hist_demand_skipped = 0
    n_future_demand = 0
    zero_nutrition_demand_keys: set = set()
    for key in Qd.keys():
        r, j, t = key
        data = idx[key]
        D0_actual = max(1e-6, data['D0'])
        P0 = max(1e-6, data['P0'])
        D0 = D0_actual * inv_qty_scale
        
        # Historical years: fix Qd = D0 without an elasticity equation.
        if t <= hist_end_year:
            # Fix demand to historical observations.
            # Cache parameters.
            alpha_d_cache[key] = D0
            eps_d_cache[key] = 0.0
            eps_pop_cache[key] = 0.0
            eps_inc_cache[key] = 0.0
            D0_cache[key] = D0
            pop_base_cache[key] = 1.0
            inc_base_cache[key] = 1.0
            constr_demand[key] = m.addConstr(
                Qd[key] == D0,
                name=f"demand_hist[{r},{j},{t}]",
            )
            n_hist_demand_skipped += 1
            continue
        
        # Future years: apply elasticity equations or nutrition-driven demand.
        profile_demand = None
        if method in {'nutrition', 'nutrition_band', 'nutrition_anchor'} and j not in nonfood_commodities:
            nut_key = (r, j, t)
            profile_demand = nutrition_demand_map.get(nut_key)
            if profile_demand is None and data.get('feed_only_entry',False):
                profile_demand = 0.0  # Explicit feed-only market entry, no imputed human food.
            if profile_demand is None and strict_nutrition:
                raise ValueError(f"[LINEAR] 营养需求缺失: {nut_key}")
        residual_scaled_for_profile = 0.0
        if method in {'nutrition_band', 'nutrition_anchor'}:
            residual_scaled_for_profile = (
                float(nutrition_residual_demand_map.get(key, 0.0) or 0.0) * inv_qty_scale
            )
        if method == 'nutrition':
            residual_scaled = float(nutrition_residual_demand_map.get(key, 0.0) or 0.0) * inv_qty_scale
            feed_expr_exact = _build_dynamic_feed_expr(r, j, t)
            exact_demand_needed = (
                j not in nonfood_commodities
                or residual_scaled > 0.0
                or feed_expr_exact is not None
            )
            if exact_demand_needed and j not in nonfood_commodities and profile_demand is None:
                raise ValueError(f"[LINEAR] 营养需求缺失: {(r, j, t)}")
            if exact_demand_needed:
                profile_scaled = 0.0
                if j not in nonfood_commodities and profile_demand is not None:
                    profile_scaled = float(profile_demand) * inv_qty_scale
                feed_credit_scaled = _actual_feed_credit(r,j,t,feed_expr_exact)
                demand_const_scaled = profile_scaled + residual_scaled - feed_credit_scaled
                if feed_expr_exact is not None:
                    feed_demand_expr_by_key[key] = feed_expr_exact
                    m.addConstr(
                        Qd[key] == demand_const_scaled + feed_expr_exact,
                        name=f"demand_nutrition[{r},{j},{t}]",
                    )
                else:
                    m.addConstr(Qd[key] == demand_const_scaled, name=f"demand_nutrition[{r},{j},{t}]")
                    if bool(zero_demand_production_shutdown) and abs(float(demand_const_scaled)) <= 1e-12:
                        zero_nutrition_demand_keys.add(key)
                alpha_d_cache[key] = profile_scaled + residual_scaled
                eps_d_cache[key] = 0.0
                eps_pop_cache[key] = 0.0
                eps_inc_cache[key] = 0.0
                D0_cache[key] = demand_const_scaled
                pop_base_cache[key] = 1.0
                inc_base_cache[key] = 1.0
                n_future_demand += 1
                continue
        n_future_demand += 1
        
        # Future years use D0 and P0 from hist_end_year as the baseline.
        base_key = (r, j, hist_end_year)
        if base_key in idx:
            D0_base_actual = max(1e-6, idx[base_key].get('D0', 1e-6))
            P0_base = max(1e-6, idx[base_key].get('P0', 1.0))
        else:
            # Fall back to current-year data if historical base-period data are unavailable.
            D0_base_actual = max(1e-6, data['D0'])
            P0_base = max(1e-6, data['P0'])
        
        nutrition_anchor_uses_profile = False
        # nutrition_band constrains Qd around the nutrition profile, so its
        # elastic demand curve must be anchored to the same profile baseline.
        if method in {'nutrition_anchor', 'nutrition_band'} and j not in nonfood_commodities and profile_demand is not None:
            try:
                profile_val = float(profile_demand)
            except Exception:
                profile_val = None
            if profile_val is not None and np.isfinite(profile_val) and profile_val >= 0:
                D0_base_actual = max(1e-6, profile_val)
                nutrition_anchor_uses_profile = True

        D0 = D0_base_actual * inv_qty_scale
        P0 = P0_base

        if method == 'nutrition_band' and j not in nonfood_commodities and profile_demand is not None:
            try:
                profile_val = float(profile_demand)
            except Exception:
                profile_val = None
            if profile_val is not None and np.isfinite(profile_val):
                lower = max(0.0, profile_val * (1.0 - band_eps))
                upper = max(lower, profile_val * (1.0 + band_eps))
                residual_scaled_for_band = residual_scaled_for_profile
                band_abs_tol_scaled = max(
                    1e-4,
                    1e-9 * (abs(profile_val) + abs(residual_scaled_for_band / max(inv_qty_scale, 1e-300))),
                ) * inv_qty_scale
                lower_expr = max(0.0, lower * inv_qty_scale + residual_scaled_for_band - band_abs_tol_scaled)
                upper_expr = upper * inv_qty_scale + residual_scaled_for_band + band_abs_tol_scaled
                band_feed_expr = _build_dynamic_feed_expr(r, j, t)
                if band_feed_expr is not None:
                    feed_credit_scaled = _actual_feed_credit(r,j,t,band_feed_expr)
                    lower_expr = lower_expr - feed_credit_scaled
                    upper_expr = upper_expr - feed_credit_scaled
                    feed_demand_expr_by_key[key] = band_feed_expr
                    lower_expr = lower_expr + band_feed_expr
                    upper_expr = upper_expr + band_feed_expr
                m.addConstr(Qd[key] >= lower_expr, name=f"demand_nutrition_lb[{r},{j},{t}]")
                m.addConstr(Qd[key] <= upper_expr, name=f"demand_nutrition_ub[{r},{j},{t}]")
                if bool(zero_demand_production_shutdown) and upper <= 1e-12:
                    residual_scaled_for_shutdown = residual_scaled_for_profile
                    if abs(residual_scaled_for_shutdown) <= 1e-12 and band_feed_expr is None:
                        zero_nutrition_demand_keys.add(key)
        
        # Demand elasticities
        eps_d = data.get('eps_demand', 0.0) or 0.0
        eps_pop = data.get('eps_pop_demand', 0.0) or 0.0      # Population elasticity
        eps_inc = data.get('eps_income_demand', 0.0) or 0.0   # Income elasticity
        
        # Population and income are exogenous constants known before solving.
        pop_base = max(1e-6, data.get('pop_base', 1.0) or 1.0)
        pop_t = max(1e-6, data.get('pop_t', pop_base) or pop_base)
        inc_base = max(1e-6, data.get('inc_base', 1.0) or 1.0)
        inc_t = max(1e-6, data.get('inc_t', inc_base) or inc_base)
        
        # Apply population and income effects using powers, since these inputs are constants.
        # Keeping the multiplicative form here is correct.
        pop_ratio = pop_t / pop_base
        inc_ratio = inc_t / inc_base
        pop_effect = pop_ratio ** eps_pop if eps_pop != 0 else 1.0
        inc_effect = inc_ratio ** eps_inc if eps_inc != 0 else 1.0
        eps_pop_effective = eps_pop
        eps_inc_effective = eps_inc
        if nutrition_anchor_uses_profile:
            # The nutrition profile demand is already the target future food demand.
            # Do not multiply it again by population or income effects, otherwise
            # the elastic demand curve can sit outside the nutrition band.
            pop_effect = 1.0
            inc_effect = 1.0
            eps_pop_effective = 0.0
            eps_inc_effective = 0.0
        
        # Cross-price elasticities
        epsD_row_raw = {k: v for k, v in (data.get('epsD_row', {}) or {}).items() if k in comm_set}
        epsD_row_scaled = _scale_cross_terms(epsD_row_raw, cross_terms_scale)
        epsD_row_limited = _limit_cross_terms(epsD_row_scaled, cross_terms_top_n)
        epsD_row, sum_cross_eps, _ = _normalize_cross_eps(epsD_row_limited, eps_d)
        
        # Adjusted D0 baseline with multiplicative population and income effects
        feed_base_scaled = 0.0
        if feed_link_mode != 'off' and not nutrition_anchor_uses_profile:
            feed_base_scaled = _lookup_feed_base(r, j) * inv_qty_scale
        seed_base_scaled = 0. if nutrition_anchor_uses_profile else float(data.get('seed_reference_t',0.) or 0.) * inv_qty_scale
        D0_nonfeed = max(D0 - feed_base_scaled - seed_base_scaled, 0.0)
        D0_nonfeed_adjusted = D0_nonfeed * pop_effect * inc_effect

        feed_const_scaled = 0.0
        feed_expr = None
        feed_expr_used = False
        if feed_link_mode == 'iterative':
            feed_const_scaled = _lookup_feed_override(r, j, t) * inv_qty_scale
        elif feed_link_mode == 'dynamic_constraints':
            feed_expr = _build_dynamic_feed_expr(r, j, t)
            feed_expr_used = feed_expr is not None
        feed_credit_scaled = _actual_feed_credit(r,j,t,feed_expr if feed_expr_used else feed_const_scaled)

        # Phase 2: feed intensity change feed_reduction_by
        # For a feed intensity scenario, multiply feed-use commodity demand by (1 + rate).
        # Simplification: apply rate directly to D0_adjusted.
        feed_mult = 1.0
        if feed_reduction_by:
            rate = float(feed_reduction_by.get(key, 0.0) or 0.0)
            rate = max(-1.0, min(1.0, rate))  # Restrict to [-1, 1].
            feed_mult = 1.0 + rate
            if feed_link_mode == 'off':
                D0_nonfeed_adjusted = D0_nonfeed_adjusted * feed_mult
            elif feed_link_mode == 'iterative':
                feed_const_scaled *= feed_mult

        # Phase 3: losses ratio / waste delta (elasticity only)
        if method == 'elasticity' and (waste_reduction_by or losses_ratio_by) and loss_ratio_by_item and demand_item_map and normalize_comp_item:
            m49_code = _norm_m49_code(r)
            if not m49_code and region_to_m49:
                m49_code = region_to_m49.get(str(r).strip())
            if m49_code:
                items = demand_item_map.get(j, [])
                ratios = []
                for item in items:
                    item_norm = normalize_comp_item(item)
                    val = loss_ratio_by_item.get((m49_code, item_norm))
                    if val is not None and np.isfinite(val):
                        ratios.append(float(val))
                if items:
                    loss_ratio = float(sum(ratios) / len(ratios)) if ratios else 0.0
                    loss_delta, _ = _lookup_loss_delta(
                        key=key,
                        all_key=(r, 'All', t),
                        waste_reduction_by=waste_reduction_by,
                        losses_ratio_by=losses_ratio_by,
                    )
                    if loss_delta is not None:
                        mult = _loss_multiplier_from_delta(loss_ratio, loss_delta)
                        D0_nonfeed_adjusted *= mult
                        feed_const_scaled *= mult
                        if feed_expr_used and feed_expr is not None:
                            feed_expr = feed_expr * mult

        a_d = D0_nonfeed_adjusted * (1.0 - eps_d)
        b_d_abs = D0_nonfeed_adjusted * eps_d / P0
        b_d = b_d_abs
        if use_relative_price:
            price_ref_self = float(price_ref_by_comm.get(j, 1.0) or 1.0)
            if not np.isfinite(price_ref_self) or price_ref_self <= 0:
                price_ref_self = 1.0
            b_d = b_d_abs * price_ref_self
        
        # Filter tiny coefficients to avoid numerical instability.
        COEFF_THRESHOLD = 1e-6 * inv_qty_scale
        if abs(b_d) < COEFF_THRESHOLD:
            b_d = 0.0
            b_d_abs = 0.0

        price_wedge_self = _price_wedge(r, j, t)
        
        # Construct cross-price terms.
        cross_terms_d = gp.LinExpr(0.0)
        cross_min = 0.0
        cross_max = 0.0
        cross_wedge_shift = 0.0
        cross_const_offset_d = 0.0
        for other_comm, cross_eps in epsD_row.items():
            if other_comm == j:  # Skip the commodity itself.
                continue
            pc_other = _consumer_price_var(r, other_comm, t)
            if pc_other is not None:
                other_base_key = (r, other_comm, hist_end_year)
                P0_other = idx.get(other_base_key, {}).get('P0')
                if P0_other is None:
                    other_key = (r, other_comm, t)
                    P0_other = idx.get(other_key, {}).get('P0', P0) or P0
                P0_other = max(1e-6, P0_other)
                b_cross_abs = D0_nonfeed_adjusted * float(cross_eps) / P0_other
                b_cross = b_cross_abs
                pc_other_base_for_offset = P0_other
                if use_relative_price:
                    price_ref_other = float(price_ref_by_comm.get(other_comm, 1.0) or 1.0)
                    if not np.isfinite(price_ref_other) or price_ref_other <= 0:
                        price_ref_other = 1.0
                    b_cross = b_cross_abs * price_ref_other
                    pc_other_base_for_offset = P0_other / price_ref_other
                # Filter tiny cross-elasticity coefficients.
                b_cross = _sanitize_cross_coef(
                    b_cross,
                    base_qty=D0_nonfeed_adjusted,
                    cross_eps=float(cross_eps),
                    use_relative_price=use_relative_price,
                    ratio_tol=CROSS_COEF_RATIO_TOL,
                    logger=logger,
                    log_state=cross_fix_demand,
                    region=r,
                    commodity=j,
                    year=t,
                    other_comm=other_comm,
                    tag='demand',
                )
                if abs(b_cross) < COEFF_THRESHOLD:
                    continue
                cross_terms_d += b_cross * pc_other
                cross_const_offset_d += b_cross * pc_other_base_for_offset
                other_wedge = _price_wedge(r, other_comm, t)
                if other_wedge:
                    cross_wedge_shift += b_cross * other_wedge
                pmin_o, pmax_o = _price_bounds_for_comm(other_comm)
                if b_cross >= 0:
                    cross_min += b_cross * pmin_o
                    cross_max += b_cross * pmax_o
                else:
                    cross_min += b_cross * pmax_o
                    cross_max += b_cross * pmin_o
        
        a_d = a_d - cross_const_offset_d
        a_d_eff = a_d + b_d * price_wedge_self + cross_wedge_shift
        a_d_total = a_d_eff + feed_const_scaled + residual_scaled_for_profile
        pmin_j, pmax_j = _price_bounds_for_comm(j)
        # Conservative Pc bounds to keep demand nonnegative (ignore feed_expr).
        pc_bound_key = _pc_key(r, j, t)
        if b_d < -1e-12:
            ub = (a_d_total + cross_min) / max(-b_d, 1e-12)
            if np.isfinite(ub):
                prev = pc_nonneg_ub.get(pc_bound_key, pmax_j)
                if ub < prev:
                    pc_nonneg_ub[pc_bound_key] = ub
                    pc_nonneg_ub_src[pc_bound_key] = (r, j, t, float(ub))
        elif b_d > 1e-12:
            lb = (-a_d_total - cross_min) / max(b_d, 1e-12)
            if np.isfinite(lb):
                prev = pc_nonneg_lb.get(pc_bound_key, pmin_j)
                if lb > prev:
                    pc_nonneg_lb[pc_bound_key] = lb
                    pc_nonneg_lb_src[pc_bound_key] = (r, j, t, float(lb))
        else:
            if (a_d_total + cross_min) < 0:
                if use_regional_price:
                    pc_nonneg_impossible.append((r, j, t, "b_d=0", float(a_d_total + cross_min)))
                else:
                    pc_nonneg_impossible.append((j, t, "b_d=0", float(a_d_total + cross_min)))
        pc_self = _consumer_price_var(r, j, t)
        if pc_self is None:
            raise ValueError(f"[LINEAR] missing Pc for demand key {r},{j},{t}")
        if feed_expr_used and feed_expr is not None:
            feed_demand_expr_by_key[key] = feed_expr
            cd = m.addConstr(
                Qd[key] == a_d_total + b_d * pc_self + cross_terms_d + feed_expr - feed_credit_scaled,
                name=f"demand[{r},{j},{t}]",
            )
        else:
            cd = m.addConstr(
                Qd[key] == a_d_total + b_d * pc_self + cross_terms_d - feed_credit_scaled,
                name=f"demand[{r},{j},{t}]",
            )
        constr_demand[key] = cd
        _update_pc_range_diag(
            pc_range_diag,
            commodity=j,
            year=t,
            region=r,
            constr_type='demand',
            constr_name=f"demand[{r},{j},{t}]",
            a=a_d_total,
            b=b_d,
            cross_min=cross_min,
            cross_max=cross_max,
            q_lb=float(Qd_bounds.get(key, (Qmin, Qmax))[0]),
            q_ub=float(Qd_bounds.get(key, (Qmin, Qmax))[1]),
            pmin=pmin_j,
            pmax=pmax_j,
        )
        if str(r).lstrip("'") == "356" and int(t) == 2080:
            eps_d = data.get('eps_demand', 0.0) or 0.0
            eps_pop = data.get('eps_pop_demand', 0.0) or 0.0
            eps_inc = data.get('eps_income_demand', 0.0) or 0.0
            eps_d_row = epsD_row
            nz_d = {k: v for k, v in eps_d_row.items() if abs(v) > 1e-12}
            logger.info(
                "[DEMAND-DIAG] region=%s year=%s item=%s D0=%.6g P0=%.6g eps_d=%.6g eps_pop=%.6g eps_inc=%.6g "
                "a_d=%.6g b_d=%.6g cross_terms=%d",
                r, t, j, D0, P0, eps_d, eps_pop, eps_inc, a_d_total, b_d, len(nz_d)
            )
        
        # Cache calibration parameters.
        alpha_d_cache[key] = a_d_total
        eps_d_cache[key] = eps_d
        eps_pop_cache[key] = eps_pop_effective
        eps_inc_cache[key] = eps_inc_effective
        D0_cache[key] = D0
        pop_base_cache[key] = 1.0 if nutrition_anchor_uses_profile else pop_base
        inc_base_cache[key] = 1.0 if nutrition_anchor_uses_profile else inc_base

    # B03: production-linked seed, independent of food and residual adjustments.
    m.update()
    seed_demand_expr_by_key = {}
    for key in future_keys:
        r, j, t = key
        row = idx[key]
        if bool(row.get('seed_coefficient_unidentified', False)):
            m.addConstr(Qs[key] == 0.0, name=f'seed_missing_coefficient_no_new_production[{r},{j},{t}]')
        alpha = float(row.get('seed_coefficient', 0.0))
        if alpha <= 0:
            continue
        seed_demand_expr_by_key[key] = alpha * Qs[key]
        linked = []
        for prefix in ('demand_nutrition', 'demand_nutrition_lb', 'demand_nutrition_ub', 'demand'):
            con = m.getConstrByName(f'{prefix}[{r},{j},{t}]')
            if con is not None:
                linked.append(con)
        if key in constr_demand and constr_demand[key] not in linked:
            linked.append(constr_demand[key])
        for con in linked:
            m.chgCoeff(con, Qs[key], m.getCoeff(con, Qs[key]) - alpha)
    m.update()

    zero_demand_shutdown_constr: Dict[Tuple[str, str, int], gp.Constr] = {}
    if zero_nutrition_demand_keys:
        logger.info(
            "[LINEAR] zero nutrition demand keys=%d; skip Qs=0 shutdown because "
            "zero domestic demand can still be satisfied by export production through net_import",
            len(zero_nutrition_demand_keys),
        )
    
    # Demand nonneg Pc bounds.
    
    # Qd already has a nonnegative lower bound, so the demand equation itself
    # defines the necessary feasibility region. The bounds below are only a
    # conservative sufficient condition derived with cross_min (all cross-price
    # terms at their worst case simultaneously). Enforcing that approximation
    # as hard own-price bounds can make otherwise feasible regional Armington
    # models infeasible, especially when many cross-price terms are enabled.
    enforce_demand_nonneg_price_bounds = False
    demand_nonneg_constr: Dict[Tuple[Any, ...], gp.Constr] = {}
    ub_items = sorted(pc_nonneg_ub.items(), key=lambda x: x[1])
    lb_items = sorted(pc_nonneg_lb.items(), key=lambda x: x[1], reverse=True)
    if enforce_demand_nonneg_price_bounds:
        for key, ub in ub_items:
            if not np.isfinite(ub):
                continue
            if use_regional_price:
                r, j, t = key
            else:
                j, t = key
            pmin_j, pmax_j = _price_bounds_for_comm(j)
            if ub < pmin_j:
                if use_regional_price:
                    pc_nonneg_impossible.append((r, j, t, "ub<Pmin", float(ub)))
                else:
                    pc_nonneg_impossible.append((j, t, "ub<Pmin", float(ub)))
                continue
            ub_eff = min(pmax_j, ub)
            if ub_eff < pmax_j:
                if use_regional_price:
                    pc_var = Pc.get((r, j, t))
                    if pc_var is None:
                        continue
                    demand_nonneg_constr[(r, j, t, "ub")] = m.addConstr(
                        pc_var <= ub_eff,
                        name=f"demand_nonneg_ub[{r},{j},{t}]",
                    )
                else:
                    demand_nonneg_constr[(j, t, "ub")] = m.addConstr(
                        Pc[j, t] <= ub_eff,
                        name=f"demand_nonneg_ub[{j},{t}]",
                    )
        for key, lb in lb_items:
            if not np.isfinite(lb):
                continue
            if use_regional_price:
                r, j, t = key
            else:
                j, t = key
            pmin_j, pmax_j = _price_bounds_for_comm(j)
            if lb > pmax_j:
                if use_regional_price:
                    pc_nonneg_impossible.append((r, j, t, "lb>Pmax", float(lb)))
                else:
                    pc_nonneg_impossible.append((j, t, "lb>Pmax", float(lb)))
                continue
            lb_eff = max(pmin_j, lb)
            if lb_eff > pmin_j:
                if use_regional_price:
                    pc_var = Pc.get((r, j, t))
                    if pc_var is None:
                        continue
                    demand_nonneg_constr[(r, j, t, "lb")] = m.addConstr(
                        pc_var >= lb_eff,
                        name=f"demand_nonneg_lb[{r},{j},{t}]",
                    )
                else:
                    demand_nonneg_constr[(j, t, "lb")] = m.addConstr(
                        Pc[j, t] >= lb_eff,
                        name=f"demand_nonneg_lb[{j},{t}]",
                    )
    elif ub_items or lb_items:
        logger.info(
            "[LINEAR] demand_nonneg price bounds diagnostic only: proposed=%d (ub=%d, lb=%d); not added as hard constraints",
            len(ub_items) + len(lb_items),
            len(ub_items),
            len(lb_items),
        )
    if pc_nonneg_impossible:
        if use_regional_price:
            sample = ", ".join(
                f"{r}:{j}:{t}:{reason}:{val:.3e}" for r, j, t, reason, val in pc_nonneg_impossible[:10]
            )
        else:
            sample = ", ".join(f"{j}:{t}:{reason}:{val:.3e}" for j, t, reason, val in pc_nonneg_impossible[:10])
        logger.warning("[LINEAR] demand_nonneg diagnostic impossible bounds=%d sample=%s", len(pc_nonneg_impossible), sample)
    if demand_nonneg_constr:
        logger.info(
            "[LINEAR] demand_nonneg bounds added: %d (ub=%d, lb=%d)",
            len(demand_nonneg_constr),
            len([k for k in demand_nonneg_constr if k[-1] == "ub"]),
            len([k for k in demand_nonneg_constr if k[-1] == "lb"]),
        )
        if use_regional_price:
            ub_sample = ", ".join(
                f"{r}:{j}:{t}:{pc_nonneg_ub[(r,j,t)]:.3e}@{pc_nonneg_ub_src.get((r,j,t), ('', '', 0, 0.0))[0]}"
                for (r, j, t), _ in ub_items[:5]
                if (r, j, t) in pc_nonneg_ub
            )
            lb_sample = ", ".join(
                f"{r}:{j}:{t}:{pc_nonneg_lb[(r,j,t)]:.3e}@{pc_nonneg_lb_src.get((r,j,t), ('', '', 0, 0.0))[0]}"
                for (r, j, t), _ in lb_items[:5]
                if (r, j, t) in pc_nonneg_lb
            )
        else:
            ub_sample = ", ".join(
                f"{j}:{t}:{pc_nonneg_ub[(j,t)]:.3e}@{pc_nonneg_ub_src.get((j,t), ('', '', 0, 0.0))[0]}"
                for (j, t), _ in ub_items[:5]
                if (j, t) in pc_nonneg_ub
            )
            lb_sample = ", ".join(
                f"{j}:{t}:{pc_nonneg_lb[(j,t)]:.3e}@{pc_nonneg_lb_src.get((j,t), ('', '', 0, 0.0))[0]}"
                for (j, t), _ in lb_items[:5]
                if (j, t) in pc_nonneg_lb
            )
        if ub_sample:
            logger.info("[LINEAR] demand_nonneg ub sample=%s", ub_sample)
        if lb_sample:
            logger.info("[LINEAR] demand_nonneg lb sample=%s", lb_sample)

    # Log historical/future constraint counts.
    logger.info(
        f"[LINEAR] 供给约束: 历史跳过={n_hist_skipped}, 未来弹性={n_future_supply - n_nutrition_supply_driven}, "
        f"nutrition需求驱动={n_nutrition_supply_driven}, "
        f"tiny_floor={n_tiny_supply_floor} (Q0<={TINY_Q0_THRESHOLD:.0e} kt)"
    )
    if n_zero_price_shutdown_rebased:
        logger.info(
            "[LINEAR] zero-price shutdown supply curves: rebased=%d, supply_cross_terms_dropped=%d "
            "(Pc=0 => Qs=0 for those supply keys)",
            n_zero_price_shutdown_rebased,
            n_zero_price_shutdown_cross_dropped,
        )
    logger.info(f"[LINEAR] 需求约束: 历史跳过={n_hist_demand_skipped}, 未来弹性={n_future_demand}")
    if enable_output_diagnostics:
        eps_d_log_path = _write_eps_d_distribution(eps_d_cache, output_dir, hist_end_year)
        if eps_d_log_path:
            logger.info(f"[LINEAR] eps_d distribution log written: {eps_d_log_path}")
        pc_range_path = _write_pc_range_diagnosis(
            pc_range_diag,
            output_dir,
            use_relative_price=use_relative_price,
            price_ref_by_comm=price_ref_by_comm,
        )
        if pc_range_path:
            logger.info(f"[LINEAR] pc range diagnosis written: {pc_range_path}")
    
    # 2.5 Ruminant demand cap constraints (Phase 2: ruminant_intake_cap)
    
    # Constrain each commodity separately by (country, commodity, year).
    # Regional aggregation requires summing country caps within each region.
    # Qd[region,j,t] <= Σ(cap[country,j,t] for country in region)
    
    rumi_intake_constr: Dict[Tuple[str, str, int], gp.Constr] = {}
    if ruminant_intake_cap and method not in {'nutrition', 'nutrition_band', 'nutrition_anchor'}:
        logger.info(f"[LINEAR] 开始添加ruminant_intake_cap约束，共{len(ruminant_intake_cap)}条country-level caps")
        
        # Step 1: build the country-to-region mapping from nodes.
        country_to_region = {}
        for node in nodes:
            if node.country not in country_to_region:
                region = get_region(node.country, m49=getattr(node, 'm49_code', None), dict_v3_path=dict_v3_path)
                country_to_region[node.country] = region
        
        # Step 2: aggregate caps by (region, commodity, year).
        regional_caps: Dict[Tuple[str, str, int], float] = {}
        for (country_key, j_key, t_key), cap_val in ruminant_intake_cap.items():
            if cap_val is None or cap_val <= 0:
                continue
            
            # Map countries to regions.
            if country_key not in country_to_region:
                logger.warning(f"[LINEAR] 未找到country {country_key}的region映射，跳过")
                continue
            
            r_key = country_to_region[country_key]
            regional_key = (r_key, j_key, t_key)
            
            if regional_key not in regional_caps:
                regional_caps[regional_key] = 0.0
            regional_caps[regional_key] += cap_val
        
        logger.info(f"[LINEAR] 聚合后得到{len(regional_caps)}个regional-level caps")
        
        # Step 3: add constraints.
        constraint_count = 0
        for (r_key, j_key, t_key), cap_val in regional_caps.items():
            if t_key <= hist_end_year:
                continue
            if (r_key, j_key, t_key) in Qd:
                cap_scaled = float(cap_val) * inv_qty_scale
                con = m.addConstr(
                    Qd[r_key, j_key, t_key] <= cap_scaled,
                    name=f"rumi_cap[{r_key[:5]},{j_key[:12]},{t_key}]"
                )
                rumi_intake_constr[(r_key, j_key, t_key)] = con
                constraint_count += 1
        
        logger.info(f"[LINEAR]  已添加{constraint_count}个ruminant_intake_cap约束到模型")
    
    if ruminant_intake_cap and method in {'nutrition', 'nutrition_band', 'nutrition_anchor'}:
        logger.info("[LINEAR] demand_method=%s -> skip ruminant_intake_cap constraints (profile-only)", method)

    # 3. Regional balance via net imports
    regional_balance_constr: Dict[Tuple[str, str, int], gp.Constr] = {}
    for key in future_keys:
        r, j, t = key
        qs_var = Qs.get(key)
        qd_var = Qd.get(key)
        mi_var = net_import.get(key)
        if qs_var is None or qd_var is None or mi_var is None:
            continue
        bioenergy_scaled = float(bioenergy_crop_demand_map.get(key, 0.0) or 0.0) * inv_qty_scale
        regional_balance_constr[key] = m.addConstr(
            qs_var + mi_var == qd_var + bioenergy_scaled,
            name=f"balance[{r},{j},{t}]"
        )

    # 3. Global market clearing
    for j in commodities:
        for t in years:
            if t <= hist_end_year:
                continue
            region_keys = future_region_keys_by_comm_year.get((j, t), [])
            net_sum = gp.quicksum(net_import[(r, j, t)] for r in region_keys)
            if ces_state.get('records'):
                # Gross international flows must close without an external
                # source/sink represented by the legacy market slack.
                excess[j,t].UB=0.
                shortage[j,t].UB=0.
            m.addConstr(
                net_sum + excess[j, t] - shortage[j, t] == 0,
                name=f"clear[{j},{t}]"
            )
    if any(str(r).lstrip("'") == "356" for r in regions) and 2080 in years:
        logger.info(
            "[MARKET-BAL] region='356 year=2080 uses regional balance via net_import; "
            "global clearing enforced through net_import + slack."
        )
    
    
    # 3b. Optional hard shortage/surplus constraints based on global total energy demand
    
    # Use global total energy constraints instead of separate commodity constraints.
    # shortage_energy ≤ max_slack_rate × total_energy_demand
    # energy = sum(kcal_per_ton[j] * quantity[j]).
    
    def _resolve_directional_slack_rate(raw: Any, fallback: Optional[float], name: str) -> Optional[float]:
        if isinstance(raw, str):
            text = raw.strip().lower()
            if text in {"", "inherit", "default"}:
                return fallback
            if text in {"none", "null", "off", "false", "disable", "disabled"}:
                return None
        if raw is None:
            return None
        try:
            val = float(raw)
        except Exception:
            logger.warning("[LINEAR] %s=%r invalid; inherit max_slack_rate=%s", name, raw, fallback)
            return fallback
        if not np.isfinite(val) or val <= 0:
            logger.info("[LINEAR] %s=%r -> disabled", name, raw)
            return None
        return val

    shortage_slack_rate = _resolve_directional_slack_rate(
        max_shortage_slack_rate,
        max_slack_rate,
        "max_shortage_slack_rate",
    )
    excess_slack_rate = _resolve_directional_slack_rate(
        max_excess_slack_rate,
        max_slack_rate,
        "max_excess_slack_rate",
    )
    if max_slack_rate is not None:
        logger.info(f"[LINEAR] max_slack_rate input={max_slack_rate} ({max_slack_rate:.4%})")
    logger.info(
        "[LINEAR] directional slack rates: shortage=%s excess=%s",
        "disabled" if shortage_slack_rate is None else f"{shortage_slack_rate:.4%}",
        "disabled" if excess_slack_rate is None else f"{excess_slack_rate:.4%}",
    )
    slack_limit_constr: Dict[int, Tuple[gp.Constr, gp.Constr]] = {}
    slack_demand_terms: Dict[int, int] = {}
    if shortage_slack_rate is not None or excess_slack_rate is not None:
        if nutrient_per_unit_by_comm:  # Nutrient coefficients are required for conversion to energy.
            for t in years:
                # Apply shortage limits only in future years.
                if t <= hist_end_year:
                    continue
                
                # Calculate global total energy demand (kcal).
                total_energy_demand = gp.LinExpr(0.0)
                demand_terms = 0
                for j in commodities:
                    kcal_per_ton = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                    if kcal_per_ton > 0:
                        qd_vars = [Qd[r, j, t] for r in regions if (r, j, t) in Qd]
                        if qd_vars:
                            demand_sum_j = gp.quicksum(qd_vars)
                            total_energy_demand += kcal_per_ton * qty_scale * demand_sum_j
                            demand_terms += len(qd_vars)
                        bioenergy_t = sum(
                            float(bioenergy_crop_demand_map.get((r, j, t), 0.0) or 0.0)
                            for r in regions
                        )
                        if bioenergy_t > 0.0:
                            total_energy_demand += kcal_per_ton * bioenergy_t
                slack_demand_terms[t] = demand_terms
                if demand_terms <= 0:
                    logger.warning(
                        "[LINEAR] slack demand has 0 Qd terms for year %s; skip energy slack constraints",
                        t,
                    )
                    continue
                
                # Calculate energy shortage (kcal).
                energy_shortage = gp.LinExpr(0.0)
                for j in commodities:
                    kcal_per_ton = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                    if kcal_per_ton > 0:
                        energy_shortage += kcal_per_ton * qty_scale * shortage[j, t]
                
                # Calculate energy surplus (kcal).
                energy_excess = gp.LinExpr(0.0)
                for j in commodities:
                    kcal_per_ton = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                    if kcal_per_ton > 0:
                        energy_excess += kcal_per_ton * qty_scale * excess[j, t]
                
                # Constrain energy shortage/surplus <= corresponding fraction * total energy demand.
                if isinstance(total_energy_demand, gp.LinExpr) and total_energy_demand.size() > 0:
                    cn_shortage = None
                    cn_excess = None
                    if shortage_slack_rate is not None:
                        cn_shortage = m.addConstr(
                            energy_shortage <= shortage_slack_rate * total_energy_demand,
                            name=f"slack_limit_energy_short[{t}]"
                        )
                    if excess_slack_rate is not None:
                        cn_excess = m.addConstr(
                            energy_excess <= excess_slack_rate * total_energy_demand,
                            name=f"slack_limit_energy_excess[{t}]"
                        )
                    slack_limit_constr[t] = (cn_shortage, cn_excess)
            
            logger.info(
                "[LINEAR] 短缺/过剩能量约束: %d 个未来年份 "
                "(shortage=%s, excess=%s; 历史年份不限制)",
                len(slack_limit_constr),
                "disabled" if shortage_slack_rate is None else f"≤{shortage_slack_rate:.1%}",
                "disabled" if excess_slack_rate is None else f"≤{excess_slack_rate:.1%}",
            )
        else:
            logger.warning(f"[LINEAR] 未提供营养系数，跳过能量约束")
    
    
    # 4. Emissions constraints: E = sum(e0_proc * Qs) - sum(abatement).
    
    
    total_abatement_cost = gp.LinExpr(0.0)
    total_E_land = gp.LinExpr(0.0)
    total_E_other = gp.LinExpr(0.0)
    cost_method_key = str(cost_calculation_method or 'MACC').strip().lower()
    baseline_emissions_map: Dict[Tuple[str, str, int, str], float] = {}
    baseline_source: Optional[str] = None
    if baseline_scenario_result:
        baseline_raw = None
        if isinstance(baseline_scenario_result, dict):
            baseline_raw = baseline_scenario_result.get('baseline_emissions')
        if baseline_raw:
            baseline_emissions_map = _normalize_baseline_emissions(
                baseline_raw,
                dict_v3_path=dict_v3_path,
            )
            baseline_source = 'baseline_emissions'
        elif isinstance(baseline_scenario_result, dict) and 'Qs' in baseline_scenario_result:
            if cost_method_key == 'unit_cost':
                raise ValueError(
                    "Unit-cost attribution requires an independent "
                    "baseline_emissions ledger in tCO2e; a Qs-only reference "
                    "uses current-scenario intensities and cannot recover "
                    "fully removed processes."
                )
            baseline_Qs = baseline_scenario_result['Qs']
            for key in baseline_Qs.keys():
                r, j, t = key
                Q_base = baseline_Qs[key]
                e0_map = e0_by_region.get(key, {})
                for proc, e0p in e0_map.items():
                    e_base = float(e0p) * float(Q_base)
                    baseline_emissions_map[(r, j, t, proc)] = e_base
            baseline_source = 'Qs'
    if baseline_emissions_map:
        logger.info(
            "[LINEAR] baseline emissions loaded: %d entries (source=%s)",
            len(baseline_emissions_map),
            baseline_source,
        )
    
    # Select the cost calculation method.
    if cost_method_key == 'unit_cost':
        unit_cost_owner_mode, selected_unit_cost_keys = _resolve_unit_cost_attribution(
            active_strategy_cost_keys
        )
        if unit_cost_owner_mode == 'process':
            if not process_cost_mapping:
                raise ValueError(
                    "Unit-cost process attribution requires a non-empty "
                    "process_cost_mapping."
                )
            mapped_database_keys = {
                str(value).strip()
                for value in process_cost_mapping.values()
                if str(value).strip()
            }
            unmapped_selected_keys = sorted(
                set(selected_unit_cost_keys) - mapped_database_keys
            )
            if unmapped_selected_keys:
                raise ValueError(
                    "Selected v2 process cost key(s) are absent from "
                    f"process_cost_mapping: {unmapped_selected_keys}"
                )
        if unit_cost_owner_mode != 'none' and not baseline_emissions_map:
            raise ValueError(
                "Unit-cost attribution requires non-empty baseline_emissions "
                "when a process or system-strategy cost owner is active."
            )
        future_regions = sorted(
            {
                str(region)
                for region, _commodity, year in Qs.keys()
                if int(year) > int(hist_end_year)
            }
        )
        priced_regions = [
            region
            for region in future_regions
            if _cost_region_is_selected(region, strategy_cost_regions)
        ]
        if unit_cost_owner_mode != 'none' and future_regions and not priced_regions:
            raise ValueError(
                "The active unit-cost owner has no selected solver regions; "
                f"strategy_cost_regions={tuple(strategy_cost_regions or ())!r}"
            )
        baseline_overlap_regions = {
            str(region)
            for region, commodity, year, _process in baseline_emissions_map
            if int(year) > int(hist_end_year)
            and (region, commodity, int(year)) in Qs
        }
        missing_baseline_regions = sorted(
            set(priced_regions) - baseline_overlap_regions
        )
        if unit_cost_owner_mode != 'none' and missing_baseline_regions:
            raise ValueError(
                "BASE emissions do not overlap the current future solve slice "
                "for selected cost region(s): "
                f"{missing_baseline_regions[:10]}"
                + (
                    f" (+{len(missing_baseline_regions) - 10} more)"
                    if len(missing_baseline_regions) > 10
                    else ""
                )
            )
        missing_unit_cost_keys = [
            (region, database_key)
            for region in priced_regions
            for database_key in selected_unit_cost_keys
            if not unit_cost_data or (region, database_key) not in unit_cost_data
        ]
        if unit_cost_owner_mode != 'none' and missing_unit_cost_keys:
            raise KeyError(
                "Missing selected v2 unit-cost key(s): "
                f"{missing_unit_cost_keys[:10]}"
                + (
                    f" (+{len(missing_unit_cost_keys) - 10} more)"
                    if len(missing_unit_cost_keys) > 10
                    else ""
                )
            )
        invalid_unit_costs = []
        if unit_cost_owner_mode != 'none' and unit_cost_data:
            for region in priced_regions:
                for database_key in selected_unit_cost_keys:
                    lookup_key = (region, database_key)
                    try:
                        value = float(unit_cost_data[lookup_key])
                    except (KeyError, TypeError, ValueError):
                        invalid_unit_costs.append((lookup_key, unit_cost_data.get(lookup_key)))
                        continue
                    if not np.isfinite(value) or value < 0.0:
                        invalid_unit_costs.append((lookup_key, value))
        if invalid_unit_costs:
            raise ValueError(
                "Invalid selected v2 unit cost(s); values must be finite and "
                f"non-negative: {invalid_unit_costs[:10]}"
            )
    else:
        unit_cost_owner_mode, selected_unit_cost_keys = "process", V2_PROCESS_COST_KEYS
    if cost_method_key in {'off', 'none', 'skip', 'disabled'}:
        logger.info("[LINEAR] 成本模块已关闭：skip MACC / unit_cost abatement")

        for key in Qs.keys():
            r, j, t = key
            if t <= hist_end_year:
                continue
            e0_map = e0_by_region.get(key, {})
            sum_e0 = sum(float(v) for v in e0_map.values())

            emis_expr = sum_e0 * qty_scale * Qs[key]
            ce = m.addConstr(Eij[key] == emis_expr, name=f"E_def[{r},{j},{t}]")
            constr_Edef[key] = ce
            m.addConstr(Cij[key] == 0.0, name=f"C_def[{r},{j},{t}]")

            e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
            e_other = sum_e0 - e_land
            total_E_land += e_land * qty_scale * Qs[key]
            total_E_other += e_other * qty_scale * Qs[key]

    elif cost_method_key == 'unit_cost':
        
        # Method 2: calculate abatement costs from specified unit costs.
        # Cost = sum(unit_cost * abatement).
        # Abatement = E_baseline - E_current, relative to the BASE scenario.
        
        logger.info("[LINEAR] 使用单位成本方法计算减排成本")
        emission_kt_to_tco2e = KT_CO2E_TO_T_CO2E
        
        # Build the BASELINE emissions lookup.
        baseline_emissions = baseline_emissions_map
        if not baseline_emissions:
            logger.warning("[LINEAR] 未提供BASELINE情景结果，无法计算减排量和成本")
        
        # Index reference processes once so a process that is fully removed by
        # a scenario is still priced. Iterating only the current e0 map would
        # silently drop exactly those 100%-abatement cases.
        baseline_processes_by_node: Dict[Tuple[str, str, int], set] = defaultdict(set)
        for baseline_key in baseline_emissions:
            if not isinstance(baseline_key, tuple) or len(baseline_key) != 4:
                continue
            base_region, base_commodity, base_year, base_process = baseline_key
            try:
                base_year_i = int(base_year)
            except Exception:
                continue
            baseline_processes_by_node[
                (str(base_region), str(base_commodity), base_year_i)
            ].add(str(base_process))

        # Calculate emissions and costs for each node.
        use_baseline_for_macc = bool(baseline_emissions_map)
        baseline_missing = 0
        baseline_used = 0
        baseline_missing_samples: List[Tuple[str, str, int, str]] = []
        for key in Qs.keys():
            r, j, t = key
            if t <= hist_end_year:
                continue
            e0_map = e0_by_region.get(key, {})
            sum_e0 = sum(float(v) for v in e0_map.values())
            
            # Total emissions in the current scenario
            emis_expr = sum_e0 * qty_scale * Qs[key]
            
            # Calculate abatement and costs by process.
            cost_expr = gp.LinExpr(0.0)
            
            process_names = set(e0_map)
            process_names.update(
                baseline_processes_by_node.get((str(r), str(j), int(t)), set())
            )
            for proc in sorted(process_names):
                e0p = float(e0_map.get(proc, 0.0) or 0.0)
                if not np.isfinite(e0p) or e0p < 0:
                    continue
                
                # Current process emissions. e0_by_proc uses ktCO2e/t for compatibility with
                # emissions outputs; unit-cost abatement is priced in tCO2e.
                e_current = e0p * qty_scale * Qs[key]
                
                # Get the process-to-cost-process mapping from dict_v3.
                process_cost_name = None
                if process_cost_mapping:
                    process_cost_name = process_cost_mapping.get(proc)
                
                # Handle different cost mapping types.
                if process_cost_name is None:
                    # Skip if cost data are unavailable.
                    continue
                elif process_cost_name == 'Production value':
                    # Special handling: economic losses from De/Reforestation.
                    # Calculate reduced output due to land constraints relative to BASE, multiplied by market price.
                    # TODO: this requires post-solve production changes and prices.
                    # Omit from the model for now; handle during S4_0_main postprocessing.
                    pass
                else:
                    # Process costs are mutually exclusive with system-strategy
                    # costs.  Explicit singleton process runs price only their
                    # selected database key; an explicit empty selection prices
                    # no mitigation component (coalition/Shapley run).
                    if unit_cost_owner_mode != 'process':
                        continue
                    if process_cost_name not in selected_unit_cost_keys:
                        continue
                    if not _cost_region_is_selected(r, strategy_cost_regions):
                        continue

                    cost_lookup_key = (r, process_cost_name)
                    if not unit_cost_data or cost_lookup_key not in unit_cost_data:
                        continue
                    unit_cost = float(unit_cost_data[cost_lookup_key])
                    if not np.isfinite(unit_cost) or unit_cost < 0.0:
                        raise ValueError(
                            f"Invalid unit cost for {cost_lookup_key}: {unit_cost!r}"
                        )
                    if not baseline_emissions:
                        continue

                    e_baseline = float(baseline_emissions.get((r, j, t, proc), 0.0) or 0.0)
                    if e_baseline <= 0.0:
                        baseline_missing += 1
                        if len(baseline_missing_samples) < 5:
                            baseline_missing_samples.append((r, j, t, proc))
                        continue

                    abatement_key = (r, j, t, proc)
                    if unit_cost > 0.0:
                        # For a positive unit cost the convex epigraph is exact
                        # at the minimum and keeps the model linear.
                        abat_var = m.addVar(lb=0.0, name=f"abat_unit[{r},{j},{t},{proc}]")
                        m.addConstr(
                            abat_var >= e_baseline - emission_kt_to_tco2e * e_current,
                            name=f"abat_def_unit[{r},{j},{t},{proc}]",
                        )
                        cost_expr += unit_cost * abat_var
                        abatement_vars[abatement_key] = abat_var
                    else:
                        # A zero-cost epigraph would be unpinned.  Retain its
                        # exact affine quantity for deterministic post-solve
                        # evaluation instead of introducing a binary max.
                        zero_cost_abatement_specs[abatement_key] = (
                            e_baseline,
                            emission_kt_to_tco2e * e_current,
                        )
                    abatement_costs[abatement_key] = unit_cost
                    abatement_database_keys[abatement_key] = process_cost_name
                    baseline_used += 1
            
            # Emissions definition
            ce = m.addConstr(Eij[key] == emis_expr, name=f"E_def[{r},{j},{t}]")
            constr_Edef[key] = ce
            
            # Cost definition
            if cost_expr.size() > 0:
                m.addConstr(Cij[key] == cost_expr, name=f"C_def[{r},{j},{t}]")
                total_abatement_cost += Cij[key]
            else:
                m.addConstr(Cij[key] == 0.0, name=f"C_def[{r},{j},{t}]")
            
            # Distinguish LULUCF from other emissions.
            e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
            e_other = sum_e0 - e_land
            
            total_E_land += e_land * qty_scale * Qs[key]
            total_E_other += e_other * qty_scale * Qs[key]

        if unit_cost_owner_mode == 'process' and len(selected_unit_cost_keys) == 1:
            # A country/process singleton remains a valid priced run when its
            # reference has no positive opportunity for the selected key.
            # Emit an auditable zero row so downstream resume checks can tell
            # this case apart from a missing/corrupt cost summary.
            selected_process_key = selected_unit_cost_keys[0]
            represented_region_years = {
                (str(key[0]), int(key[2]))
                for key, database_key in abatement_database_keys.items()
                if isinstance(key, tuple)
                and len(key) >= 4
                and str(database_key) == selected_process_key
            }
            candidate_region_years = sorted(
                {
                    (str(region), int(year))
                    for region, _commodity, year in Qs
                    if int(year) > hist_end_year
                    and _cost_region_is_selected(region, strategy_cost_regions)
                }
            )
            for region, year_i in candidate_region_years:
                if (region, year_i) in represented_region_years:
                    continue
                lookup_key = (region, selected_process_key)
                if not unit_cost_data or lookup_key not in unit_cost_data:
                    continue
                placeholder_key = (
                    region,
                    "",
                    year_i,
                    f"__NO_BASELINE_OPPORTUNITY__:{selected_process_key}",
                )
                no_opportunity_abatement_specs[placeholder_key] = 0.0
                abatement_costs[placeholder_key] = float(unit_cost_data[lookup_key])
                abatement_database_keys[placeholder_key] = selected_process_key

        if unit_cost_owner_mode == 'strategy' and baseline_emissions:
            strategy_key = selected_unit_cost_keys[0]
            baseline_total_by_region_year: Dict[Tuple[str, int], float] = defaultdict(float)
            current_total_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
            for (base_region, commodity, year, process), raw_baseline in baseline_emissions.items():
                try:
                    year_i = int(year)
                    baseline_value = float(raw_baseline or 0.0)
                except Exception:
                    continue
                if year_i <= hist_end_year or baseline_value <= 0.0:
                    continue
                # System-strategy costs price production-process abatement only.
                # Explicit LULUCF is governed by the land-state/LUC objective and
                # can be redistributed between commodity rows by the emissions
                # postprocessor without a corresponding change in Qs.  Including
                # those rows here both double-counts land mitigation and breaks
                # the declared cost-basis closure against the physical ledger.
                if _is_lulucf_process(process):
                    continue
                if not _cost_region_is_selected(base_region, strategy_cost_regions):
                    continue
                qs_var = Qs.get((base_region, commodity, year_i))
                if qs_var is None:
                    # Rolling and excluded-commodity solves receive a broader
                    # reference table. Only price emissions represented in the
                    # current model slice; otherwise future steps are charged
                    # early and out-of-domain commodities become false full
                    # abatement.
                    continue
                region_year = (base_region, year_i)
                baseline_total_by_region_year[region_year] += baseline_value
                current_intensity = float(
                    (e0_by_region.get((base_region, commodity, year_i), {}) or {}).get(process, 0.0)
                    or 0.0
                )
                if not np.isfinite(current_intensity) or current_intensity < 0.0:
                    continue
                current_expr = current_total_by_region_year.get(region_year)
                if current_expr is None:
                    current_expr = gp.LinExpr(0.0)
                    current_total_by_region_year[region_year] = current_expr
                current_expr += (
                    emission_kt_to_tco2e * current_intensity * qty_scale * qs_var
                )

            for (region, year_i), baseline_total in sorted(baseline_total_by_region_year.items()):
                cost_lookup_key = (region, strategy_key)
                if not unit_cost_data or cost_lookup_key not in unit_cost_data:
                    logger.warning(
                        "[LINEAR][COST-V2] missing strategy unit cost for %s",
                        cost_lookup_key,
                    )
                    continue
                unit_cost = float(unit_cost_data[cost_lookup_key])
                if not np.isfinite(unit_cost) or unit_cost < 0.0:
                    raise ValueError(
                        f"Invalid strategy unit cost for {cost_lookup_key}: {unit_cost!r}"
                    )
                current_expr = current_total_by_region_year.get(
                    (region, year_i), gp.LinExpr(0.0)
                )
                strategy_abatement_key = (region, year_i, strategy_key)
                if unit_cost > 0.0:
                    abat_var = m.addVar(
                        lb=0.0,
                        name=f"abat_strategy[{region},{year_i},{strategy_key}]",
                    )
                    m.addConstr(
                        abat_var >= baseline_total - current_expr,
                        name=f"abat_strategy_def[{region},{year_i},{strategy_key}]",
                    )
                    total_abatement_cost += unit_cost * abat_var
                    strategy_abatement_vars[strategy_abatement_key] = abat_var
                else:
                    zero_cost_strategy_abatement_specs[strategy_abatement_key] = (
                        baseline_total,
                        current_expr,
                    )
                strategy_abatement_costs[strategy_abatement_key] = unit_cost

            logger.info(
                "[LINEAR][COST-V2] owner=%s strategy=%s priced_regions=%d zero_cost_regions=%d",
                unit_cost_owner_mode,
                strategy_key,
                len(strategy_abatement_vars),
                len(zero_cost_strategy_abatement_specs),
            )
        elif unit_cost_owner_mode == 'none':
            logger.info("[LINEAR][COST-V2] explicit no-owner mode; mitigation costs suppressed")

        if baseline_missing:
            logger.warning(
                "[LINEAR][COST-V2] process baseline unavailable/nonpositive for %d keys; sample=%s",
                baseline_missing,
                baseline_missing_samples,
            )
        logger.info(
            "[LINEAR][COST-V2] owner=%s selected_keys=%s process_abatement_keys=%d",
            unit_cost_owner_mode,
            list(selected_unit_cost_keys),
            baseline_used,
        )
    
    else:
        
        # Method 1: original MACC abatement cost calculation
        
        logger.info("[LINEAR] 使用MACC方法计算减排成本")
        
        if has_macc and not baseline_emissions_map:
            raise ValueError(
                "[LINEAR] MACC requires baseline emissions to compute delta_frac; "
                "set CFG['base_case'] and ensure baseline outputs exist."
            )
        baseline_missing_total = 0
        baseline_used_total = 0
        baseline_missing_samples: List[Tuple[str, str, int, str]] = []
        
        for key in Qs.keys():
            r, j, t = key
            if t <= hist_end_year:
                continue
            e0_map = e0_by_region.get(key, {})
            sum_e0 = sum(float(v) for v in e0_map.values())
            
            # Baseline emissions expression
            emis_expr = sum_e0 * qty_scale * Qs[key]
            
            # MACC abatement variables, when MACC data exist
            abat_vars_this_node: List[gp.Var] = []
            cost_terms: List[gp.LinExpr] = []
            
            if has_macc and sum_e0 > 0:
                
                for proc, e0p in e0_map.items():
                    e0p = float(e0p)
                    if e0p <= 0:
                        continue
                    
                    # Look up MACC data by region or globally.
                    dfp = pd.DataFrame()
                    if 'Country' in macc_df.columns:
                        dfp = macc_df[(macc_df['Country'] == r) & (macc_df['Process'] == proc)]
                    if dfp.empty and 'Process' in macc_df.columns:
                        dfp = macc_df[macc_df['Process'] == proc]
                    
                    if dfp.empty:
                        continue
                    
                    # Parse the MACC curve.
                    if 'cumulative_fraction_of_process' not in dfp.columns or 'marginal_cost_$per_tco2e' not in dfp.columns:
                        continue
                    dfp = dfp[['cumulative_fraction_of_process', 'marginal_cost_$per_tco2e']].dropna()
                    if dfp.empty:
                        continue
                    dfp = dfp.sort_values('cumulative_fraction_of_process')
                    
                    base_emis = baseline_emissions_map.get((r, j, t, proc))
                    if base_emis is None or not np.isfinite(base_emis) or base_emis <= 0:
                        baseline_missing_total += 1
                        if len(baseline_missing_samples) < 5:
                            baseline_missing_samples.append((r, j, t, proc))
                        continue
                    base_emis = float(base_emis)
                    baseline_used_total += 1
                    
                    prev_frac = 0.0
                    proc_abat_vars: List[gp.Var] = []
                    proc_cost_vars: List[gp.Var] = []
                    
                    for seg_idx, (_, row) in enumerate(dfp.iterrows()):
                        frac = float(row['cumulative_fraction_of_process'])
                        mu = float(row['marginal_cost_$per_tco2e'])
                        
                        if frac <= prev_frac:
                            continue
                        
                        # Create abatement variables a[r,j,t,proc,seg].
                        # Constraint: a <= delta_frac * baseline_emissions.
                        delta_frac = frac - prev_frac
                        
                        a_var = m.addVar(lb=0.0, name=f"a[{r},{j},{t},{proc},{seg_idx}]")
                        cap_val = delta_frac * base_emis
                        cap_con = m.addConstr(a_var <= cap_val, name=f"cap[{r},{j},{t},{proc},{seg_idx}]")
                        proc_cap_basecoeff[(r, j, t, proc, seg_idx)] = None
                        
                        abatement_vars[(r, j, t, proc, seg_idx)] = a_var
                        abatement_caps[(r, j, t, proc, seg_idx)] = cap_con
                        abat_vars_this_node.append(a_var)
                        proc_abat_vars.append(a_var)
                        
                        cost_var = m.addVar(lb=0.0, name=f"a_cost[{r},{j},{t},{proc},{seg_idx}]")
                        cost_cap = delta_frac * base_emis
                        cost_cap_con = m.addConstr(cost_var <= cost_cap, name=f"cap_cost[{r},{j},{t},{proc},{seg_idx}]")
                        abatement_cost_vars[(r, j, t, proc, seg_idx)] = cost_var
                        abatement_cost_caps[(r, j, t, proc, seg_idx)] = cost_cap_con
                        abatement_costs[(r, j, t, proc, seg_idx)] = mu
                        
                        proc_cost_vars.append(cost_var)
                        # cost_var is ktCO2e while mu is USD/tCO2e.
                        cost_terms.append(
                            _macc_cost_coefficient_usd_per_kt(mu) * cost_var
                        )
                        
                        prev_frac = frac
                    
                    if proc_cost_vars:
                        emis_proc_expr = e0p * qty_scale * Qs[key]
                        if proc_abat_vars:
                            emis_proc_expr -= gp.quicksum(proc_abat_vars)
                        abat_req = m.addVar(lb=0.0, name=f"abat_req[{r},{j},{t},{proc}]")
                        req_con = m.addConstr(
                            abat_req + emis_proc_expr >= base_emis,
                            name=f"abat_req[{r},{j},{t},{proc}]"
                        )
                        m.addConstr(abat_req <= base_emis, name=f"abat_req_cap[{r},{j},{t},{proc}]")
                        m.addConstr(gp.quicksum(proc_cost_vars) == abat_req, name=f"abat_req_sum[{r},{j},{t},{proc}]")
                        abatement_req_vars[(r, j, t, proc)] = abat_req
                        abatement_req_constr[(r, j, t, proc)] = req_con
                
                if baseline_missing_total:
                    raise ValueError(
                        f"[LINEAR] MACC baseline emissions missing for {baseline_missing_total} process keys; "
                        f"sample={baseline_missing_samples}"
                    )
            
            # Emissions definition: E = e0 * Qs - abatement.
            if abat_vars_this_node:
                emis_expr -= gp.quicksum(abat_vars_this_node)
            ce = m.addConstr(Eij[key] == emis_expr, name=f"E_def[{r},{j},{t}]")
            constr_Edef[key] = ce
            
            # Cost definition
            if cost_terms:
                m.addConstr(Cij[key] == gp.quicksum(cost_terms), name=f"C_def[{r},{j},{t}]")
                total_abatement_cost += Cij[key]
            else:
                m.addConstr(Cij[key] == 0.0, name=f"C_def[{r},{j},{t}]")
            
            # Distinguish LULUCF from other emissions for the land carbon price.
            e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
            e_other = sum_e0 - e_land
            
            # LULUCF abatement
            abat_land = gp.quicksum(
                abatement_vars.get((r, j, t, p, s), 0)
                for (rr, jj, tt, p, s) in abatement_vars.keys()
                if (rr, jj, tt) == (r, j, t) and _is_lulucf_process(p)
            ) if abatement_vars else 0.0
            
            total_E_land += e_land * qty_scale * Qs[key]
            if isinstance(abat_land, gp.LinExpr) and abat_land.size() > 0:
                total_E_land -= abat_land
            total_E_other += e_other * qty_scale * Qs[key]
    
        if has_macc:
            logger.info("[LINEAR] MACC baseline emissions used=%d", baseline_used_total)
    
    
    # 5. Optional nutrition constraints
    
    
    nutri_constr: Dict[Tuple[str, int], gp.Constr] = {}
    if method == 'nutrition' and nutrition_demand_map:
        logger.info("[LINEAR] demand_method=nutrition uses exact item demands; skip variable nutrition constraints so feed/residual demand is not counted as intake")
    elif nutrition_rhs and nutrient_per_unit_by_comm:
        for r in regions:
            for t in years:
                if t <= hist_end_year:  # Future years only
                    continue
                rhs = nutrition_rhs.get((r, t))
                if rhs is None:
                    continue
                # Skip NaN values to avoid Gurobi errors.
                try:
                    rhs_float = float(rhs)
                    if np.isnan(rhs_float) or np.isinf(rhs_float):
                        continue
                except (ValueError, TypeError):
                    continue
                
                expr = gp.LinExpr(0.0)
                for j in commodities:
                    if j in nonfood_commodities:
                        continue
                    if (r, j, t) in Qd:
                        v = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                        if v > 0:
                            expr += v * qty_scale * Qd[r, j, t]
                
                if expr.size() > 0:
                    cn = m.addConstr(expr >= rhs_float, name=f"nutri[{r},{t}]")
                    nutri_constr[(r, t)] = cn
        
        # Display nutrition constraint RHS values for the first five samples.
        if nutri_constr:
            logger.info(f"[LINEAR] 营养约束: {len(nutri_constr)} 个")
            sample_count = 0
            for (r, t), rhs_val in sorted(nutrition_rhs.items())[:5]:
                if (r, t) in nutri_constr:
                    logger.info(f"  - 样本: {r[:20]}, {t}年: RHS={rhs_val:.2e} kcal")
                    sample_count += 1
                    if sample_count >= 5:
                        break
    
    
    # 6. Optional land constraints
    
    # Actual future land occupation must not exceed the base-period physical total.
    # Crop/grass expansion maps changes in target land demand to net conversion flows.
    # Actual land stocks must depend only on base-period stocks plus conversion flows, without adding demand deltas,
    # otherwise rolling runs count the same change twice.
    # grassland_actual = base_grassland + crop_to_grass + forest_to_grass + other_to_grass - grass_to_crop - grass_to_forest
    # cropland_actual = base_cropland + grass_to_crop + forest_to_crop + other_to_crop - crop_to_grass - crop_to_forest
    # forest_nonneg_ratio only expands the gross forest_to_* convertible stock.
    # Actual forest stock remains the unscaled base forest, so postprocess cannot
    # create negative physical forest through the ratio reserve.
    # forest_nonneg: cropland_actual + grassland_actual + forest_actual <= base_total (including the nonforest expansion cap).
    # forest_area_by_region_year supplies a priority distribution; only its global
    # year total is enforced, so surplus regions can absorb more conversion.

    land_constr: Dict[Tuple[str, int], gp.Constr] = {}
    land_slack: Dict[Tuple[str, int], gp.Var] = {}
    land_constr_diagnostics = []
    global_forest_actual_expr_by_year: Dict[int, gp.LinExpr] = {}
    global_forest_target_by_year: Dict[int, float] = defaultdict(float)
    global_forest_target_constr_count = 0
    luc_qs_forest_cap_constraints = 0
    forest_global_target_shortfall: Dict[int, gp.Var] = {}
    forest_global_target_surplus: Dict[int, gp.Var] = {}
    forest_global_target_constr: Dict[int, gp.Constr] = {}
    luc_qs_forest_cap_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    luc_qs_forest_cap_rhs_by_region_year: Dict[Tuple[str, int], float] = {}
    land_demand_expansion_need: Dict[Tuple[str, str, int], gp.Var] = {}
    land_demand_contraction_need: Dict[Tuple[str, str, int], gp.Var] = {}
    cropland_demand_effective_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    grassland_demand_effective_expr_by_region_year: Dict[Tuple[str, int], gp.LinExpr] = {}
    land_stock_absorb_rows = 0
    land_stock_absorb_samples: List[Tuple[str, int, float, float, float, float]] = []
    land_soft_enabled = bool(land_soft_constraints_enabled)
    try:
        land_slack_rate = float(land_slack_max_rate) if land_slack_max_rate is not None else None
    except Exception:
        land_slack_rate = None
    if land_slack_rate is not None and land_slack_rate < 0:
        land_slack_rate = 0.0
    try:
        land_slack_penalty_val = float(land_slack_penalty) if land_slack_penalty is not None else None
    except Exception:
        land_slack_penalty_val = None
    if land_slack_penalty_val is not None and land_slack_penalty_val <= 0:
        land_slack_penalty_val = None
    limit_ag_restoration_to_release = bool(limit_reforestation_to_released_ag_land)
    prevent_conversion_cycles = bool(prevent_land_conversion_cycles)
    reforest_cap_active = bool(reforestation_physical_cap_enabled)
    try:
        reforest_max_ratio = float(reforestation_max_forest_increase_ratio)
    except Exception:
        reforest_max_ratio = np.nan
    if not np.isfinite(reforest_max_ratio) or reforest_max_ratio < 0.0:
        reforest_max_ratio = 0.0
    land_conversion_guard_counts: Dict[str, int] = defaultdict(int)
    BASE_YEAR_FOR_LAND = 2020
    intens_cfg = LUCConfig()
    intens_per_usd = float(intens_cfg.intensification_per_usd or 0.0)
    intens_cap = float(intens_cfg.intensification_cap or 0.0)
    land_limit_reconcile_samples = []
    use_land_demand_delta_link = land_demand_mode == 'none'
    if not use_land_demand_delta_link:
        logger.info(
            "[LINEAR] land conversion link uses actual_state_coverage: "
            "conversion flows are driven by cropland_actual/grassland_actual demand coverage "
            "and forest targets, not by a separate demand-delta equality."
        )
    for r in regions:
        base_crop = float(base_cropland_area.get(r, 0.0) or 0.0)
        base_grass = float(base_grassland_area.get(r, 0.0) or 0.0)
        base_forest_raw = float(base_forest_area.get(r, 0.0) or 0.0)
        base_forest = base_forest_raw
        base_forest_convertible = (
            float(base_forest_area_scaled.get(r, base_forest_raw) or 0.0)
            if forest_conversion_stock_uses_scaled
            else base_forest_raw
        )
        crop_nonforest_extra_cap = float(nonforest_cropland_extra_cap_by_region.get(r, 0.0) or 0.0)
        grass_nonforest_extra_cap = float(nonforest_grassland_extra_cap_by_region.get(r, 0.0) or 0.0)
        nonforest_extra_cap = crop_nonforest_extra_cap + grass_nonforest_extra_cap
        calibrated_actual_base_total_ha = base_crop + base_grass + base_forest
        base_total_ha = base_crop + base_grass + base_forest + nonforest_extra_cap
        base_crop_demand = float(base_cropland_delta_anchor.get(r, 0.0) or 0.0)
        base_grass_demand = float(base_grassland_delta_anchor.get(r, 0.0) or 0.0)

        limit_ha = None
        if land_area_limits:
            limit = land_area_limits.get((r, BASE_YEAR_FOR_LAND))
            if limit is not None:
                try:
                    limit_float = float(limit)
                    if not np.isnan(limit_float) and not np.isinf(limit_float):
                        # land_area_limits were converted to ha during loading.
                        limit_ha = limit_float
                except (ValueError, TypeError):
                    limit_ha = None

        for t in years:
            if t <= hist_end_year:
                continue

            cropland_demand_raw_for_release = cropland_expr_by_region_year.get((r, t), gp.LinExpr(0.0))
            cropland_demand = cropland_demand_raw_for_release
            crop_scale = float(land_demand_crop_scale_by_region.get(r, 1.0) or 1.0)
            if crop_scale != 1.0:
                cropland_demand = crop_scale * cropland_demand
            if grassland_method == 'dynamic':
                grassland_demand_raw_for_release = grassland_expr_by_region_year.get((r, t), gp.LinExpr(0.0))
                grassland_demand = grassland_demand_raw_for_release
                grass_scale = float(land_demand_grass_scale_by_region.get(r, 1.0) or 1.0)
                if grass_scale != 1.0:
                    grassland_demand = grass_scale * grassland_demand
            else:
                grass_val = base_grass_demand
                if grass_area_by_region_year:
                    grass_val = float(grass_area_by_region_year.get((r, t), base_grass_demand) or 0.0)
                grassland_demand_raw_for_release = gp.LinExpr(grass_val)
                grassland_demand = gp.LinExpr(grass_val)

            land_price = 0.0
            if land_carbon_price_by_year:
                try:
                    land_price = float(land_carbon_price_by_year.get(t, 0.0) or 0.0)
                except Exception:
                    land_price = 0.0
            red = 0.0
            if land_price > 0 and intens_per_usd > 0 and intens_cap > 0:
                red = min(intens_per_usd * land_price, intens_cap)
                if red < 0.0:
                    red = 0.0
                if red > 1.0:
                    red = 1.0
            if red > 0:
                cropland_demand_eff = base_crop_demand + (1.0 - red) * (cropland_demand - base_crop_demand)
                grassland_demand_eff = base_grass_demand + (1.0 - red) * (grassland_demand - base_grass_demand)
            else:
                cropland_demand_eff = cropland_demand
                grassland_demand_eff = grassland_demand
            energy_crop_land_req = float(energy_crop_land_requirement_map.get((r, int(t)), 0.0) or 0.0)
            if energy_crop_land_req > 0.0:
                # Dedicated energy crops are non-food biomass land demand. They
                # are added after food-crop intensification so land-carbon-price
                # yield responses do not silently shrink exogenous biomass area.
                cropland_demand_eff = cropland_demand_eff + energy_crop_land_req
            cropland_demand_effective_expr_by_region_year[(r, t)] = cropland_demand_eff
            grassland_demand_effective_expr_by_region_year[(r, t)] = grassland_demand_eff
            crop_release_base = max(
                0.0,
                float(base_cropland_demand_raw.get(r, base_crop_demand) or 0.0),
            )
            if crop_release_base <= 0.0:
                crop_release_base = max(0.0, base_crop_demand)
            grass_release_base = max(
                0.0,
                float(base_grassland_demand_raw.get(r, base_grass_demand) or 0.0),
            )
            if grass_release_base <= 0.0:
                grass_release_base = max(0.0, base_grass_demand)
            if red > 0:
                cropland_release_demand_eff = (
                    crop_release_base
                    + (1.0 - red) * (cropland_demand_raw_for_release - crop_release_base)
                )
                grassland_release_demand_eff = (
                    grass_release_base
                    + (1.0 - red) * (grassland_demand_raw_for_release - grass_release_base)
                )
            else:
                cropland_release_demand_eff = cropland_demand_raw_for_release
                grassland_release_demand_eff = grassland_demand_raw_for_release
            if energy_crop_land_req > 0.0:
                cropland_release_demand_eff = cropland_release_demand_eff + energy_crop_land_req

            crop_exp_anchor = max(base_crop_demand, base_crop)
            grass_exp_anchor = max(base_grass_demand, base_grass)
            crop_delta = None
            grass_delta = None
            if use_land_demand_delta_link:
                crop_exp_need = m.addVar(lb=0.0, name=f"crop_expansion_need_beyond_stock[{r},{t}]")
                crop_contract_need = m.addVar(lb=0.0, name=f"crop_contraction_need_below_base[{r},{t}]")
                grass_exp_need = m.addVar(lb=0.0, name=f"grass_expansion_need_beyond_stock[{r},{t}]")
                grass_contract_need = m.addVar(lb=0.0, name=f"grass_contraction_need_below_base[{r},{t}]")
                land_demand_expansion_need[('crop', r, t)] = crop_exp_need
                land_demand_expansion_need[('grass', r, t)] = grass_exp_need
                land_demand_contraction_need[('crop', r, t)] = crop_contract_need
                land_demand_contraction_need[('grass', r, t)] = grass_contract_need
                m.addConstr(
                    crop_exp_need >= cropland_demand_eff - crop_exp_anchor,
                    name=f"crop_expansion_need_beyond_stock_def[{r},{t}]",
                )
                m.addConstr(
                    crop_contract_need >= base_crop_demand - cropland_demand_eff,
                    name=f"crop_contraction_need_below_base_def[{r},{t}]",
                )
                m.addConstr(
                    grass_exp_need >= grassland_demand_eff - grass_exp_anchor,
                    name=f"grass_expansion_need_beyond_stock_def[{r},{t}]",
                )
                m.addConstr(
                    grass_contract_need >= base_grass_demand - grassland_demand_eff,
                    name=f"grass_contraction_need_below_base_def[{r},{t}]",
                )
                crop_delta = crop_exp_need - crop_contract_need
                grass_delta = grass_exp_need - grass_contract_need
                if crop_exp_anchor > base_crop_demand + 1e-6 or grass_exp_anchor > base_grass_demand + 1e-6:
                    land_stock_absorb_rows += 1
                    if len(land_stock_absorb_samples) < 8:
                        land_stock_absorb_samples.append((
                            r,
                            int(t),
                            crop_exp_anchor - base_crop_demand,
                            grass_exp_anchor - base_grass_demand,
                            base_crop,
                            base_grass,
                        ))

            grass_to_crop = grassland_to_cropland.get((r, t))
            grass_to_forest = grassland_to_forest.get((r, t))
            grass_to_othernat = grassland_to_othernat.get((r, t))
            crop_to_grass = cropland_to_grassland.get((r, t))
            crop_to_forest = cropland_to_forest.get((r, t))
            crop_to_othernat = cropland_to_othernat.get((r, t))
            forest_to_crop = forest_to_cropland.get((r, t))
            forest_to_grass = forest_to_grassland.get((r, t))
            other_to_crop = nonforest_to_cropland.get((r, t), 0.0)
            other_to_grass = nonforest_to_grassland.get((r, t), 0.0)
            bg_forest_to_crop = float(background_forest_to_cropland.get((r, int(t)), 0.0) or 0.0)
            bg_forest_to_grass = float(background_forest_to_grassland.get((r, int(t)), 0.0) or 0.0)
            bg_crop_to_forest = float(background_cropland_to_forest.get((r, int(t)), 0.0) or 0.0)
            bg_grass_to_forest = float(background_grassland_to_forest.get((r, int(t)), 0.0) or 0.0)

            luc_crop_demand = luc_cropland_expr_by_region_year.get((r, t), gp.LinExpr(0.0))
            luc_crop_base = float(base_luc_cropland_demand.get(r, 0.0) or 0.0)
            luc_grass_base = float(base_grass_demand or 0.0)
            luc_forest_loss_expr = (
                cropland_demand_eff
                + grassland_demand_eff
                - crop_exp_anchor
                - grass_exp_anchor
                - other_to_crop
                - other_to_grass
                + bg_forest_to_crop
                + bg_forest_to_grass
                - bg_crop_to_forest
                - bg_grass_to_forest
            )
            luc_base_forest_raw = max(0.0, float(base_forest_area_luc_raw.get(r, base_forest_raw) or 0.0))
            luc_soft_rate = float(land_slack_rate or 0.0)
            luc_forest_loss_cap = luc_base_forest_raw * float(forest_nonneg_ratio) * (1.0 + luc_soft_rate)
            if luc_forest_loss_cap > 0.0 or luc_forest_loss_expr.size() > 0:
                luc_qs_forest_cap_expr_by_region_year[(r, t)] = luc_forest_loss_expr
                luc_qs_forest_cap_rhs_by_region_year[(r, t)] = float(luc_forest_loss_cap)
                m.addConstr(
                    luc_forest_loss_expr <= luc_forest_loss_cap,
                    name=f"luc_qs_forest_stock_cap[{r},{t}]",
                )
                luc_qs_forest_cap_constraints += 1

            if (
                grass_delta is not None
                and crop_to_grass is not None
                and forest_to_grass is not None
                and grass_to_crop is not None
                and grass_to_forest is not None
                and grass_to_othernat is not None
            ):
                m.addConstr(
                    crop_to_grass + forest_to_grass + other_to_grass + bg_forest_to_grass
                    - grass_to_crop - grass_to_forest - grass_to_othernat - bg_grass_to_forest
                    == grass_delta,
                    name=f"grass_expansion[{r},{t}]",
                )
            if (
                crop_delta is not None
                and grass_to_crop is not None
                and forest_to_crop is not None
                and crop_to_grass is not None
                and crop_to_forest is not None
                and crop_to_othernat is not None
            ):
                m.addConstr(
                    grass_to_crop + forest_to_crop + other_to_crop + bg_forest_to_crop
                    - crop_to_grass - crop_to_forest - crop_to_othernat - bg_crop_to_forest
                    == crop_delta,
                    name=f"crop_expansion[{r},{t}]",
                )

            cropland_actual = (
                base_crop
                + grass_to_crop
                + forest_to_crop
                + bg_forest_to_crop
                + other_to_crop
                - crop_to_grass
                - crop_to_forest
                - crop_to_othernat
                - bg_crop_to_forest
            )
            grassland_actual = (
                base_grass
                + crop_to_grass
                + forest_to_grass
                + bg_forest_to_grass
                + other_to_grass
                - grass_to_crop
                - grass_to_forest
                - grass_to_othernat
                - bg_grass_to_forest
            )
            forest_actual = (
                base_forest
                + crop_to_forest
                + grass_to_forest
                + bg_crop_to_forest
                + bg_grass_to_forest
                - forest_to_crop
                - forest_to_grass
                - bg_forest_to_crop
                - bg_forest_to_grass
            )

            if limit_ag_restoration_to_release or prevent_conversion_cycles or reforest_cap_active:
                crop_inflow = grass_to_crop + forest_to_crop + other_to_crop
                crop_outflow = crop_to_grass + crop_to_forest + crop_to_othernat
                grass_inflow = crop_to_grass + forest_to_grass + other_to_grass
                grass_outflow = grass_to_crop + grass_to_forest + grass_to_othernat
                forest_inflow = crop_to_forest + grass_to_forest
                forest_outflow = forest_to_crop + forest_to_grass

                if limit_ag_restoration_to_release:
                    # Use the unscaled base agricultural-use anchor, not the
                    # full LUH2 land-cover stock. This prevents baseline idle
                    # cropland/pasture from being counted as newly abandoned
                    # future land just because carbon price rewards restoration.
                    m.addConstr(
                        crop_outflow <= crop_release_base,
                        name=f"crop_outflow_limited_to_base_ag_use[{r},{t}]",
                    )
                    m.addConstr(
                        grass_outflow <= grass_release_base,
                        name=f"grass_outflow_limited_to_base_ag_use[{r},{t}]",
                    )
                    land_conversion_guard_counts['ag_outflow_base_use_cap_constraints'] += 2

                if prevent_conversion_cycles:
                    # Continuous anti-cycle guard: restoration must be backed by
                    # agricultural outflow, and forest loss must be backed by
                    # agricultural inflow demand. Pairwise complementarity would
                    # require binaries; this LP-safe guard prevents unbacked
                    # gross forest in/out loops while gross-conversion penalties
                    # minimize remaining churn.
                    m.addConstr(
                        forest_inflow <= crop_outflow + grass_outflow,
                        name=f"forest_inflow_backed_by_ag_release[{r},{t}]",
                    )
                    m.addConstr(
                        forest_outflow <= crop_inflow + grass_inflow,
                        name=f"forest_outflow_backed_by_ag_expansion[{r},{t}]",
                    )
                    land_conversion_guard_counts['continuous_cycle_guard_constraints'] += 2

                if reforest_cap_active:
                    reforest_cap_ha = max(0.0, base_forest * reforest_max_ratio)
                    m.addConstr(
                        forest_inflow <= reforest_cap_ha,
                        name=f"reforestation_physical_cap[{r},{t}]",
                    )
                    land_conversion_guard_counts['reforestation_physical_cap_constraints'] += 1

            m.addConstr(cropland_actual >= 0, name=f"cropland_actual_nonneg[{r},{t}]")
            m.addConstr(grassland_actual >= 0, name=f"grassland_actual_nonneg[{r},{t}]")
            m.addConstr(forest_actual >= 0, name=f"forest_actual_nonneg[{r},{t}]")
            # Actual agricultural land is a land-cover stock/capacity, not
            # necessarily the currently used production area. It must cover
            # effective demand, but unused baseline cropland/pasture can remain
            # idle without generating abandonment emissions unless conversion
            # flows explicitly move it to forest/other-natural land.
            m.addConstr(cropland_actual >= cropland_demand_eff, name=f"cropland_actual_covers_demand[{r},{t}]")
            m.addConstr(grassland_actual >= grassland_demand_eff, name=f"grassland_actual_covers_demand[{r},{t}]")
            if forest_area_by_region_year:
                forest_target = forest_area_by_region_year.get((r, t))
                if forest_target is not None:
                    try:
                        forest_target_val = float(forest_target)
                    except (TypeError, ValueError):
                        forest_target_val = np.nan
                    if np.isfinite(forest_target_val) and forest_target_val >= 0:
                        year_key = int(t)
                        global_expr = global_forest_actual_expr_by_year.get(year_key)
                        if global_expr is None:
                            global_expr = gp.LinExpr(0.0)
                        global_expr += forest_actual
                        global_forest_actual_expr_by_year[year_key] = global_expr
                        global_forest_target_by_year[year_key] += forest_target_val

            cropland_actual_expr_by_region_year[(r, t)] = cropland_actual
            grassland_actual_expr_by_region_year[(r, t)] = grassland_actual
            forest_actual_expr_by_region_year[(r, t)] = forest_actual
            crop_anchor_clip = float(land_anchor_clip_cropland_by_region.get(r, 0.0) or 0.0)
            grass_anchor_clip = float(land_anchor_clip_grassland_by_region.get(r, 0.0) or 0.0)
            cropland_luc_expr_by_region_year[(r, t)] = cropland_actual - other_to_crop - crop_anchor_clip
            grassland_luc_expr_by_region_year[(r, t)] = grassland_actual - other_to_grass - grass_anchor_clip

            land_expr = gp.LinExpr(0.0)
            land_expr += cropland_actual
            land_expr += grassland_actual
            land_expr += forest_actual

            if base_total_ha > 0 or land_expr.size() > 0:
                m.addConstr(land_expr <= base_total_ha, name=f"forest_nonneg[{r},{t}]")

            if land_expr.size() > 0 or grassland_method == 'static':
                if limit_ha is not None:
                    available_for_land = limit_ha
                    if (
                        bool(future_last_only)
                        and (forest_nonneg_ratio > 1.0 or nonforest_extra_cap > 0.0)
                        and calibrated_actual_base_total_ha + nonforest_extra_cap > available_for_land + 1e-6
                    ):
                        if len(land_limit_reconcile_samples) < 20:
                            land_limit_reconcile_samples.append(
                                (
                                    r,
                                    t,
                                    available_for_land,
                                    calibrated_actual_base_total_ha + nonforest_extra_cap,
                                    base_crop + base_grass + base_forest_convertible + nonforest_extra_cap,
                                )
                            )
                        available_for_land = calibrated_actual_base_total_ha + nonforest_extra_cap

                    if str(r).lstrip("'") == "356" and int(t) == 2080:
                        logger = logging.getLogger(__name__)
                        logger.info(
                            "[LAND-DIAG] region=%s year=%s land_limit_ha=%.6g base_forest_ha=%.6g base_grass_ha=%.6g available_for_land_ha=%.6g grassland_method=%s",
                            r, t, limit_ha, base_forest, base_grass, available_for_land, grassland_method
                        )
                        est_items = []
                        for j in commodities:
                            if j == "Fish, Seafood":
                                continue
                            node_data = idx.get((r, j, t))
                            if node_data is None:
                                continue
                            q0_val = float(node_data.get('Q0', 0.0) or 0.0)
                            yield_used = _cropland_yield0(
                                node_data,
                                region=r,
                                commodity=j,
                                year=t,
                                context="land_diag",
                            )
                            coef_crop = 1.0 / yield_used
                            coef_grass = float(node_data.get('grassland_coef', 0.0) or 0.0)
                            land_coef = coef_crop + (coef_grass if grassland_method == 'dynamic' else 0.0)
                            est_land = q0_val * land_coef
                            est_items.append((est_land, j, q0_val, yield_used, coef_crop, coef_grass))
                        est_items.sort(reverse=True, key=lambda x: x[0])
                        for est_land, j, q0_val, yield_used, coef_crop, coef_grass in est_items[:20]:
                            logger.info(
                                "[LAND-DIAG] item=%s Q0=%.6g yield_used=%.6g coef_crop=%.6g coef_grass=%.6g est_land_ha=%.6g",
                                j, q0_val, yield_used, coef_crop, coef_grass, est_land
                            )
                            if j == 'Sugar cane':
                                yield0_val = float(idx.get((r, j, t), {}).get('yield0', 0.0) or 0.0)
                                logger.info(
                                    "[LAND-DIAG-DETAIL] item=%s Q0=%.6g yield0=%.6g "
                                    "yield_used=%.6g coef_crop=1/yield_used=%.6g "
                                    "coef_grass=%.6g land_coef=%.6g",
                                    j, q0_val, yield0_val,
                                    yield_used, coef_crop, coef_grass,
                                    coef_crop + (coef_grass if grassland_method == 'dynamic' else 0.0)
                                )

                    diag_info = {
                        'region': r, 'year': t,
                        'land_limit_ha': limit_ha,
                        'forest_ha': base_forest,
                        'available_for_land_ha': available_for_land,
                        'constraint_added': available_for_land > 0
                    }
                    land_constr_diagnostics.append(diag_info)

                    if available_for_land > 0:
                        if land_soft_enabled:
                            slack_var = m.addVar(lb=0.0, name=f"land_slack[{r},{t}]")
                            if land_slack_rate is not None:
                                m.addConstr(
                                    slack_var <= available_for_land * land_slack_rate,
                                    name=f"land_slack_cap[{r},{t}]",
                                )
                            cl = m.addConstr(
                                land_expr <= available_for_land + slack_var,
                                name=f"landU[{r},{t}]",
                            )
                            land_slack[(r, t)] = slack_var
                        else:
                            cl = m.addConstr(land_expr <= available_for_land, name=f"landU[{r},{t}]")
                        land_constr[(r, t)] = cl
                    else:
                        if len(land_constr_diagnostics) <= 20:
                            print(f"     土地约束未创建: {r} {t}年, available={available_for_land:,.0f} ha <= 0")
                            print(f"       (limit={limit_ha:,.0f}, forest={base_forest:,.0f})")
    
    # [DIAGNOSTIC] Report commodity yields used.
    if idx:
        print(f"\n[Cropland产率诊断]")
        sample_commodities = ['Wheat', 'Maize', 'Rice', 'Soybeans', 'Cattle', 'Pigs', 'Poultry']
        sample_regions_for_yield = ['China', 'U.S.', 'India', 'Brazil']
        sample_year_for_yield = 2020
        
        print(f"  区域-商品产率示例（{sample_year_for_yield}年）：")
        for r in sample_regions_for_yield:
            for j in sample_commodities:
                node_data = idx.get((r, j, sample_year_for_yield))
                if node_data:
                    yield0 = node_data.get('yield0', 0.0)
                    if yield0 and yield0 > 0:
                        print(f"    {r:20s} | {j:15s}: {yield0:.2f} t/ha")
                    elif (r, j, sample_year_for_yield) in idx:
                        print(f"    {r:20s} | {j:15s}: yield0 missing/invalid")
    if luc_qs_forest_cap_constraints:
        logger.info(
            "[LINEAR] Qs-LUC forest stock cap constraints: %d "
            "(crop-only Qs land + grass demand - nonforest expansion <= base_forest*forest_nonneg_ratio*(1+soft))",
            luc_qs_forest_cap_constraints,
        )
    if land_stock_absorb_rows:
        logger.info(
            "[LINEAR] land demand delta uses base stock absorption band: rows=%d sample=%s",
            land_stock_absorb_rows,
            land_stock_absorb_samples,
        )
    if land_conversion_guard_counts:
        logger.info(
            "[LINEAR] land conversion safeguards active: "
            "limit_to_released_ag_land=%s prevent_cycles=%s "
            "reforestation_physical_cap=%s max_forest_increase_ratio=%.6g counts=%s",
            limit_ag_restoration_to_release,
            prevent_conversion_cycles,
            reforest_cap_active,
            reforest_max_ratio,
            dict(land_conversion_guard_counts),
        )
    
    try:
        forest_target_slack_rate = (
            float(forest_global_target_slack_max_rate)
            if forest_global_target_slack_max_rate is not None
            else None
        )
    except Exception:
        forest_target_slack_rate = None
    if forest_target_slack_rate is not None and forest_target_slack_rate < 0:
        forest_target_slack_rate = None

    for year_key, forest_expr in sorted(global_forest_actual_expr_by_year.items()):
        forest_target_val = float(global_forest_target_by_year.get(year_key, 0.0) or 0.0)
        if np.isfinite(forest_target_val) and forest_target_val >= 0:
            if bool(forest_global_target_slack_enabled):
                shortfall = m.addVar(lb=0.0, name=f"forest_global_target_shortfall[{year_key}]")
                surplus = m.addVar(lb=0.0, name=f"forest_global_target_surplus[{year_key}]")
                if forest_target_slack_rate is not None:
                    max_slack = forest_target_val * forest_target_slack_rate
                    m.addConstr(
                        shortfall <= max_slack,
                        name=f"forest_global_target_shortfall_cap[{year_key}]",
                    )
                    m.addConstr(
                        surplus <= max_slack,
                        name=f"forest_global_target_surplus_cap[{year_key}]",
                    )
                forest_global_target_shortfall[year_key] = shortfall
                forest_global_target_surplus[year_key] = surplus
                forest_global_target_constr[year_key] = m.addConstr(
                    forest_expr + shortfall - surplus == forest_target_val,
                    name=f"forest_global_target[{year_key}]",
                )
            else:
                forest_global_target_constr[year_key] = m.addConstr(
                    forest_expr == forest_target_val,
                    name=f"forest_global_target[{year_key}]",
                )
            global_forest_target_constr_count += 1

    if global_forest_target_constr_count:
        logger.info(
            "[LINEAR] global forest target constraints: %d (soft_slack=%s)",
            global_forest_target_constr_count,
            bool(forest_global_target_slack_enabled),
        )

    if land_constr_diagnostics:
        constraints_added = sum(1 for d in land_constr_diagnostics if d.get('constraint_added'))
        logger.info(
            "[LINEAR] landU constraints: attempted=%d added=%d soft=%s slack_vars=%d "
            "max_over_cap_rate=%s penalty_per_ha=%s",
            len(land_constr_diagnostics),
            constraints_added,
            land_soft_enabled,
            len(land_slack),
            land_slack_rate,
            land_slack_penalty_val,
        )

    if land_limit_reconcile_samples:
        logger.info(
            "[LINEAR] landU base-limit reconcile active: adjusted_region_years=%d "
            "(future_last_only=%s, forest_nonneg_ratio=%s)",
            len(land_limit_reconcile_samples),
            bool(future_last_only),
            forest_nonneg_ratio,
        )
        for r, t, old_limit, new_limit, scaled_limit in land_limit_reconcile_samples[:10]:
            logger.info(
                "[LINEAR] landU base-limit reconcile sample region=%s year=%s "
                "land_limit=%.6e calibrated_base_total=%.6e scaled_base_total=%.6e",
                r,
                t,
                old_limit,
                new_limit,
                scaled_limit,
            )

    # [DIAGNOSTIC] Summarize land constraint creation.
    if land_constr_diagnostics:
        total_attempts = len(land_constr_diagnostics)
        constraints_added = sum(1 for d in land_constr_diagnostics if d['constraint_added'])
        constraints_skipped = total_attempts - constraints_added
        
        print(f"\n[土地约束创建汇总]")
        print(f"  尝试创建: {total_attempts} 个 (区域, 年份) 组合")
        print(f"  成功创建: {constraints_added} 个约束")
        print(f"  跳过: {constraints_skipped} 个 (available_cropland <= 0)")
        
        # Print example constraints.
        sample_regions = ['China', 'U.S.', 'India', 'Brazil', 'EUR-Continental']
        sample_years = [2020, 2080]
        print(f"\n[约束示例]")
        for diag in land_constr_diagnostics:
            if diag['region'] in sample_regions and diag['year'] in sample_years:
                r, t = diag['region'], diag['year']
                status = "" if diag['constraint_added'] else ""
                print(f"  {status} {r} {t}: land_limit={diag['land_limit_ha']:,.0f} ha, "
                      f"forest={diag['forest_ha']:,.0f} ha, "
                      f"available_for_land={diag['available_for_land_ha']:,.0f} ha")

    
    # 6b. LUC emissions in optimization (optional)
    
    luc_emis_expr_by_year: Dict[int, gp.LinExpr] = {}
    luc_d_crop: Dict[Tuple[str, int], gp.Var] = {}
    luc_d_grass: Dict[Tuple[str, int], gp.Var] = {}
    luc_d_crop_pos: Dict[Tuple[str, int], gp.Var] = {}
    luc_d_grass_pos: Dict[Tuple[str, int], gp.Var] = {}
    luc_d_crop_neg: Dict[Tuple[str, int], gp.Var] = {}
    luc_d_grass_neg: Dict[Tuple[str, int], gp.Var] = {}
    luc_direct_emis_by_year: Dict[int, gp.LinExpr] = {}

    luc_mode = str(luc_opt_mode or 'none').strip().lower()
    shift_area_mode = str(luc_shift_area_mode or 'abs').strip().lower()
    if shift_area_mode not in {'abs', 'delta_pos'}:
        shift_area_mode = 'abs'

    direct_carbon_price = bool(luc_direct_carbon_price)
    use_gross_forest_luc_objective = bool(
        direct_carbon_price
        and land_carbon_price_by_year
        and (forest_to_cropland or forest_to_grassland)
    )
    need_luc_delta_pos = (
        shift_area_mode == 'delta_pos'
        or bool(luc_penalty_by_region_year)
        or (direct_carbon_price and land_carbon_price_by_year and not use_gross_forest_luc_objective)
    )
    need_luc_delta = luc_mode == 'explicit' or need_luc_delta_pos or use_gross_forest_luc_objective
    if need_luc_delta:
        years_sorted = sorted(set(years))
        future_years = [t for t in years_sorted if t > hist_end_year]
        prev_year_map = {years_sorted[i]: years_sorted[i - 1] if i > 0 else None
                         for i in range(len(years_sorted))}

        def _crop_luc_expr(rt: Tuple[str, int]) -> gp.LinExpr:
            r, t = rt
            expr = cropland_luc_expr_by_region_year.get((r, t))
            if expr is not None:
                return expr
            expr = cropland_actual_expr_by_region_year.get((r, t))
            if expr is not None:
                return expr
            return gp.LinExpr(0.0)

        def _grass_actual_expr(rt: Tuple[str, int]) -> gp.LinExpr:
            r, t = rt
            expr = grassland_actual_expr_by_region_year.get((r, t))
            if expr is not None:
                return expr
            if grassland_method == 'dynamic':
                return grassland_expr_by_region_year.get((r, t), gp.LinExpr(0.0))
            if grass_area_by_region_year:
                return gp.LinExpr(float(grass_area_by_region_year.get((r, t), 0.0) or 0.0))
            return gp.LinExpr(0.0)

        def _grass_luc_expr(rt: Tuple[str, int]) -> gp.LinExpr:
            r, t = rt
            expr = grassland_luc_expr_by_region_year.get((r, t))
            if expr is not None:
                return expr
            return _grass_actual_expr(rt)

        for r in regions:
            for t in future_years:
                prev_t = prev_year_map.get(t)
                if prev_t is None or prev_t < hist_end_year:
                    crop_prev = base_cropland_area.get(r, 0.0)
                    grass_prev = base_grassland_area.get(r, 0.0)
                else:
                    crop_prev = _crop_luc_expr((r, prev_t))
                    grass_prev = _grass_luc_expr((r, prev_t))

                crop_curr = _crop_luc_expr((r, t))
                grass_curr = _grass_luc_expr((r, t))

                d_crop = m.addVar(lb=-gp.GRB.INFINITY, name=f"luc_d_crop[{r},{t}]")
                d_grass = m.addVar(lb=-gp.GRB.INFINITY, name=f"luc_d_grass[{r},{t}]")
                m.addConstr(d_crop == crop_curr - crop_prev, name=f"luc_d_crop_def[{r},{t}]")
                m.addConstr(d_grass == grass_curr - grass_prev, name=f"luc_d_grass_def[{r},{t}]")
                luc_d_crop[(r, t)] = d_crop
                luc_d_grass[(r, t)] = d_grass

                if need_luc_delta_pos:
                    d_crop_pos = m.addVar(lb=0.0, name=f"luc_d_crop_pos[{r},{t}]")
                    d_crop_neg = m.addVar(lb=0.0, name=f"luc_d_crop_neg[{r},{t}]")
                    d_grass_pos = m.addVar(lb=0.0, name=f"luc_d_grass_pos[{r},{t}]")
                    d_grass_neg = m.addVar(lb=0.0, name=f"luc_d_grass_neg[{r},{t}]")
                    m.addConstr(d_crop == d_crop_pos - d_crop_neg, name=f"luc_d_crop_split[{r},{t}]")
                    m.addConstr(d_grass == d_grass_pos - d_grass_neg, name=f"luc_d_grass_split[{r},{t}]")
                    luc_d_crop_pos[(r, t)] = d_crop_pos
                    luc_d_grass_pos[(r, t)] = d_grass_pos
                    luc_d_crop_neg[(r, t)] = d_crop_neg
                    luc_d_grass_neg[(r, t)] = d_grass_neg

        if luc_mode == 'explicit':
            if use_gross_forest_luc_objective:
                logger.info(
                    "[LINEAR][LUC] luc_opt_mode=explicit uses gross forest conversion objective; "
                    "skip net land-state LUC pool approximation."
                )
            elif not luc_params:
                logger.warning("[LINEAR][LUC] luc_opt_mode=explicit but luc_params missing; skip LUC objective.")
            else:
                cfg = _extract_luc_params(luc_params)
                use_exp = bool(cfg.get('use_exponential_response', True))
                tau_veg = float(cfg.get('tau_veg', 20.0))
                tau_soil = float(cfg.get('tau_soil', 20.0))
                a_veg = 1.0 - math.exp(-1.0 / max(tau_veg, 1e-6)) if use_exp else 1.0
                a_soil = 1.0 - math.exp(-1.0 / max(tau_soil, 1e-6)) if use_exp else 1.0

                forest_c_ha = float(cfg.get('forest_c_ha', 150.0))
                cropland_c_ha = float(cfg.get('cropland_c_ha', 5.0))
                pasture_c_ha = float(cfg.get('pasture_c_ha', 10.0))
                forest_soil_c_ha = float(cfg.get('forest_soil_c_ha', 80.0))
                cropland_soil_c_ha = float(cfg.get('cropland_soil_c_ha', 50.0))
                pasture_soil_c_ha = float(cfg.get('pasture_soil_c_ha', 70.0))

                enable_shift = bool(cfg.get('enable_shift', True))
                tau_shift = float(cfg.get('tau_shift', 15.0))
                harvest_intensity = float(cfg.get('harvest_intensity', 1.0))
                if use_exp and tau_shift > 0:
                    shift_veg_frac = 1.0 - math.exp(-tau_shift / max(tau_veg, 1e-6))
                    shift_soil_frac = 1.0 - math.exp(-tau_shift / max(tau_soil, 1e-6))
                else:
                    shift_veg_frac = 1.0
                    shift_soil_frac = 1.0

                for r in regions:
                    prev_pool_veg_crop: Union[gp.LinExpr, float] = 0.0
                    prev_pool_soil_crop: Union[gp.LinExpr, float] = 0.0
                    prev_pool_veg_pasture: Union[gp.LinExpr, float] = 0.0
                    prev_pool_soil_pasture: Union[gp.LinExpr, float] = 0.0

                    for t in future_years:
                        d_crop = luc_d_crop.get((r, t))
                        d_grass = luc_d_grass.get((r, t))
                        if d_crop is None or d_grass is None:
                            continue

                        delta_veg_crop = (forest_c_ha - cropland_c_ha) * land_scale * d_crop
                        delta_soil_crop = (forest_soil_c_ha - cropland_soil_c_ha) * land_scale * d_crop
                        delta_veg_pasture = (forest_c_ha - pasture_c_ha) * land_scale * d_grass
                        delta_soil_pasture = (forest_soil_c_ha - pasture_soil_c_ha) * land_scale * d_grass

                        if enable_shift and tau_shift > 0:
                            if shift_area_mode == 'abs':
                                crop_abs = _crop_luc_expr((r, t))
                                grass_abs = _grass_luc_expr((r, t))
                                shift_crop = crop_abs / tau_shift
                                shift_grass = grass_abs / tau_shift
                            else:
                                shift_crop = luc_d_crop_pos.get((r, t), gp.LinExpr(0.0)) / tau_shift
                                shift_grass = luc_d_grass_pos.get((r, t), gp.LinExpr(0.0)) / tau_shift

                            delta_veg_crop += (
                                (forest_c_ha * shift_veg_frac - cropland_c_ha)
                                * land_scale * shift_crop * harvest_intensity
                            )
                            delta_soil_crop += (
                                (forest_soil_c_ha * shift_soil_frac - cropland_soil_c_ha)
                                * land_scale * shift_crop * harvest_intensity
                            )
                            delta_veg_pasture += (
                                (forest_c_ha * shift_veg_frac - pasture_c_ha)
                                * land_scale * shift_grass * harvest_intensity
                            )
                            delta_soil_pasture += (
                                (forest_soil_c_ha * shift_soil_frac - pasture_soil_c_ha)
                                * land_scale * shift_grass * harvest_intensity
                            )

                        pool_veg_crop = m.addVar(lb=-gp.GRB.INFINITY, name=f"luc_pool_veg_crop[{r},{t}]")
                        pool_soil_crop = m.addVar(lb=-gp.GRB.INFINITY, name=f"luc_pool_soil_crop[{r},{t}]")
                        pool_veg_pasture = m.addVar(lb=-gp.GRB.INFINITY, name=f"luc_pool_veg_pasture[{r},{t}]")
                        pool_soil_pasture = m.addVar(lb=-gp.GRB.INFINITY, name=f"luc_pool_soil_pasture[{r},{t}]")

                        pool_before_veg_crop = prev_pool_veg_crop + delta_veg_crop
                        pool_before_soil_crop = prev_pool_soil_crop + delta_soil_crop
                        pool_before_veg_pasture = prev_pool_veg_pasture + delta_veg_pasture
                        pool_before_soil_pasture = prev_pool_soil_pasture + delta_soil_pasture

                        m.addConstr(pool_veg_crop == (1.0 - a_veg) * pool_before_veg_crop,
                                    name=f"luc_pool_veg_crop_def[{r},{t}]")
                        m.addConstr(pool_soil_crop == (1.0 - a_soil) * pool_before_soil_crop,
                                    name=f"luc_pool_soil_crop_def[{r},{t}]")
                        m.addConstr(pool_veg_pasture == (1.0 - a_veg) * pool_before_veg_pasture,
                                    name=f"luc_pool_veg_pasture_def[{r},{t}]")
                        m.addConstr(pool_soil_pasture == (1.0 - a_soil) * pool_before_soil_pasture,
                                    name=f"luc_pool_soil_pasture_def[{r},{t}]")

                        emit_veg_crop = a_veg * pool_before_veg_crop
                        emit_soil_crop = a_soil * pool_before_soil_crop
                        emit_veg_pasture = a_veg * pool_before_veg_pasture
                        emit_soil_pasture = a_soil * pool_before_soil_pasture
                        emit_total = (emit_veg_crop + emit_soil_crop + emit_veg_pasture + emit_soil_pasture) * TC2CO2
                        luc_emis_expr_by_year[t] = luc_emis_expr_by_year.get(t, gp.LinExpr(0.0)) + emit_total

                        prev_pool_veg_crop = pool_veg_crop
                        prev_pool_soil_crop = pool_soil_crop
                        prev_pool_veg_pasture = pool_veg_pasture
                        prev_pool_soil_pasture = pool_soil_pasture

        if use_gross_forest_luc_objective:
            crop_ef_by_region, pasture_ef_by_region, ef_meta = _build_report_luc_gross_ef_by_region(
                luc_params,
                [str(r) for r in regions],
                dict_v3_path,
                luc_shift_area_mode=shift_area_mode,
                logger=logger,
            )
            grass_to_crop_ef_by_region, grass_to_crop_meta = _build_report_luc_grass_to_crop_ef_by_region(
                luc_params,
                [str(r) for r in regions],
                dict_v3_path,
                logger=logger,
            )
            crop_to_othernat_ef_by_region, grass_to_othernat_ef_by_region, othernat_meta = (
                _build_report_luc_to_othernat_ef_by_region(
                    luc_params,
                    [str(r) for r in regions],
                    dict_v3_path,
                    logger=logger,
                )
            )
            crop_terms = 0
            pasture_terms = 0
            crop_to_forest_terms = 0
            grass_to_forest_terms = 0
            grass_to_crop_terms = 0
            crop_to_othernat_terms = 0
            grass_to_othernat_terms = 0
            for (r, t), var in forest_to_cropland.items():
                coef = float(crop_ef_by_region.get(str(r), 0.0) or 0.0)
                if coef <= 0:
                    continue
                luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) + coef * land_scale * var
                crop_terms += 1
            for (r, t), var in forest_to_grassland.items():
                coef = float(pasture_ef_by_region.get(str(r), 0.0) or 0.0)
                if coef <= 0:
                    continue
                luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) + coef * land_scale * var
                pasture_terms += 1
            # Reforestation is the symmetric negative-emission counterpart of
            # deforestation. A positive land carbon price therefore rewards
            # abandoned cropland/grassland that is converted back to forest.
            for (r, t), var in cropland_to_forest.items():
                coef = float(crop_ef_by_region.get(str(r), 0.0) or 0.0)
                if coef <= 0:
                    continue
                luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) - coef * land_scale * var
                crop_to_forest_terms += 1
            for (r, t), var in grassland_to_forest.items():
                coef = float(pasture_ef_by_region.get(str(r), 0.0) or 0.0)
                if coef <= 0:
                    continue
                luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) - coef * land_scale * var
                grass_to_forest_terms += 1
            for (r, t), var in grassland_to_cropland.items():
                coef = float(grass_to_crop_ef_by_region.get(str(r), 0.0) or 0.0)
                if abs(coef) <= 1e-12:
                    continue
                luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) + coef * land_scale * var
                grass_to_crop_terms += 1
            for (r, t), var in cropland_to_othernat.items():
                coef = float(crop_to_othernat_ef_by_region.get(str(r), 0.0) or 0.0)
                if abs(coef) <= 1e-12:
                    continue
                luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) + coef * land_scale * var
                crop_to_othernat_terms += 1
            for (r, t), var in grassland_to_othernat.items():
                coef = float(grass_to_othernat_ef_by_region.get(str(r), 0.0) or 0.0)
                if abs(coef) <= 1e-12:
                    continue
                luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) + coef * land_scale * var
                grass_to_othernat_terms += 1

            crop_vals = [float(v or 0.0) for v in crop_ef_by_region.values()]
            pasture_vals = [float(v or 0.0) for v in pasture_ef_by_region.values()]
            grass_to_crop_vals = [float(v or 0.0) for v in grass_to_crop_ef_by_region.values()]
            crop_to_othernat_vals = [float(v or 0.0) for v in crop_to_othernat_ef_by_region.values()]
            grass_to_othernat_vals = [float(v or 0.0) for v in grass_to_othernat_ef_by_region.values()]
            if (
                crop_terms or pasture_terms or crop_to_forest_terms or grass_to_forest_terms
                or grass_to_crop_terms or crop_to_othernat_terms or grass_to_othernat_terms
            ):
                logger.info(
                    "[LINEAR][LUC] report-aligned gross LUC objective active: "
                    "forest_to_crop_terms=%d forest_to_grass_terms=%d "
                    "crop_to_forest_terms=%d grass_to_forest_terms=%d grass_to_crop_terms=%d "
                    "crop_to_othernat_terms=%d grass_to_othernat_terms=%d "
                    "forest_ef_country_specific_regions=%d forest_ef_fallback_regions=%d "
                    "grass_to_crop_country_specific_regions=%d grass_to_crop_fallback_regions=%d "
                    "othernat_country_specific_regions=%d othernat_fallback_regions=%d "
                    "shift_included=%s "
                    "crop_ef_tCO2_ha[min/mean/max]=%.6g/%.6g/%.6g "
                    "pasture_ef_tCO2_ha[min/mean/max]=%.6g/%.6g/%.6g "
                    "grass_to_crop_ef_tCO2_ha[min/mean/max]=%.6g/%.6g/%.6g "
                    "crop_to_othernat_ef_tCO2_ha[min/mean/max]=%.6g/%.6g/%.6g "
                    "grass_to_othernat_ef_tCO2_ha[min/mean/max]=%.6g/%.6g/%.6g",
                    crop_terms,
                    pasture_terms,
                    crop_to_forest_terms,
                    grass_to_forest_terms,
                    grass_to_crop_terms,
                    crop_to_othernat_terms,
                    grass_to_othernat_terms,
                    int(ef_meta.get('country_specific_regions', 0) or 0),
                    int(ef_meta.get('fallback_regions', 0) or 0),
                    int(grass_to_crop_meta.get('country_specific_regions', 0) or 0),
                    int(grass_to_crop_meta.get('fallback_regions', 0) or 0),
                    int(othernat_meta.get('country_specific_regions', 0) or 0),
                    int(othernat_meta.get('fallback_regions', 0) or 0),
                    bool(ef_meta.get('include_shift', False)),
                    min(crop_vals) if crop_vals else 0.0,
                    (sum(crop_vals) / len(crop_vals)) if crop_vals else 0.0,
                    max(crop_vals) if crop_vals else 0.0,
                    min(pasture_vals) if pasture_vals else 0.0,
                    (sum(pasture_vals) / len(pasture_vals)) if pasture_vals else 0.0,
                    max(pasture_vals) if pasture_vals else 0.0,
                    min(grass_to_crop_vals) if grass_to_crop_vals else 0.0,
                    (sum(grass_to_crop_vals) / len(grass_to_crop_vals)) if grass_to_crop_vals else 0.0,
                    max(grass_to_crop_vals) if grass_to_crop_vals else 0.0,
                    min(crop_to_othernat_vals) if crop_to_othernat_vals else 0.0,
                    (sum(crop_to_othernat_vals) / len(crop_to_othernat_vals)) if crop_to_othernat_vals else 0.0,
                    max(crop_to_othernat_vals) if crop_to_othernat_vals else 0.0,
                    min(grass_to_othernat_vals) if grass_to_othernat_vals else 0.0,
                    (sum(grass_to_othernat_vals) / len(grass_to_othernat_vals)) if grass_to_othernat_vals else 0.0,
                    max(grass_to_othernat_vals) if grass_to_othernat_vals else 0.0,
                )
            else:
                logger.warning(
                    "[LINEAR][LUC] gross LUC objective requested but no positive EF terms were added."
                )

        elif direct_carbon_price and land_carbon_price_by_year and luc_d_crop_pos and luc_d_grass_pos:
            cfg = _extract_luc_params(luc_params)
            forest_c_ha = float(cfg.get('forest_c_ha', 150.0))
            cropland_c_ha = float(cfg.get('cropland_c_ha', 5.0))
            pasture_c_ha = float(cfg.get('pasture_c_ha', 10.0))
            forest_soil_c_ha = float(cfg.get('forest_soil_c_ha', 80.0))
            cropland_soil_c_ha = float(cfg.get('cropland_soil_c_ha', 50.0))
            pasture_soil_c_ha = float(cfg.get('pasture_soil_c_ha', 70.0))
            delta_crop = max(0.0, (forest_c_ha - cropland_c_ha) + (forest_soil_c_ha - cropland_soil_c_ha)) * TC2CO2
            delta_pasture = max(0.0, (forest_c_ha - pasture_c_ha) + (forest_soil_c_ha - pasture_soil_c_ha)) * TC2CO2
            if delta_crop > 0:
                for (r, t), var in luc_d_crop_pos.items():
                    luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) + delta_crop * land_scale * var
                for (r, t), var in luc_d_crop_neg.items():
                    luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) - delta_crop * land_scale * var
            if delta_pasture > 0:
                for (r, t), var in luc_d_grass_pos.items():
                    luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) + delta_pasture * land_scale * var
                for (r, t), var in luc_d_grass_neg.items():
                    luc_direct_emis_by_year[t] = luc_direct_emis_by_year.get(t, gp.LinExpr(0.0)) - delta_pasture * land_scale * var

    
    # 7. Optional growth constraints, applied only to future years
    
    
    hist_max_constr: Dict[Tuple[str, str, int], gp.Constr] = {}
    if hist_max_production:
        # Default growth rate: 5% if unspecified.
        anchor_growth_rate = max_growth_rate_per_period if max_growth_rate_per_period is not None else 0.05
        n_hist_constrs = 0
        relax_small_prod = not bool(future_last_only)
        exempt_threshold = None
        floor_threshold = None
        if relax_small_prod:
            try:
                exempt_val = float(hist_max_small_prod_exempt_t)
            except (TypeError, ValueError):
                exempt_val = np.nan
            if np.isfinite(exempt_val) and exempt_val > 0:
                exempt_threshold = exempt_val

            try:
                floor_val = float(hist_max_small_prod_floor_t)
            except (TypeError, ValueError):
                floor_val = np.nan
            if np.isfinite(floor_val) and floor_val > 0:
                floor_threshold = floor_val

            if exempt_threshold is not None and floor_threshold is not None and floor_threshold <= exempt_threshold:
                floor_threshold = exempt_threshold

        exempt_pair_count = 0
        exempt_constr_skipped = 0
        exempt_samples: List[Tuple[str, str, float, int]] = []
        floor_pair_count = 0
        floor_constr_adjusted = 0
        floor_samples: List[Tuple[str, str, float, float, int]] = []
        missing_hist_max_pairs = 0
        missing_hist_max_tiny_base_pairs = 0
        missing_hist_max_samples: List[Tuple[str, str, float, int]] = []
        
        for r in regions:
            for j in commodities:
                # Constrain only future years (> hist_end_year).
                future_years = [t for t in years if t > hist_end_year and (r, j, t) in Qs]
                if not future_years:
                    continue
                hist_max = hist_max_production.get((r, j))
                if hist_max is None or hist_max <= 0:
                    missing_hist_max_pairs += 1
                    base_q0_for_hist = 0.0
                    try:
                        base_q0_for_hist = float(idx.get((r, j, hist_end_year), {}).get('Q0', 0.0) or 0.0)
                    except Exception:
                        base_q0_for_hist = 0.0
                    if base_q0_for_hist <= TINY_Q0_THRESHOLD:
                        missing_hist_max_tiny_base_pairs += 1
                    if len(missing_hist_max_samples) < 10:
                        missing_hist_max_samples.append((r, j, base_q0_for_hist, len(future_years)))
                    continue

                hist_max_eff = float(hist_max)
                if relax_small_prod:
                    if exempt_threshold is not None and hist_max_eff <= exempt_threshold:
                        exempt_pair_count += 1
                        exempt_constr_skipped += len(future_years)
                        if len(exempt_samples) < 5:
                            exempt_samples.append((r, j, hist_max_eff, len(future_years)))
                        continue
                    if floor_threshold is not None and hist_max_eff < floor_threshold:
                        floor_pair_count += 1
                        floor_constr_adjusted += len(future_years)
                        if len(floor_samples) < 5:
                            floor_samples.append((r, j, hist_max_eff, floor_threshold, len(future_years)))
                        hist_max_eff = floor_threshold

                for t in future_years:
                    years_since = t - hist_end_year
                    max_allowed = hist_max_eff * ((1.0 + anchor_growth_rate) ** years_since)
                    max_allowed_scaled = max_allowed * inv_qty_scale
                    
                    cn = m.addConstr(
                        Qs[r, j, t] <= max_allowed_scaled,
                        name=f"hist_max_anchor[{r},{j},{t}]"
                    )
                    hist_max_constr[(r, j, t)] = cn
                    n_hist_constrs += 1
        
        if n_hist_constrs > 0:
            logger.info(f"[LINEAR] 历史最大产量约束: {n_hist_constrs} 个 (growth_rate={anchor_growth_rate:.2%})")
            if relax_small_prod and (exempt_threshold is not None or floor_threshold is not None):
                rule_bits = []
                if exempt_threshold is not None:
                    rule_bits.append(f"exempt<={exempt_threshold:.6g} t")
                if floor_threshold is not None:
                    rule_bits.append(f"floor<{floor_threshold:.6g} t")
                logger.info(
                    "[LINEAR] hist_max_anchor small-prod rule active (future_last_only=%s, %s)",
                    bool(future_last_only),
                    ", ".join(rule_bits) if rule_bits else "no-op",
                )
                logger.info(
                    "[LINEAR] hist_max_anchor small-prod summary: exempt_pairs=%d skipped_constrs=%d floored_pairs=%d adjusted_constrs=%d",
                    exempt_pair_count,
                    exempt_constr_skipped,
                    floor_pair_count,
                    floor_constr_adjusted,
                )
                if exempt_samples:
                    logger.info(
                        "[LINEAR] hist_max_anchor exempt sample: %s",
                        ", ".join(
                            f"{r}:{j}:{hist_max_val:.3g}t:{n_future}y"
                            for r, j, hist_max_val, n_future in exempt_samples
                        ),
                    )
                if floor_samples:
                    logger.info(
                        "[LINEAR] hist_max_anchor floor sample: %s",
                        ", ".join(
                            f"{r}:{j}:{hist_max_val:.3g}->{floor_val:.3g}t:{n_future}y"
                            for r, j, hist_max_val, floor_val, n_future in floor_samples
                        ),
                    )
            
            # Display historical maximum production anchors for representative commodities.
            future_year_sample = max(years) if years else hist_end_year + 60
            years_ahead = future_year_sample - hist_end_year
            sample_count = 0
            for (r, j), hist_max in (hist_max_production.items() if hist_max_production else []):
                if sample_count >= 5:  # Show only the first five samples.
                    break
                if hist_max > 0:
                    max_allowed = hist_max * ((1.0 + anchor_growth_rate) ** years_ahead)
                    logger.info(f"  - 样本: {r[:20]}, {j[:30]}: 历史最大={hist_max:.0f}, {future_year_sample}年上限={max_allowed:.0f} (×{max_allowed/hist_max:.1f})")
                    sample_count += 1
        if missing_hist_max_pairs:
            logger.warning(
                "[LINEAR] hist_max_anchor did not bind pairs with no positive historical max: "
                "pairs=%d tiny_base_pairs=%d sample=%s",
                missing_hist_max_pairs,
                missing_hist_max_tiny_base_pairs,
                missing_hist_max_samples,
            )
    
    
    # Objective function
    
    
    # Split the base objective:
    # 1) Feasibility/slack stage: minimize only supply-demand/trade soft-constraint slack.
    # 2) Cost stage: minimize costs/emissions over the minimum-slack solution set.
    # Preserve the original weighted objective for single-stage compatibility.
    curtail_penalty_val = supply_curtailment_penalty
    if curtail_penalty_val is None:
        curtail_penalty_val = slack_penalty if slack_penalty is not None else 1e6

    slack_stage_obj = gp.quicksum(
        excess[j, t] + shortage[j, t]
        for j in commodities for t in years if t > hist_end_year
    )
    SLACK_PENALTY = float(slack_penalty if slack_penalty is not None else 1e6) * qty_scale
    single_stage_obj = SLACK_PENALTY * slack_stage_obj

    nutrition_trade_stage_obj = gp.LinExpr(0.0)
    nutrition_trade_penalty = 0.0
    if nutrition_supply_driven and nutrition_import_pos:
        # Penalize total trade volume so the model first uses local carrying
        # capacity, then trades only when local land/capacity constraints bind.
        nutrition_trade_stage_obj = gp.quicksum(
            nutrition_import_pos[key] + nutrition_export_pos[key]
            for key in nutrition_import_pos
        )
        nutrition_trade_penalty = max(1.0, SLACK_PENALTY * 0.01)

    armington_stage_obj = gp.LinExpr(0.0)
    ARMINGTON_TRADE_PENALTY = float(
        armington_trade_slack_penalty
        if armington_trade_slack_penalty is not None
        else (slack_penalty if slack_penalty is not None else 1e6)
    ) * qty_scale
    if armington_slack_pos:
        armington_stage_obj = gp.quicksum(
            armington_slack_pos[key] + armington_slack_neg[key]
            for key in armington_slack_pos
        )
        slack_stage_obj += armington_stage_obj
        single_stage_obj += ARMINGTON_TRADE_PENALTY * armington_stage_obj

    supply_curtail_stage_obj = gp.LinExpr(0.0)
    SUPPLY_CURTAIL_PENALTY = float(curtail_penalty_val) * qty_scale
    if supply_curtailment and SUPPLY_CURTAIL_PENALTY > 0:
        supply_curtail_stage_obj = gp.quicksum(v for v in supply_curtailment.values())
        slack_stage_obj += supply_curtail_stage_obj
        single_stage_obj += SUPPLY_CURTAIL_PENALTY * supply_curtail_stage_obj

    cost_stage_obj = gp.LinExpr(0.0)

    if nutrition_trade_penalty > 0 and nutrition_trade_stage_obj.size() > 0:
        cost_stage_obj += nutrition_trade_penalty * nutrition_trade_stage_obj
        logger.info(
            "[LINEAR] demand_method=nutrition: local-supply priority active via trade volume penalty %.6g",
            nutrition_trade_penalty,
        )

    # Land soft-constraint penalty (per ha)
    if land_slack and land_slack_penalty_val is not None:
        cost_stage_obj += gp.quicksum(land_slack_penalty_val * land_scale * v for v in land_slack.values())

    if forest_global_target_shortfall or forest_global_target_surplus:
        try:
            forest_slack_penalty_val = (
                float(forest_global_target_slack_penalty)
                if forest_global_target_slack_penalty is not None
                else float(slack_penalty if slack_penalty is not None else 1e6)
            )
        except Exception:
            forest_slack_penalty_val = float(slack_penalty if slack_penalty is not None else 1e6)
        if forest_slack_penalty_val > 0:
            cost_stage_obj += forest_slack_penalty_val * land_scale * gp.quicksum(
                forest_global_target_shortfall.get(t, 0.0)
                + forest_global_target_surplus.get(t, 0.0)
                for t in sorted(set(forest_global_target_shortfall) | set(forest_global_target_surplus))
            )

    def _normalize_grassland_to_cropland_cost_mode(raw: Any) -> str:
        mode = str(raw or 'per_ha_cost').strip().lower().replace('-', '_')
        aliases = {
            'perha': 'per_ha_cost',
            'per_ha': 'per_ha_cost',
            'per_ha_cost': 'per_ha_cost',
            'cost': 'per_ha_cost',
            'continuous': 'per_ha_cost',
            'carbon': 'per_ha_cost',
            'carbon_cost': 'per_ha_cost',
            'ef_cost': 'per_ha_cost',
            'priority': 'priority_ranking',
            'ranking': 'priority_ranking',
            'priority_ranking': 'priority_ranking',
            'legacy': 'priority_ranking',
        }
        return aliases.get(mode, 'per_ha_cost')

    grass_to_crop_cost_mode = _normalize_grassland_to_cropland_cost_mode(
        grassland_to_cropland_cost_mode
    )
    grass_conv_penalty = float(grassland_conversion_penalty or 0.0)
    if grass_conv_penalty > 0 and grassland_to_cropland:
        cost_stage_obj += grass_conv_penalty * land_scale * gp.quicksum(grassland_to_cropland.values())
    logger.info(
        "[LINEAR] grassland_to_cropland cost mode=%s per_ha_penalty=%.6g "
        "direct_luc_carbon_price=%s",
        grass_to_crop_cost_mode,
        grass_conv_penalty,
        bool(luc_direct_carbon_price and land_carbon_price_by_year),
    )
    crop_to_grass_penalty = float(cropland_to_grassland_penalty or 0.0)
    if crop_to_grass_penalty > 0 and cropland_to_grassland:
        cost_stage_obj += crop_to_grass_penalty * land_scale * gp.quicksum(cropland_to_grassland.values())

    # These variables are introduced with lower-bound definitions for the
    # positive/negative land-demand pieces. A numerically meaningful per-ha
    # tie-break is required; otherwise the LP can carry large crop/grass
    # conversion cycles that do not change feasibility but explode postprocess
    # gross LUC. Keep this far below shortage/land-slack penalties.
    land_demand_delta_tiebreak_penalty = 100.0
    land_conversion_tiebreak_penalty = 100.0
    land_demand_delta_terms = []
    for delta_map in (land_demand_expansion_need, land_demand_contraction_need):
        if delta_map:
            land_demand_delta_terms.append(gp.quicksum(delta_map.values()))
    if land_demand_delta_terms:
        cost_stage_obj += land_demand_delta_tiebreak_penalty * land_scale * gp.quicksum(land_demand_delta_terms)
        logger.info(
            "[LINEAR] land demand delta tiebreak penalty active: penalty_per_ha=%.6g maps=%d",
            land_demand_delta_tiebreak_penalty,
            len(land_demand_delta_terms),
        )

    gross_land_conversion_terms = []
    for conv_map in (
        grassland_to_cropland,
        grassland_to_forest,
        grassland_to_othernat,
        cropland_to_grassland,
        cropland_to_forest,
        cropland_to_othernat,
        forest_to_cropland,
        forest_to_grassland,
        nonforest_to_cropland,
        nonforest_to_grassland,
    ):
        if conv_map:
            gross_land_conversion_terms.append(gp.quicksum(conv_map.values()))
    if gross_land_conversion_terms:
        cost_stage_obj += land_conversion_tiebreak_penalty * land_scale * gp.quicksum(gross_land_conversion_terms)
        logger.info(
            "[LINEAR] land conversion tiebreak penalty active: penalty_per_ha=%.6g maps=%d",
            land_conversion_tiebreak_penalty,
            len(gross_land_conversion_terms),
        )

    land_alloc_mode = str(land_conversion_allocation_mode or 'unconstrained').strip().lower()
    land_alloc_aliases = {
        'none': 'unconstrained',
        'off': 'unconstrained',
        'free': 'unconstrained',
        'unconstrained': 'unconstrained',
        'priority': 'priority_nonforest_pasture_forest',
        'priority_nonforest_pasture_forest': 'priority_nonforest_pasture_forest',
        'nonforest_pasture_forest': 'priority_nonforest_pasture_forest',
        'nonforest_grass_forest': 'priority_nonforest_pasture_forest',
    }
    land_alloc_mode = land_alloc_aliases.get(land_alloc_mode, 'priority_nonforest_pasture_forest')
    try:
        land_priority_penalty = float(land_conversion_priority_penalty_per_ha or 0.0)
    except Exception:
        land_priority_penalty = 1.0
    def _resolve_land_priority_weight(raw: Any, default: float) -> float:
        try:
            weight = float(default if raw is None else raw)
        except Exception:
            weight = float(default)
        if not np.isfinite(weight) or weight < 0.0:
            return 0.0
        return weight
    land_priority_weight_grass_to_crop = _resolve_land_priority_weight(
        land_priority_weight_grassland_to_cropland,
        1.0,
    )
    land_priority_weight_forest_to_crop = _resolve_land_priority_weight(
        land_priority_weight_forest_to_cropland,
        100.0,
    )
    land_priority_weight_forest_to_grass = _resolve_land_priority_weight(
        land_priority_weight_forest_to_grassland,
        100.0,
    )
    land_priority_ranking_active = (
        land_alloc_mode == 'priority_nonforest_pasture_forest'
        and land_priority_penalty > 0.0
        and grass_to_crop_cost_mode == 'priority_ranking'
    )
    if land_priority_ranking_active:
        # Source priority for new cropland/pasture in otherwise underdetermined
        # land allocations. The common gross-conversion tiebreak above still
        # minimizes total churn; these differential weights rank the source:
        # nonforest first, then existing pasture/grassland, forest last.
        priority_terms = []
        if grassland_to_cropland and land_priority_weight_grass_to_crop > 0.0:
            priority_terms.append(land_priority_weight_grass_to_crop * gp.quicksum(grassland_to_cropland.values()))
        if forest_to_cropland and land_priority_weight_forest_to_crop > 0.0:
            priority_terms.append(land_priority_weight_forest_to_crop * gp.quicksum(forest_to_cropland.values()))
        if forest_to_grassland and land_priority_weight_forest_to_grass > 0.0:
            priority_terms.append(land_priority_weight_forest_to_grass * gp.quicksum(forest_to_grassland.values()))
        if priority_terms:
            cost_stage_obj += land_priority_penalty * land_scale * gp.quicksum(priority_terms)
            logger.info(
                "[LINEAR] land conversion allocation priority active: "
                "mode=%s penalty_per_ha=%.6g weights(grass_to_crop=%.6g,forest_to_crop=%.6g,forest_to_grass=%.6g) "
                "order=nonforest->pasture/grassland->forest",
                land_alloc_mode,
                land_priority_penalty,
                land_priority_weight_grass_to_crop,
                land_priority_weight_forest_to_crop,
                land_priority_weight_forest_to_grass,
            )
    else:
        logger.info(
            "[LINEAR] land conversion allocation priority inactive: mode=%s "
            "grassland_to_cropland_cost_mode=%s priority_penalty=%.6g",
            land_alloc_mode,
            grass_to_crop_cost_mode,
            land_priority_penalty,
        )

    # Export priority for nutrition-driven supply:
    # 1) keep total trade volume small via nutrition_trade_penalty above;
    # 2) among feasible exporters, prefer countries with larger remaining
    # cropland/pasture carrying headroom;
    # 3) production cost, abatement cost, and land-carbon cost then rank suppliers.
    nutrition_export_headroom_weight_by_key: Dict[Tuple[str, str, int], float] = {}
    nutrition_export_headroom_penalty = 0.0
    if nutrition_supply_driven and nutrition_export_pos:
        headroom_weights: List[float] = []
        headroom_samples: List[Tuple[str, str, int, float, float]] = []
        for key, export_var in nutrition_export_pos.items():
            r, j, t = key
            if t <= hist_end_year or j == "Fish, Seafood":
                continue
            node_data = idx.get((r, j, t)) if idx else None
            if node_data is None:
                continue
            try:
                q0_for_land = float(node_data.get('Q0', 0.0) or 0.0)
            except Exception:
                q0_for_land = 0.0
            try:
                yield_check = float(node_data.get('yield0'))
            except Exception:
                yield_check = 0.0
            if (not node_data.get('livestock_feed_account', False)) and (not np.isfinite(yield_check) or yield_check <= 0.0) and q0_for_land <= 1e-3:
                # Already hard-blocked by no_export_missing_land_yield.
                continue
            try:
                yield_j = _cropland_yield0(
                    node_data,
                    region=r,
                    commodity=j,
                    year=t,
                    context="export_headroom_priority",
                )
            except Exception:
                continue
            crop_coef_ton = 1.0 / max(1e-12, yield_j)
            grass_coef_ton = 0.0
            if grassland_method == 'dynamic':
                try:
                    grass_coef_ton = float(node_data.get('grassland_coef', 0.0) or 0.0)
                except Exception:
                    grass_coef_ton = 0.0
                if not np.isfinite(grass_coef_ton) or grass_coef_ton < 0.0:
                    grass_coef_ton = 0.0

            base_crop_cap = max(0.0, float(base_cropland_area.get(r, 0.0) or 0.0))
            base_grass_cap = max(0.0, float(base_grassland_area.get(r, 0.0) or 0.0))
            if forest_conversion_stock_uses_scaled:
                forest_conv_cap = max(0.0, float(base_forest_area_scaled.get(r, base_forest_area.get(r, 0.0)) or 0.0))
            else:
                forest_conv_cap = max(0.0, float(base_forest_area.get(r, 0.0) or 0.0))
            crop_extra_cap = max(0.0, float(nonforest_cropland_extra_cap_by_region.get(r, 0.0) or 0.0))
            grass_extra_cap = max(0.0, float(nonforest_grassland_extra_cap_by_region.get(r, 0.0) or 0.0))
            crop_reserved = max(0.0, float(base_cropland_delta_anchor.get(r, 0.0) or 0.0))
            grass_reserved = max(0.0, float(base_grassland_delta_anchor.get(r, 0.0) or 0.0))
            crop_headroom_ha = max(0.0, base_crop_cap + crop_extra_cap + forest_conv_cap - crop_reserved)
            grass_headroom_ha = max(0.0, base_grass_cap + grass_extra_cap + forest_conv_cap - grass_reserved)

            capacity_tons: List[float] = []
            if crop_coef_ton > 0.0:
                capacity_tons.append(crop_headroom_ha / crop_coef_ton)
            if grass_coef_ton > 0.0:
                capacity_tons.append(grass_headroom_ha / grass_coef_ton)
            if not capacity_tons:
                continue
            headroom_tons = max(0.0, min(capacity_tons))

            base_node = idx.get((r, j, hist_end_year), {}) if idx else {}
            try:
                scale_tons = max(
                    abs(float(base_node.get('Q0', 0.0) or 0.0)),
                    abs(float(node_data.get('Q0', 0.0) or 0.0)),
                    1.0,
                )
            except Exception:
                scale_tons = 1.0
            scarcity = scale_tons / (headroom_tons + scale_tons)
            if not np.isfinite(scarcity):
                scarcity = 1.0
            scarcity = min(1.0, max(0.0, scarcity))
            if scarcity <= 0.0:
                continue
            nutrition_export_headroom_weight_by_key[key] = scarcity
            headroom_weights.append(scarcity)
            if len(headroom_samples) < 8:
                headroom_samples.append((r, j, int(t), headroom_tons, scarcity))

        if nutrition_export_headroom_weight_by_key:
            nutrition_export_headroom_penalty = max(
                1.0,
                (nutrition_trade_penalty * 0.1) if nutrition_trade_penalty > 0.0 else 1.0,
            )
            cost_stage_obj += nutrition_export_headroom_penalty * gp.quicksum(
                nutrition_export_headroom_weight_by_key[key] * nutrition_export_pos[key]
                for key in nutrition_export_headroom_weight_by_key
            )
            arr = np.array(headroom_weights, dtype=float)
            logger.info(
                "[LINEAR] nutrition export headroom priority active: terms=%d penalty=%.6g "
                "scarcity[min/p50/max]=%.4g/%.4g/%.4g sample=%s",
                len(nutrition_export_headroom_weight_by_key),
                nutrition_export_headroom_penalty,
                float(np.min(arr)),
                float(np.median(arr)),
                float(np.max(arr)),
                headroom_samples,
            )

    # Optional production cost proxy: sum(P0 * Qs). disable=True strictly disables it in every demand mode.
    # Supply-location ranking in nutrition mode must come from an explicit objective term; do not silently re-enable it here.
    production_cost_obj, PRODUCTION_COST_WEIGHT = _build_production_cost_objective(
        qs=Qs,
        p0_by_key=P0_cache,
        hist_end_year=hist_end_year,
        qty_scale=qty_scale,
        disable_production_cost_term=disable_production_cost_term,
        production_cost_weight=production_cost_weight,
        tax_unit_adder=tax_unit_adder,
    )
    cost_stage_obj += production_cost_obj
    if disable_production_cost_term:
        logger.info(
            "[LINEAR] production-cost term disabled by configuration "
            "(demand_method=%s; no implicit override)",
            method,
        )
    elif PRODUCTION_COST_WEIGHT > 0.0:
        logger.info(
            "[LINEAR] production-cost term enabled explicitly: weight=%.6g "
            "(demand_method=%s)",
            PRODUCTION_COST_WEIGHT,
            method,
        )

    # Add abatement costs.
    cost_stage_obj += total_abatement_cost

    # Add land carbon pricing: cp * E_land, encouraging lower LULUCF emissions.
    if land_carbon_price_by_year:
        for key in Qs.keys():
            r, j, t = key
            if t <= hist_end_year:
                continue
            cp = float(land_carbon_price_by_year.get(t, 0.0) or 0.0)
            if cp > 0:
                e0_map = e0_by_region.get(key, {})
                e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
                if e_land > 0:
                    cost_stage_obj += cp * e_land * qty_scale * Qs[key]

    # LUC emissions (explicit mode)
    if luc_emis_expr_by_year and land_carbon_price_by_year:
        for t, emis_expr in luc_emis_expr_by_year.items():
            cp = float(land_carbon_price_by_year.get(t, 0.0) or 0.0)
            if cp != 0:
                cost_stage_obj += cp * emis_expr

    # LUC emissions (area-based, only when the explicit linear LUC pool
    # objective is absent; otherwise optimize_luc would double count the same
    # land conversion signal.
    if luc_direct_emis_by_year and land_carbon_price_by_year and not luc_emis_expr_by_year:
        for t, emis_expr in luc_direct_emis_by_year.items():
            cp = float(land_carbon_price_by_year.get(t, 0.0) or 0.0)
            if cp != 0:
                cost_stage_obj += cp * emis_expr
    elif luc_direct_emis_by_year and luc_emis_expr_by_year:
        logger.info("[LINEAR][LUC] skip area-based direct LUC objective because explicit linear LUC objective is active")

    # LUC penalties from iterative loop (per-ha)
    if luc_penalty_by_region_year and land_carbon_price_by_year:
        for (r, t), pen in luc_penalty_by_region_year.items():
            if t <= hist_end_year:
                continue
            cp = float(land_carbon_price_by_year.get(t, 0.0) or 0.0)
            if cp == 0:
                continue
            crop_pen = float((pen or {}).get('crop', 0.0) or 0.0)
            grass_pen = float((pen or {}).get('pasture', 0.0) or 0.0)
            if crop_pen > 0 and (r, t) in luc_d_crop_pos:
                cost_stage_obj += cp * crop_pen * land_scale * luc_d_crop_pos[(r, t)]
            if grass_pen > 0 and (r, t) in luc_d_grass_pos:
                cost_stage_obj += cp * grass_pen * land_scale * luc_d_grass_pos[(r, t)]

    if ces_state.get('linearization_slacks'):
        ces_restoration = gp.quicksum(ces_state['linearization_slacks'])
        slack_stage_obj += ces_restoration
        single_stage_obj += 1e14 * qty_scale * ces_restoration
    single_stage_obj += cost_stage_obj

    m.setObjective(single_stage_obj, gp.GRB.MINIMIZE)
    
    
    # Cache
    
    
    m._nzf_cache = {
        # Variables
        'Pc': Pc, 'Pw': Pw, 'Qs': Qs, 'Qd': Qd,
        'armington_ces': ces_state,
        'supply_curtailment': supply_curtailment,
        'net_import': net_import,
        'nutrition_import_pos': nutrition_import_pos,
        'nutrition_export_pos': nutrition_export_pos,
        'armington_slack_pos': armington_slack_pos,
        'armington_slack_neg': armington_slack_neg,
        'Eij': Eij, 'Cij': Cij,
        'excess': excess, 'shortage': shortage,
        # Constraint references
        'constr_supply': constr_supply,
        'constr_demand': constr_demand,
        'constr_Edef': constr_Edef,
        'nutri_constr': nutri_constr,
        'land_constr': land_constr,
        'land_slack': land_slack,
        'forest_global_target_constr': forest_global_target_constr,
        'forest_global_target_shortfall': forest_global_target_shortfall,
        'forest_global_target_surplus': forest_global_target_surplus,
        'forest_global_target_by_year': dict(global_forest_target_by_year),
        'forest_global_actual_expr_by_year': global_forest_actual_expr_by_year,
        'rumi_intake_constr': rumi_intake_constr,  # Phase 2: ruminant demand cap constraints
        'zero_demand_shutdown_constr': zero_demand_shutdown_constr,
        'regional_balance_constr': regional_balance_constr,
        'nutrition_trade_abs_constr': nutrition_trade_abs_constr,
        'world_price_constr': world_price_constr,
        'armington_trade_constr': armington_trade_constr,
        'hist_max_constr': hist_max_constr,  # Historical maximum production anchor constraints
        'slack_demand_terms': slack_demand_terms,
        # MACC
        'abatement_vars': abatement_vars,
        'abatement_caps': abatement_caps,
        'abatement_cost_vars': abatement_cost_vars,
        'abatement_cost_caps': abatement_cost_caps,
        'abatement_req_vars': abatement_req_vars,
        'abatement_req_constr': abatement_req_constr,
        'abatement_costs': abatement_costs,
        'abatement_database_keys': abatement_database_keys,
        'zero_cost_abatement_specs': zero_cost_abatement_specs,
        'no_opportunity_abatement_specs': no_opportunity_abatement_specs,
        'strategy_abatement_vars': strategy_abatement_vars,
        'strategy_abatement_blocks': strategy_abatement_vars,
        'strategy_abatement_delta_vars': strategy_abatement_delta_vars,
        'strategy_abatement_costs': strategy_abatement_costs,
        'zero_cost_strategy_abatement_specs': zero_cost_strategy_abatement_specs,
        'strategy_cost_metadata': dict(cost_strategy_metadata or {}),
        'cost_database_metadata': dict(cost_database_metadata or {}),
        'active_strategy_cost_keys': tuple(selected_unit_cost_keys),
        'unit_cost_owner_mode': unit_cost_owner_mode,
        'proc_cap_basecoeff': proc_cap_basecoeff,
        # Calibration parameters
        'alpha_s': alpha_s_cache,
        'alpha_d': alpha_d_cache,
        'eps_s': eps_s_cache,
        'eps_d': eps_d_cache,
        'eps_pop': eps_pop_cache,
        'eps_inc': eps_inc_cache,
        'eta_y': eta_y_cache,
        'eta_temp': eta_temp_cache,
        'Q0': Q0_cache,
        'D0': D0_cache,
        'P0': P0_cache,
        'Ymult0': Ymult0_cache,
        'Tmult0': Tmult0_cache,
        'pop_base': pop_base_cache,
        'inc_base': inc_base_cache,
        # Metadata
        'regions': regions, 'commodities': commodities, 'years': years,
        'idx': idx,
        'e0_by_region': e0_by_region,
        'nutrient_per_unit_by_comm': nutrient_per_unit_by_comm,  # Used to calculate energy shortage
        'nutrition_demand_map': nutrition_demand_map,
        'nutrition_residual_demand_map': nutrition_residual_demand_map,
        'bioenergy_crop_demand_map': bioenergy_crop_demand_map,
        'energy_crop_land_requirement_map': energy_crop_land_requirement_map,
        'nonfood_commodities': sorted(nonfood_commodities),
        'cross_terms_top_n': cross_terms_top_n,
        'cross_terms_scale': cross_terms_scale,
        'feed_link_mode': feed_link_mode,
        'feed_link_livestock': sorted(feed_livestock_set),
        'feed_product_coefficients': dict(feed_coeff_map),
        'feed_demand_expr_by_key': feed_demand_expr_by_key,
        'seed_demand_expr_by_key': seed_demand_expr_by_key,
        'feed_credit_map': feed_credit_map,
        'feed_credit_scaled_by_key': feed_credit_scaled_by_key,
        'objective_feasibility': slack_stage_obj,
        'objective_cost': cost_stage_obj,
        'objective_single_stage': single_stage_obj,
        'nutrition_supply_driven': bool(nutrition_supply_driven),
        'nutrition_trade_penalty': float(nutrition_trade_penalty or 0.0),
        'nutrition_export_headroom_penalty': float(nutrition_export_headroom_penalty or 0.0),
        'nutrition_export_headroom_weight_by_key': dict(nutrition_export_headroom_weight_by_key),
        'disable_production_cost_term': disable_production_cost_term,
        'production_cost_weight': float(PRODUCTION_COST_WEIGHT or 0.0),
        'slack_penalty': float(slack_penalty if slack_penalty is not None else 1e6),
        'supply_curtailment_enabled': bool(supply_curtailment_enabled),
        'supply_curtailment_penalty': float(curtail_penalty_val),
        'zero_price_shutdown_enabled': bool(zero_price_shutdown_enabled),
        'zero_demand_production_shutdown': bool(zero_demand_production_shutdown),
        'land_slack_penalty': float(land_slack_penalty_val or 0.0),
        'armington_trade_slack_penalty': float(
            armington_trade_slack_penalty
            if armington_trade_slack_penalty is not None
            else (slack_penalty if slack_penalty is not None else 1e6)
        ),
        'cross_coef_overrides': {
            'supply': int(cross_fix_supply.get('count', 0) or 0),
            'demand': int(cross_fix_demand.get('count', 0) or 0),
            'ratio_tol': CROSS_COEF_RATIO_TOL,
        },
        'price_wedge_by_region_comm_year': price_wedge_by_region_comm_year_norm,
        'price_wedge_by_region_comm': price_wedge_by_region_comm_norm,
        'price_wedge_by_region': price_wedge_by_region_norm,
        'use_relative_price': use_relative_price,
        'pc_by_region': use_regional_price,
        'market_clearing_mode': market_clearing_mode,
        'price_ref_by_comm': price_ref_by_comm,
        'price_bounds': (Pmin, Pmax),
        'price_bounds_by_comm': price_bounds_by_comm,
        'price_bounds_mode': price_bounds_mode_norm,
        'price_bounds_p0_mult': price_bounds_p0_mult,
        'qty_scale': qty_scale,
        'land_scale': land_scale,
        'grassland_method': grassland_method,
        'yield_t_per_ha_default': yield_t_per_ha_default,
        'hist_end_year': hist_end_year,
        'future_last_only': bool(future_last_only),
        'hist_max_small_prod_exempt_t': hist_max_small_prod_exempt_t,
        'hist_max_small_prod_floor_t': hist_max_small_prod_floor_t,
        'grass_area_by_region_year': grass_area_by_region_year,
        'forest_area_by_region_year': forest_area_by_region_year,
        'forest_nonneg_ratio': float(forest_nonneg_ratio or 1.0),
        'cropland_nonforest_expand_ratio': float(cropland_nonforest_expand_ratio or 1.0),
        'pasture_nonforest_expand_ratio': float(pasture_nonforest_expand_ratio or 1.0),
        'land_conversion_allocation_mode': land_alloc_mode,
        'grassland_to_cropland_cost_mode': grass_to_crop_cost_mode,
        'land_conversion_priority_ranking_active': bool(land_priority_ranking_active),
        # Resolve equal-cost source choices after the primary solve, with fixed
        # production/demand. No artificial monetary weight is added in this mode.
        'land_source_selection_enabled': (
            land_alloc_mode == 'priority_nonforest_pasture_forest'
            and grass_to_crop_cost_mode == 'per_ha_cost'
        ),
        'grassland_conversion_penalty': float(grass_conv_penalty or 0.0),
        'land_conversion_priority_penalty_per_ha': float(land_priority_penalty or 0.0),
        'land_priority_weight_grassland_to_cropland': float(land_priority_weight_grass_to_crop or 0.0),
        'land_priority_weight_forest_to_cropland': float(land_priority_weight_forest_to_crop or 0.0),
        'land_priority_weight_forest_to_grassland': float(land_priority_weight_forest_to_grass or 0.0),
        'limit_reforestation_to_released_ag_land': bool(limit_ag_restoration_to_release),
        'prevent_land_conversion_cycles': bool(prevent_conversion_cycles),
        'reforestation_physical_cap_enabled': bool(reforest_cap_active),
        'reforestation_max_forest_increase_ratio': float(reforest_max_ratio),
        'land_conversion_guard_counts': dict(land_conversion_guard_counts),
        'grassland_to_cropland': grassland_to_cropland,
        'grassland_to_forest': grassland_to_forest,
        'grassland_to_othernat': grassland_to_othernat,
        'cropland_to_grassland': cropland_to_grassland,
        'cropland_to_forest': cropland_to_forest,
        'cropland_to_othernat': cropland_to_othernat,
        'forest_to_cropland': forest_to_cropland,
        'forest_to_grassland': forest_to_grassland,
        'background_forest_to_cropland': dict(background_forest_to_cropland),
        'background_forest_to_grassland': dict(background_forest_to_grassland),
        'background_cropland_to_forest': dict(background_cropland_to_forest),
        'background_grassland_to_forest': dict(background_grassland_to_forest),
        'nonforest_to_cropland': nonforest_to_cropland,
        'nonforest_to_grassland': nonforest_to_grassland,
        'cropland_actual_expr_by_region_year': cropland_actual_expr_by_region_year,
        'grassland_actual_expr_by_region_year': grassland_actual_expr_by_region_year,
        'forest_actual_expr_by_region_year': forest_actual_expr_by_region_year,
        'cropland_demand_effective_expr_by_region_year': cropland_demand_effective_expr_by_region_year,
        'grassland_demand_effective_expr_by_region_year': grassland_demand_effective_expr_by_region_year,
        'cropland_luc_expr_by_region_year': cropland_luc_expr_by_region_year,
        'grassland_luc_expr_by_region_year': grassland_luc_expr_by_region_year,
        'luc_qs_forest_cap_expr_by_region_year': luc_qs_forest_cap_expr_by_region_year,
        'luc_qs_forest_cap_rhs_by_region_year': luc_qs_forest_cap_rhs_by_region_year,
        'land_delta_anchor_to_available_stock': bool(land_delta_anchor_to_available_stock),
        'land_anchor_clip_cropland_by_region': dict(land_anchor_clip_cropland_by_region),
        'land_anchor_clip_grassland_by_region': dict(land_anchor_clip_grassland_by_region),
        'land_demand_expansion_need': land_demand_expansion_need,
        'land_demand_contraction_need': land_demand_contraction_need,
        'land_demand_calibration_mode': land_demand_mode,
        'land_demand_crop_scale_by_region': dict(land_demand_crop_scale_by_region),
        'land_demand_grass_scale_by_region': dict(land_demand_grass_scale_by_region),
        'base_cropland_demand_raw': dict(base_cropland_demand_raw),
        'base_grassland_demand_raw': dict(base_grassland_demand_raw),
        'base_cropland_demand': dict(base_cropland_delta_anchor),
        'base_grassland_demand': dict(base_grassland_delta_anchor),
        'base_cropland_area': dict(base_cropland_area),
        'base_grassland_area': dict(base_grassland_area),
        'base_forest_area': dict(base_forest_area),
        'base_forest_area_scaled': dict(base_forest_area_scaled),
        'tax_unit_adder': dict(tax_unit_adder or {}),
    }
    
    m.update()  # Update the model to get correct variable/constraint counts.
    logger.info(f"[LINEAR] 模型构建完成: 变量={m.NumVars}, 约束={m.NumConstrs}")
    if has_macc:
        logger.info(f"[LINEAR] 减排变量数: {len(abatement_vars)}")
    
    if cross_fix_supply.get('count', 0) or cross_fix_demand.get('count', 0):
        logger.warning(
            "[LINEAR] cross coef override: supply=%d demand=%d (ratio_tol=%.3g)",
            int(cross_fix_supply.get('count', 0) or 0),
            int(cross_fix_demand.get('count', 0) or 0),
            CROSS_COEF_RATIO_TOL,
        )

    return m


def _predict_excluded_commodities(
    nodes: List[Any],
    excluded_commodities: List[str],
    years: List[int],
    *,
    dict_v3_path: Optional[str] = None,
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    hist_end_year: int = 2020,
    demand_method: str = 'elasticity',
    nutrition_profile_xlsx: Optional[str] = None,
    nutrition_profile_sheet: Any = 0,
    nutrition_indicator: str = 'energy',
    nutrition_use_baseyear_for_future: bool = True,
    tax_unit_adder: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    waste_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    losses_ratio_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    use_relative_price: bool = False,
    relative_price_bounds: Tuple[float, float] = (0.1, 10.0),
    price_bounds: Tuple[float, float] = (1e-6, 1e6),
    price_bounds_mode: str = 'absolute',
    price_bounds_p0_mult: Tuple[float, float] = (0.1, 10.0),
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    if not excluded_commodities:
        return {}

    exclude_norm = {str(c).strip().lower() for c in excluded_commodities if str(c).strip()}
    if not exclude_norm:
        return {}

    use_regional_agg = is_region_aggregation_enabled()
    if use_regional_agg:
        regional_df = aggregate_nodes_to_regions(
            nodes,
            dict_v3_path=dict_v3_path,
            population_by_country_year=population_by_country_year,
            income_mult_by_country_year=income_mult_by_country_year,
            hist_end_year=hist_end_year,
        )
    else:
        base_p0_by_country_comm: Dict[Tuple[str, str], float] = {}
        for n in nodes:
            if getattr(n, 'year', None) != hist_end_year:
                continue
            key_country = str(getattr(n, 'country', '')).strip()
            if not key_country:
                continue
            try:
                p0_val = float(getattr(n, 'P0', 0.0) or 0.0)
            except Exception:
                continue
            if np.isfinite(p0_val) and p0_val > 0:
                base_p0_by_country_comm[(key_country, getattr(n, 'commodity', None))] = p0_val
        records = []
        for n in nodes:
            pop_base = 1.0
            pop_t = 1.0
            inc_base = 1.0
            inc_t = 1.0
            if population_by_country_year:
                pop_base = float(population_by_country_year.get((n.country, hist_end_year), 1.0) or 1.0)
                pop_t = float(population_by_country_year.get((n.country, n.year), pop_base) or pop_base)
            if income_mult_by_country_year:
                inc_base = float(income_mult_by_country_year.get((n.country, hist_end_year), 1.0) or 1.0)
                inc_t = float(income_mult_by_country_year.get((n.country, n.year), inc_base) or inc_base)

            meta = getattr(n, 'meta', {}) or {}
            yield0 = float(meta.get('yield0', 0.0) or 0.0)
            grassland_coef = float(meta.get('grassland_coef', 0.0) or 0.0)

            key_country = str(getattr(n, 'country', '')).strip()
            p0_val = getattr(n, 'P0', 0.0) or 0.0
            if getattr(n, 'year', None) > hist_end_year:
                base_p0 = base_p0_by_country_comm.get((key_country, getattr(n, 'commodity', None)))
                if base_p0 is not None:
                    p0_val = base_p0
            try:
                p0_val = float(p0_val)
            except Exception:
                p0_val = 0.0
            if not np.isfinite(p0_val) or p0_val <= 0:
                base_p0 = base_p0_by_country_comm.get((key_country, getattr(n, 'commodity', None)))
                if base_p0 is not None:
                    p0_val = base_p0
                else:
                    p0_val = 1.0

            records.append({
                'region': str(n.country).strip(),
                'commodity': n.commodity,
                'year': n.year,
                'Q0': getattr(n, 'Q0', 0.0) or 0.0,
                'D0': getattr(n, 'D0', 0.0) or 0.0,
                'P0': p0_val,
                'yield0': yield0,
                'grassland_coef': grassland_coef,
                'livestock_feed_account': bool(meta.get('livestock_feed_account', False)),
                'feed_only_entry': bool(meta.get('feed_only_entry', False)),
                'livestock_background_only': bool(meta.get('livestock_background_only', False)),
                'livestock_feed_unidentified': bool(meta.get('livestock_feed_unidentified', False)),
                'seed_coefficient': float(meta.get('seed_coefficient', 0.0)),
                'seed_reference_t': float(meta.get('seed_reference_t', 0.0)),
                'seed_coefficient_unidentified': bool(meta.get('seed_coefficient_unidentified', False)),
                'eps_supply': getattr(n, 'eps_supply', 0.0) or 0.0,
                'eps_supply_yield': getattr(n, 'eps_supply_yield', 0.0) or 0.0,
                'eps_supply_temp': getattr(n, 'eps_supply_temp', 0.0) or 0.0,
                'Ymult': getattr(n, 'Ymult', 1.0) or 1.0,
                'Tmult': getattr(n, 'Tmult', 1.0) or 1.0,
                'eps_demand': getattr(n, 'eps_demand', 0.0) or 0.0,
                'eps_pop_demand': getattr(n, 'eps_pop_demand', 0.0) or 0.0,
                'eps_income_demand': getattr(n, 'eps_income_demand', 0.0) or 0.0,
                'epsS_row': dict(getattr(n, 'epsS_row', {}) or {}),
                'epsD_row': dict(getattr(n, 'epsD_row', {}) or {}),
                'pop_base': pop_base,
                'pop_t': pop_t,
                'inc_base': inc_base,
                'inc_t': inc_t,
            })
        regional_df = pd.DataFrame(records)

    if regional_df.empty:
        return {}

    regional_df = regional_df[
        regional_df['commodity'].astype(str).str.strip().str.lower().isin(exclude_norm)
    ]
    if regional_df.empty:
        logger.info("[LINEAR] No excluded commodities found in data for separate prediction")
        return {}

    regions = regional_df['region'].unique().tolist()
    commodities = sorted(regional_df['commodity'].unique().tolist())
    idx: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for _, row in regional_df.iterrows():
        key = (row['region'], row['commodity'], row['year'])
        idx[key] = row.to_dict()

    method = str(demand_method).lower()
    strict_nutrition = (method == 'nutrition')
    nutrition_demand_map: Dict[Tuple[str, str, int], float] = {}
    nonfood_commodities: set = set()
    country_by_m49: Dict[str, str] = {}
    if waste_reduction_by or losses_ratio_by:
        country_by_m49 = _load_m49_to_country(dict_v3_path)
    if method in {'nutrition', 'nutrition_band', 'nutrition_anchor'}:
        if not population_by_country_year:
            raise ValueError("[LINEAR] nutrition demand requires population_by_country_year")
        if not nutrition_profile_xlsx:
            try:
                from config_paths import get_input_base
                nutrition_profile_xlsx = str(Path(get_input_base()) / 'Driver' / 'Nutrition' / 'Nutrition_profile_rescaled.xlsx')
            except Exception:
                nutrition_profile_xlsx = None
        nutrition_demand_map = build_nutrition_demand_map(
            nutrition_xlsx=nutrition_profile_xlsx or '',
            dict_v3_path=dict_v3_path,
            indicator=nutrition_indicator,
            years=years,
            population_by_country_year=population_by_country_year,
            nutrition_profile_sheet=nutrition_profile_sheet,
            use_regional_agg=_USE_REGIONAL_AGGREGATION,
            hist_end_year=hist_end_year,
            use_baseyear_for_future=nutrition_use_baseyear_for_future,
            waste_reduction_by_country_comm_year=waste_reduction_by,
            losses_ratio_by_country_comm_year=losses_ratio_by,
            country_by_m49=country_by_m49 if country_by_m49 else None,
        )
        nonfood_commodities = _load_nonfood_commodities(dict_v3_path)
        if not nutrition_demand_map:
            msg = f"[LINEAR] nutrition demand empty; demand_method={method}"
            if strict_nutrition:
                raise ValueError(msg)
            logger.warning(msg)

    loss_ratio_by_item: Dict[Tuple[str, str], float] = {}
    demand_item_map: Dict[str, List[str]] = {}
    normalize_comp_item = None
    region_to_m49: Dict[str, str] = {v: k for k, v in country_by_m49.items() if v} if country_by_m49 else {}
    if method == 'elasticity' and (waste_reduction_by or losses_ratio_by):
        try:
            from config_paths import get_input_base
            comp_path = str(Path(get_input_base()) / 'Production_Trade' / 'Demand_composition.xlsx')
            loss_ratio_by_item = _load_demand_composition_losses_ratio(comp_path)
            _, demand_item_map = _load_item_demand_extra_and_map(dict_v3_path)
            normalize_comp_item = _normalize_comp_item_name
        except Exception as exc:
            logger.warning("[LINEAR] 读取Losses比例失败: %s", exc)
            loss_ratio_by_item = {}
            demand_item_map = {}

    use_relative_price = bool(use_relative_price)
    price_bounds_by_comm, price_ref_by_comm, (Pmin, Pmax), _ = _compute_price_bounds_by_comm(
        idx=idx,
        commodities=commodities,
        regions=regions,
        hist_end_year=hist_end_year,
        price_bounds=price_bounds,
        use_relative_price=use_relative_price,
        relative_price_bounds=relative_price_bounds,
        price_bounds_mode=price_bounds_mode,
        price_bounds_p0_mult=price_bounds_p0_mult,
    )

    def _p_bounds(comm: str) -> Tuple[float, float]:
        return price_bounds_by_comm.get(comm, (Pmin, Pmax))

    Pc: Dict[Tuple[str, int], float] = {}
    Pc_rel: Dict[Tuple[str, int], float] = {}
    Qs: Dict[Tuple[str, str, int], float] = {}
    Qd: Dict[Tuple[str, str, int], float] = {}
    neg_qs: List[Tuple[str, str, int, float]] = []
    neg_qd: List[Tuple[str, str, int, float]] = []

    for j in commodities:
        ref = float(price_ref_by_comm.get(j, 1.0) or 1.0)
        if not np.isfinite(ref) or ref <= 0:
            ref = 1.0
        for t in years:
            region_keys = [(r, j, t) for r in regions if (r, j, t) in idx]
            if not region_keys:
                continue

            if t <= hist_end_year:
                for r, jj, tt in region_keys:
                    data = idx[(r, jj, tt)]
                    Qs[(r, jj, tt)] = float(data.get('Q0', 0.0) or 0.0)
                    Qd[(r, jj, tt)] = float(data.get('D0', 0.0) or 0.0)
                if use_relative_price:
                    Pc_rel[(j, t)] = 1.0
                    Pc[(j, t)] = ref
                else:
                    Pc[(j, t)] = ref
                continue

            sum_a_s = 0.0
            sum_b_s = 0.0
            sum_a_d = 0.0
            sum_b_d = 0.0
            region_params: List[Tuple[str, float, float, float, float]] = []

            for r, jj, tt in region_keys:
                data = idx[(r, jj, tt)]
                base_key = (r, jj, hist_end_year)
                if base_key in idx:
                    Q0_base = max(1e-6, idx[base_key].get('Q0', 1e-6))
                    D0_base = max(1e-6, idx[base_key].get('D0', 1e-6))
                    P0_base = max(1e-6, idx[base_key].get('P0', 1.0))
                else:
                    Q0_base = max(1e-6, data.get('Q0', 1e-6))
                    D0_base = max(1e-6, data.get('D0', 1e-6))
                    P0_base = max(1e-6, data.get('P0', 1.0))

                P0 = max(1e-6, P0_base)
                eps_s = data.get('eps_supply', 0.0) or 0.0
                eta_y = data.get('eps_supply_yield', 0.0) or 0.0
                eta_temp = data.get('eps_supply_temp', 0.0) or 0.0
                Ymult = data.get('Ymult', 1.0) or 1.0
                Tmult = data.get('Tmult', 1.0) or 1.0
                yield_adj = eta_y * (Ymult - 1.0)
                temp_adj = eta_temp * (Tmult - 1.0)
                a_s = Q0_base * (1.0 + yield_adj + temp_adj - eps_s)
                b_s_abs = Q0_base * eps_s / P0
                b_s = b_s_abs * ref if use_relative_price else b_s_abs
                tau = 0.0
                if tax_unit_adder:
                    tau = float(tax_unit_adder.get((r, jj, tt), 0.0) or 0.0)
                    a_s = a_s - b_s_abs * tau

                eps_d = data.get('eps_demand', 0.0) or 0.0
                eps_pop = data.get('eps_pop_demand', 0.0) or 0.0
                eps_inc = data.get('eps_income_demand', 0.0) or 0.0
                profile_demand = None
                if method in {'nutrition', 'nutrition_band', 'nutrition_anchor'} and j not in nonfood_commodities:
                    profile_demand = nutrition_demand_map.get((r, jj, tt))
                    if profile_demand is None and strict_nutrition:
                        raise ValueError(f"[LINEAR] nutrition demand missing for {r},{jj},{tt}")
                if method == 'nutrition' and j not in nonfood_commodities:
                    if profile_demand is None:
                        raise ValueError(f"[LINEAR] nutrition demand missing for {r},{jj},{tt}")
                    a_d = float(profile_demand)
                    b_d = 0.0
                else:
                    pop_base = max(1e-6, data.get('pop_base', 1.0) or 1.0)
                    pop_t = max(1e-6, data.get('pop_t', pop_base) or pop_base)
                    inc_base = max(1e-6, data.get('inc_base', 1.0) or 1.0)
                    inc_t = max(1e-6, data.get('inc_t', inc_base) or inc_base)
                    pop_ratio = pop_t / pop_base
                    inc_ratio = inc_t / inc_base
                    pop_effect = pop_ratio ** eps_pop if eps_pop != 0 else 1.0
                    inc_effect = inc_ratio ** eps_inc if eps_inc != 0 else 1.0
                    nutrition_anchor_uses_profile = False
                    if method in {'nutrition_anchor', 'nutrition_band'} and j not in nonfood_commodities and profile_demand is not None:
                        try:
                            profile_val = float(profile_demand)
                        except Exception:
                            profile_val = None
                        if profile_val is not None and np.isfinite(profile_val) and profile_val >= 0:
                            D0_base = max(1e-6, profile_val)
                            nutrition_anchor_uses_profile = True
                    if nutrition_anchor_uses_profile:
                        pop_effect = 1.0
                        inc_effect = 1.0
                    D0_adjusted = D0_base * pop_effect * inc_effect
                    if feed_reduction_by:
                        rate = float(feed_reduction_by.get((r, jj, tt), 0.0) or 0.0)
                        rate = max(-1.0, min(1.0, rate))
                        D0_adjusted = D0_adjusted * (1.0 + rate)
                    if method == 'elasticity' and (waste_reduction_by or losses_ratio_by) and loss_ratio_by_item and demand_item_map and normalize_comp_item:
                        m49_code = _norm_m49_code(r)
                        if not m49_code and region_to_m49:
                            m49_code = region_to_m49.get(str(r).strip())
                        if m49_code:
                            items = demand_item_map.get(jj, [])
                            ratios = []
                            for item in items:
                                item_norm = normalize_comp_item(item)
                                val = loss_ratio_by_item.get((m49_code, item_norm))
                                if val is not None and np.isfinite(val):
                                    ratios.append(float(val))
                            if items:
                                loss_ratio = float(sum(ratios) / len(ratios)) if ratios else 0.0
                                loss_delta, _ = _lookup_loss_delta(
                                    key=(r, jj, tt),
                                    all_key=(r, 'All', tt),
                                    waste_reduction_by=waste_reduction_by,
                                    losses_ratio_by=losses_ratio_by,
                                )
                                if loss_delta is not None:
                                    mult = _loss_multiplier_from_delta(loss_ratio, loss_delta)
                                    D0_adjusted *= mult
                    a_d = D0_adjusted * (1.0 - eps_d)
                    b_d_abs = D0_adjusted * eps_d / P0
                    b_d = b_d_abs * ref if use_relative_price else b_d_abs

                sum_a_s += a_s
                sum_b_s += b_s
                sum_a_d += a_d
                sum_b_d += b_d
                region_params.append((r, a_s, b_s, a_d, b_d))

            if not region_params:
                continue

            denom = sum_b_s - sum_b_d
            if abs(denom) < 1e-12:
                pc_var = 1.0 if use_relative_price else ref
            else:
                pc_var = (sum_a_d - sum_a_s) / denom
            if not np.isfinite(pc_var):
                pc_var = 1.0 if use_relative_price else ref
            pmin_j, pmax_j = _p_bounds(j)
            if pc_var < pmin_j:
                pc_var = pmin_j
            elif pc_var > pmax_j:
                pc_var = pmax_j

            if use_relative_price:
                Pc_rel[(j, t)] = pc_var
                Pc[(j, t)] = pc_var * ref
            else:
                Pc[(j, t)] = pc_var

            for r, a_s, b_s, a_d, b_d in region_params:
                qs_val = a_s + b_s * pc_var
                qd_val = a_d + b_d * pc_var
                if qs_val < 0:
                    neg_qs.append((r, j, t, qs_val))
                    qs_val = 0.0
                if qd_val < 0:
                    neg_qd.append((r, j, t, qd_val))
                    qd_val = 0.0
                Qs[(r, j, t)] = qs_val
                Qd[(r, j, t)] = qd_val

    if neg_qs:
        sample = ", ".join(f"{r}:{j}:{t}:{v:.2e}" for r, j, t, v in neg_qs[:5])
        logger.warning(f"[LINEAR] excluded Qs negative -> clamp to 0; sample={sample}")
    if neg_qd:
        sample = ", ".join(f"{r}:{j}:{t}:{v:.2e}" for r, j, t, v in neg_qd[:5])
        logger.warning(f"[LINEAR] excluded Qd negative -> clamp to 0; sample={sample}")

    result: Dict[str, Any] = {'Pc': Pc, 'Qs': Qs, 'Qd': Qd}
    if use_relative_price:
        result['Pc_rel'] = Pc_rel
    return result


def _compute_max_violation(model: gp.Model,
                           top_n: int = 20) -> Tuple[float, List[Dict[str, Any]]]:
    max_violation = 0.0
    violations: List[Dict[str, Any]] = []

    for c in model.getConstrs():
        sense = c.Sense
        rhs = float(c.RHS)
        slack = float(c.Slack)
        if sense == '=':
            violation = abs(slack)
            lhs = rhs - slack
        elif sense == '<':
            violation = max(-slack, 0.0)
            lhs = rhs - slack
        elif sense == '>':
            violation = max(slack, 0.0)
            lhs = rhs - slack
        else:
            continue
        if violation > max_violation:
            max_violation = violation
        if violation > 0:
            violations.append({
                'name': c.ConstrName,
                'type': 'constr',
                'sense': sense,
                'lhs': lhs,
                'rhs': rhs,
                'violation': violation,
            })

    inf = gp.GRB.INFINITY
    for v in model.getVars():
        x = float(v.X)
        if v.LB > -inf:
            violation = max(float(v.LB) - x, 0.0)
            if violation > max_violation:
                max_violation = violation
            if violation > 0:
                violations.append({
                    'name': f"LB[{v.VarName}]",
                    'type': 'bound',
                    'sense': '>=',
                    'lhs': x,
                    'rhs': float(v.LB),
                    'violation': violation,
                })
        if v.UB < inf:
            violation = max(x - float(v.UB), 0.0)
            if violation > max_violation:
                max_violation = violation
            if violation > 0:
                violations.append({
                    'name': f"UB[{v.VarName}]",
                    'type': 'bound',
                    'sense': '<=',
                    'lhs': x,
                    'rhs': float(v.UB),
                    'violation': violation,
                })

    violations.sort(key=lambda x: x['violation'], reverse=True)
    if top_n > 0:
        violations = violations[:top_n]
    return max_violation, violations


def _write_violation_iis(model: gp.Model,
                         output_dir: Optional[str],
                         violation_tol: float) -> Optional[Path]:
    outdir = Path(output_dir) if output_dir else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)
    iis_path = outdir / "linear_model_violation_iis.ilp"
    fixed = model.copy()
    fixed.setParam('OutputFlag', 0)

    sol = {v.VarName: float(v.X) for v in model.getVars()}
    for v in fixed.getVars():
        val = sol.get(v.VarName)
        if val is None:
            continue
        v.LB = val
        v.UB = val
    fixed.update()

    try:
        feas_tol = min(max(1e-9, float(violation_tol)), 1e-6)
        fixed.setParam('FeasibilityTol', feas_tol)
    except Exception:
        pass

    try:
        fixed.computeIIS()
        fixed.write(str(iis_path))
        return iis_path
    except Exception:
        try:
            fixed.write(str(outdir / "linear_model_violation_fixed.lp"))
        except Exception:
            pass
        return None


def _future_years_after_hist(years: List[int], hist_end_year: int) -> List[int]:
    return sorted({int(t) for t in (years or []) if int(t) > int(hist_end_year)})


def _normalize_linear_objective_strategy(objective_strategy: Optional[str],
                                         future_years: List[int]) -> str:
    strategy = str(objective_strategy or 'auto').strip().lower()
    if strategy in {'', 'auto', 'default', 'none'}:
        return 'lexicographic_slack' if len(future_years) > 1 else 'single_stage'
    if strategy in {
        'lexicographic', 'lexicographic_slack', 'two_stage', 'two-stage',
        'hierarchical', 'multiobjective', 'multi-objective'
    }:
        return 'lexicographic_slack'
    return 'single_stage'


def _optimize_model_with_retries(model: gp.Model,
                                 logger: logging.Logger,
                                 *,
                                 phase_label: Optional[str] = None) -> int:
    phase_txt = f" ({phase_label})" if phase_label else ""
    def _retry_numeric_status(status: int) -> int:
        if status != gp.GRB.NUMERIC:
            return status
        logger.warning(f"[LINEAR] status=NUMERIC{phase_txt}; retry with Method=1 (dual simplex)...")
        try:
            model.setParam('Method', 1)
            model.optimize()
            status = int(model.Status)
            logger.info(f"[LINEAR] Method=1 retry done{phase_txt}, status={status}")
        except Exception as exc:
            logger.warning(f"[LINEAR] Method=1 retry failed{phase_txt}: {exc}")
        if status == gp.GRB.NUMERIC:
            logger.warning(f"[LINEAR] status=NUMERIC{phase_txt}; retry with BarHomogeneous=1...")
            try:
                model.setParam('Method', 2)
                model.setParam('BarHomogeneous', 1)
                model.optimize()
                status = int(model.Status)
                logger.info(f"[LINEAR] BarHomogeneous retry done{phase_txt}, status={status}")
            except Exception as exc:
                logger.warning(f"[LINEAR] BarHomogeneous retry failed{phase_txt}: {exc}")
        return status

    logger.info(f"[LINEAR] 开始求解{phase_txt}（无时限）...")
    cache = getattr(model, '_nzf_cache', {}) or {}
    projected_ces = prepare_quantity_projection(model, cache)
    model.optimize()

    status = int(model.Status)
    status = _retry_numeric_status(status)

    if status == gp.GRB.INF_OR_UNBD:
        logger.warning(f"[LINEAR] 状态=INF_OR_UNBD{phase_txt}，尝试关闭DualReductions重新求解以区分不可行/无界...")
        try:
            model.setParam('DualReductions', 0)
            model.setParam('InfUnbdInfo', 1)
            model.optimize()
            status = int(model.Status)
            logger.info(f"[LINEAR] 重新求解完成{phase_txt}，状态={status}")
        except Exception as exc:
            logger.warning(f"[LINEAR] 重新求解失败{phase_txt}: {exc}")
        status = _retry_numeric_status(status)

    if status == gp.GRB.INFEASIBLE:
        logger.warning('[LINEAR] verify presolved infeasibility using primal simplex with Presolve=0')
        model.setParam('Presolve',0)
        model.setParam('Method',0)
        model.optimize()
        status = int(model.Status)

    ces_state = (getattr(model, '_nzf_cache', {}) or {}).get('armington_ces', {})
    if projected_ces and status == gp.GRB.OPTIMAL:
        residual = recover_projected_ces(ces_state)
        logger.info('[CES] analytic elimination exact residual=%.6g', residual)
    if ces_state.get('records') and not projected_ces and status == gp.GRB.OPTIMAL:
        ces_state['converged'] = False
        for iteration in range(160):
            residual = update_ces_approximation(model, ces_state, damping=1.0 if iteration > 3 else 0.5)
            logger.info('[CES] iteration=%d nonlinear_relative_residual=%.6g', iteration, residual)
            if ces_state['converged']:
                break
            model.optimize()
            status = _retry_numeric_status(int(model.Status))
            if status == gp.GRB.INFEASIBLE:
                logger.warning('[CES] cross-check inner infeasibility with Presolve=0, Method=0')
                model.setParam('Presolve',0); model.setParam('Method',0)
                model.optimize(); status=int(model.Status)
            if status != gp.GRB.OPTIMAL:
                break
        if not ces_state['converged']:
            raise RuntimeError(f'Armington CES did not converge: inner_status={status}, residual={residual}')
    logger.info(f"[LINEAR] 求解完成{phase_txt}，状态={status}")
    return status


def _clear_land_source_selection(model: gp.Model) -> None:
    """Remove previous sample's selection constraints before a model update."""
    cache = getattr(model, '_nzf_cache', {}) or {}
    constraints = cache.pop('land_source_selection_constraints', [])
    if constraints:
        model.remove(constraints)
    bounds = cache.pop('land_source_selection_bounds', [])
    for var, lower, upper in bounds:
        var.LB = lower
        var.UB = upper
    if constraints or bounds:
        model.update()
    cache.pop('land_source_selection_audit', None)


def _select_land_sources_on_optimal_face(model: gp.Model,
                                         logger: logging.Logger) -> Dict[str, Any]:
    """Prefer nonforest, pasture, then forest without trading off primary cost.

    Lock the current primary objective and production/demand, then minimize
    forest conversion followed by pasture-to-cropland. Keep these constraints
    for extraction; cached-sample updates remove them before the next solve.
    This policy selects sources, not a calibrated global land-use prediction.
    """
    cache = getattr(model, '_nzf_cache', {}) or {}
    if not cache.get('land_source_selection_enabled'):
        return {}
    if model.Status != gp.GRB.OPTIMAL or model.SolCount < 1:
        return {}
    if cache.get('land_source_selection_constraints'):
        raise RuntimeError("Clear the previous land-source selection before re-solving a changed model")
    forest_vars = list((cache.get('forest_to_cropland') or {}).values()) + list((cache.get('forest_to_grassland') or {}).values())
    pasture_vars = list((cache.get('grassland_to_cropland') or {}).values())
    if not forest_vars and not pasture_vars:
        return {}
    primary = model.getObjective()
    primary_sense = model.ModelSense
    primary_optimum = float(model.ObjVal)
    scale = float(cache.get('land_scale', 1.0) or 1.0)
    activity = [(group, key, var, float(var.X)) for group in ('Qs', 'Qd', 'Pc', 'Pw') for key, var in (cache.get(group) or {}).items()]
    # Complementary slackness describes the optimal LP face without adding
    # the primary objective (whose penalties can reach 1e12) as a matrix row.
    # Preserve every nonzero reduced-cost variable and every priced inequality.
    # Snapshot dual attributes before adding constraints invalidates the solve.
    priced_vars = [(var, float(var.X)) for var in model.getVars() if var.RC != 0.0]
    priced_rows = [(model.getRow(con), float(con.RHS)) for con in model.getConstrs()
                   if con.Sense != '=' and con.Pi != 0.0]
    primary_terms = [(primary.getVar(i), primary.getCoeff(i), float(primary.getVar(i).X))
                     for i in range(primary.size())]
    fixed = [model.addConstr(row == rhs, name=f'land_select_dual_row_{i}')
             for i, (row, rhs) in enumerate(priced_rows)]
    pins = {var.index: (var, value) for var, value in priced_vars}
    pins.update({var.index: (var, value) for _, _, var, value in activity})
    fixed.extend(model.addConstr(var == value, name=f'land_select_pin_{i}')
                 for i, (var, value) in pins.items())
    cache['land_source_selection_constraints'] = fixed
    # With activity pinned, each balance equality implies a finite net-import
    # interval. Propagate this redundant bound to avoid subtracting the default
    # +/-1e12 bounds in simplex. Keep every original cap and restore bounds when
    # clearing the selection, including failure and cached-sample updates.
    tightened_bounds = []
    cache['land_source_selection_bounds'] = tightened_bounds
    for key, var in (cache.get('net_import') or {}).items():
        con = (cache.get('regional_balance_constr') or {}).get(key)
        if con is None or con.Sense != '=':
            continue
        row = model.getRow(con)
        coefficient = 0.0
        terms = [abs(float(con.RHS))]
        for i in range(row.size()):
            other = row.getVar(i)
            if other.index == var.index:
                coefficient += row.getCoeff(i)
            elif other.index in pins:
                terms.append(abs(row.getCoeff(i) * pins[other.index][1]))
            else:
                break
        else:
            if coefficient == 0.0:
                continue
            limit = math.fsum(terms) / abs(coefficient) + 1.0
            lower, upper = float(var.LB), float(var.UB)
            new_lower, new_upper = max(lower, -limit), min(upper, limit)
            if math.isfinite(limit) and new_lower <= new_upper and (new_lower > lower or new_upper < upper):
                tightened_bounds.append((var, lower, upper))
                var.LB, var.UB = new_lower, new_upper
    audit: Dict[str, Any] = {'policy': 'primary_then_fixed_activity_then_forest_then_pasture',
                            'primary_optimum': primary_optimum, 'fixed_activity_count': len(activity),
                            'optimal_face_method': 'complementary_slackness',
                            'fixed_reduced_cost_variables': len(priced_vars),
                            'fixed_priced_inequalities': len(priced_rows),
                            'redundant_trade_bounds_tightened': len(tightened_bounds),
                            'pin_tolerance_model_units': 0.0, 'stages': [], 'solver_attempts': []}
    def _selection_failed(message: str) -> None:
        model.setObjective(primary, primary_sense)
        _clear_land_source_selection(model)
        raise RuntimeError(message)

    def _optimize_selection(phase: str) -> int:
        # Eliminating exact pins at global scale can create a false infeasible
        # presolved LP through cancellation in the nutrition equations. Retain
        # the original rows and restore the caller's presolve setting afterward.
        previous_presolve = model.Params.Presolve
        previous_method = model.Params.Method
        try:
            model.Params.Presolve = 0
            model.Params.Method = 1
            model.reset()
            status = _optimize_model_with_retries(model, logger, phase_label=phase)
            audit['solver_attempts'].append({'phase': phase, 'requested_method': 1, 'status': status})
            if status in {gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD, gp.GRB.NUMERIC}:
                # Each level retains a solution from the preceding level in
                # exact arithmetic. Check a numerical status with a different
                # simplex path before failing; never relax the constraints.
                logger.warning('[LINEAR][LAND-SOURCE] %s status=%s; retry unchanged LP with primal simplex', phase, status)
                model.Params.Method = 0
                model.reset()
                status = _optimize_model_with_retries(model, logger, phase_label=phase + '-primal-retry')
                audit['solver_attempts'].append({'phase': phase, 'requested_method': 0, 'status': status})
            return status
        except Exception:
            model.setObjective(primary, primary_sense)
            _clear_land_source_selection(model)
            raise
        finally:
            model.Params.Presolve = previous_presolve
            model.Params.Method = previous_method
    for label, variables in [('forest', forest_vars), ('pasture_to_crop', pasture_vars)]:
        if not variables:
            continue
        objective = gp.quicksum(variables)
        model.setObjective(objective, gp.GRB.MINIMIZE)
        status = _optimize_selection('land-source-' + label)
        if status != gp.GRB.OPTIMAL:
            _selection_failed(f"Land-source selection {label} failed: status={status}")
        optimum = float(objective.getValue())
        fixed.append(model.addConstr(objective == optimum, name='land_select_' + label))
        audit['stages'].append({'stage': label, 'area_ha': optimum * scale, 'status': status})
    # Restore the production objective so ObjVal retains its public meaning.
    model.setObjective(primary, primary_sense)
    status = _optimize_selection('land-source-restore-primary')
    if status != gp.GRB.OPTIMAL:
        _selection_failed(f"Land-source final solve failed: status={status}")
    audit.update({'status': status, 'primary_after': float(primary.getValue()),
                  'primary_delta': float(primary.getValue()) - primary_optimum,
                  'primary_termwise_delta': float(math.fsum(coef * (var.X - before) for var, coef, before in primary_terms)),
                  'max_activity_delta_scaled': max((abs(var.X - value) for _, _, var, value in activity), default=0.0),
                  'constraint_violation': float(model.ConstrVio), 'bound_violation': float(model.BoundVio)})
    tolerance = max(1e-5, abs(primary_optimum) * 1e-9)
    if max(abs(audit['primary_delta']), abs(audit['primary_termwise_delta'])) > tolerance or audit['max_activity_delta_scaled'] > 1e-5:
        _selection_failed(f"Land-source selection failed to preserve the primary solution: {audit}")
    cache['land_source_selection_audit'] = audit
    logger.info('[LINEAR][LAND-SOURCE] %s', audit)
    return audit


def _apply_country_result_as_next_baseline(nodes: List[Any],
                                           country_result: Dict[str, Any],
                                           target_year: int) -> int:
    pc_map = country_result.get('Pc', {}) or {}
    qs_map = country_result.get('Qs', {}) or {}
    qd_map = country_result.get('Qd', {}) or {}
    updated = 0
    target_year = int(target_year)

    for node in nodes:
        if int(getattr(node, 'year', -1)) != target_year:
            continue
        key = (node.country, node.commodity, node.year)
        touched = False
        if key in pc_map:
            val = float(pc_map[key])
            node.P = val
            node.P0 = val
            touched = True
        if key in qs_map:
            val = float(qs_map[key])
            node.Q = val
            node.Q0 = val
            touched = True
        if key in qd_map:
            val = float(qd_map[key])
            node.D = val
            node.D0 = val
            touched = True
        if touched:
            updated += 1
    return updated


def _scale_hist_max_for_rolling(hist_max_production: Optional[Dict[Tuple[str, str], float]],
                                *,
                                base_hist_end_year: int,
                                step_hist_end_year: int,
                                max_growth_rate_per_period: Optional[float]) -> Optional[Dict[Tuple[str, str], float]]:
    if not hist_max_production:
        return hist_max_production
    if int(step_hist_end_year) <= int(base_hist_end_year):
        return hist_max_production
    growth = float(max_growth_rate_per_period or 0.06)
    years_ahead = int(step_hist_end_year) - int(base_hist_end_year)
    factor = (1.0 + growth) ** years_ahead
    scaled: Dict[Tuple[str, str], float] = {}
    for key, val in hist_max_production.items():
        try:
            scaled[key] = float(val) * factor
        except Exception:
            continue
    return scaled


def _compute_step_land_state(result: Dict[str, Any],
                             *,
                             target_year: int,
                             hist_end_year: int,
                             land_carbon_price_by_year: Optional[Dict[int, float]],
                             grassland_method: str,
                             grass_area_by_region_year: Optional[Dict[Tuple[str, int], float]]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    logger = logging.getLogger(__name__)
    direct_crop = result.get('cropland_actual_by_region_year', {}) or {}
    direct_grass = result.get('grassland_actual_by_region_year', {}) or {}
    direct_forest = result.get('forest_actual_by_region_year', {}) or {}
    if direct_crop or direct_grass or direct_forest:
        crop_state = {
            r: float(val)
            for (r, t), val in direct_crop.items()
            if int(t) == int(target_year)
        }
        grass_state = {
            r: float(val)
            for (r, t), val in direct_grass.items()
            if int(t) == int(target_year)
        }
        forest_state = {
            r: float(val)
            for (r, t), val in direct_forest.items()
            if int(t) == int(target_year)
        }
        if crop_state or grass_state or forest_state:
            return crop_state, grass_state, forest_state

    idx = result.get('idx', {}) or {}
    qs = result.get('Qs', {}) or {}
    base_cropland_area = result.get('base_cropland_area', {}) or {}
    base_grassland_area = result.get('base_grassland_area', {}) or {}
    base_forest_area = result.get('base_forest_area', {}) or {}
    crop_to_grass_map = result.get('cropland_to_grassland', {}) or {}
    crop_to_forest_map = result.get('cropland_to_forest', {}) or {}
    crop_to_othernat_map = result.get('cropland_to_othernat', {}) or {}
    grass_to_crop_map = result.get('grassland_to_cropland', {}) or {}
    grass_to_forest_map = result.get('grassland_to_forest', {}) or {}
    grass_to_othernat_map = result.get('grassland_to_othernat', {}) or {}
    forest_to_crop_map = result.get('forest_to_cropland', {}) or {}
    forest_to_grass_map = result.get('forest_to_grassland', {}) or {}
    bg_forest_to_crop_map = result.get('background_forest_to_cropland', {}) or {}
    bg_forest_to_grass_map = result.get('background_forest_to_grassland', {}) or {}
    bg_crop_to_forest_map = result.get('background_cropland_to_forest', {}) or {}
    bg_grass_to_forest_map = result.get('background_grassland_to_forest', {}) or {}

    target_year = int(target_year)
    hist_end_year = int(hist_end_year)

    crop_demand = defaultdict(float)
    grass_demand = defaultdict(float)
    base_crop_demand = defaultdict(float)
    base_grass_demand = defaultdict(float)

    for (r, j, t), node_data in idx.items():
        t_int = int(t)
        if t_int == hist_end_year:
            q0_val = float(node_data.get('Q0', 0.0) or 0.0)
            if q0_val > 0:
                yield_j = _cropland_yield0(
                    node_data,
                    region=r,
                    commodity=j,
                    year=t_int,
                    context="rolling_base_land_state",
                )
                base_crop_demand[r] += q0_val / yield_j
                coef_grass = float(node_data.get('grassland_coef', 0.0) or 0.0)
                if coef_grass > 0:
                    base_grass_demand[r] += coef_grass * q0_val

        if t_int != target_year:
            continue
        qs_val = float(qs.get((r, j, t), 0.0) or 0.0)
        if qs_val <= 0:
            continue
        yield_j = _cropland_yield0(
            node_data,
            region=r,
            commodity=j,
            year=t_int,
            context="rolling_target_land_state",
        )
        crop_demand[r] += qs_val / yield_j
        coef_grass = float(node_data.get('grassland_coef', 0.0) or 0.0)
        if coef_grass > 0:
            grass_demand[r] += coef_grass * qs_val

    if grassland_method == 'static' and grass_area_by_region_year:
        for (r, t), val in grass_area_by_region_year.items():
            t_int = int(t)
            if t_int == hist_end_year:
                base_grass_demand[r] = float(val or 0.0)
            elif t_int == target_year:
                grass_demand[r] = float(val or 0.0)

    intens_cfg = LUCConfig()
    intens_per_usd = float(intens_cfg.intensification_per_usd or 0.0)
    intens_cap = float(intens_cfg.intensification_cap or 0.0)
    land_price = 0.0
    if land_carbon_price_by_year:
        try:
            land_price = float(land_carbon_price_by_year.get(target_year, 0.0) or 0.0)
        except Exception:
            land_price = 0.0
    red = 0.0
    if land_price > 0 and intens_per_usd > 0 and intens_cap > 0:
        red = min(max(intens_per_usd * land_price, 0.0), intens_cap, 1.0)

    crop_area: Dict[str, float] = {}
    grass_area: Dict[str, float] = {}
    forest_area: Dict[str, float] = {}
    all_regions = (
        set(base_cropland_area.keys()) | set(base_grassland_area.keys()) | set(base_forest_area.keys())
        | set(crop_demand.keys()) | set(grass_demand.keys())
        | {r for (r, t) in crop_to_grass_map.keys() if int(t) == target_year}
        | {r for (r, t) in crop_to_forest_map.keys() if int(t) == target_year}
        | {r for (r, t) in crop_to_othernat_map.keys() if int(t) == target_year}
        | {r for (r, t) in grass_to_crop_map.keys() if int(t) == target_year}
        | {r for (r, t) in grass_to_forest_map.keys() if int(t) == target_year}
        | {r for (r, t) in grass_to_othernat_map.keys() if int(t) == target_year}
        | {r for (r, t) in forest_to_crop_map.keys() if int(t) == target_year}
        | {r for (r, t) in forest_to_grass_map.keys() if int(t) == target_year}
        | {r for (r, t) in bg_forest_to_crop_map.keys() if int(t) == target_year}
        | {r for (r, t) in bg_forest_to_grass_map.keys() if int(t) == target_year}
        | {r for (r, t) in bg_crop_to_forest_map.keys() if int(t) == target_year}
        | {r for (r, t) in bg_grass_to_forest_map.keys() if int(t) == target_year}
    )
    for r in all_regions:
        base_crop = float(base_cropland_area.get(r, 0.0) or 0.0)
        base_grass = float(base_grassland_area.get(r, 0.0) or 0.0)
        base_forest = float(base_forest_area.get(r, 0.0) or 0.0)

        crop_delta = float(crop_demand.get(r, 0.0) or 0.0) - float(base_crop_demand.get(r, 0.0) or 0.0)
        grass_delta = float(grass_demand.get(r, 0.0) or 0.0) - float(base_grass_demand.get(r, 0.0) or 0.0)
        if red > 0:
            crop_delta *= (1.0 - red)
            grass_delta *= (1.0 - red)

        crop_to_grass = float(crop_to_grass_map.get((r, target_year), 0.0) or 0.0)
        crop_to_forest = float(crop_to_forest_map.get((r, target_year), 0.0) or 0.0)
        crop_to_othernat = float(crop_to_othernat_map.get((r, target_year), 0.0) or 0.0)
        grass_to_crop = float(grass_to_crop_map.get((r, target_year), 0.0) or 0.0)
        grass_to_forest = float(grass_to_forest_map.get((r, target_year), 0.0) or 0.0)
        grass_to_othernat = float(grass_to_othernat_map.get((r, target_year), 0.0) or 0.0)
        forest_to_crop = float(forest_to_crop_map.get((r, target_year), 0.0) or 0.0)
        forest_to_grass = float(forest_to_grass_map.get((r, target_year), 0.0) or 0.0)
        bg_forest_to_crop = float(bg_forest_to_crop_map.get((r, target_year), 0.0) or 0.0)
        bg_forest_to_grass = float(bg_forest_to_grass_map.get((r, target_year), 0.0) or 0.0)
        bg_crop_to_forest = float(bg_crop_to_forest_map.get((r, target_year), 0.0) or 0.0)
        bg_grass_to_forest = float(bg_grass_to_forest_map.get((r, target_year), 0.0) or 0.0)

        # Rolling carry-over should use land stock conservation rather than the
        # step-local "actual" algebra. Otherwise outgoing conversions get
        # subtracted twice and the next base land stock can become nonsensical.
        crop_net = (
            grass_to_crop
            + forest_to_crop
            + bg_forest_to_crop
            - crop_to_grass
            - crop_to_forest
            - crop_to_othernat
            - bg_crop_to_forest
        )
        grass_net = (
            crop_to_grass
            + forest_to_grass
            + bg_forest_to_grass
            - grass_to_crop
            - grass_to_forest
            - grass_to_othernat
            - bg_grass_to_forest
        )
        forest_net = (
            crop_to_forest
            + grass_to_forest
            + bg_crop_to_forest
            + bg_grass_to_forest
            - forest_to_crop
            - forest_to_grass
            - bg_forest_to_crop
            - bg_forest_to_grass
        )

        crop_val = base_crop + crop_net
        grass_val = base_grass + grass_net
        forest_val = base_forest + forest_net

        # Keep the legacy demand-based calculation only for diagnostics.
        legacy_crop = base_crop + crop_delta - crop_to_grass - crop_to_forest
        legacy_grass = base_grass + grass_delta - grass_to_crop - grass_to_forest
        if (
            abs(crop_val - legacy_crop) > 1e4
            or abs(grass_val - legacy_grass) > 1e4
        ):
            logger.warning(
                "[LINEAR] rolling land carry uses stock update (region=%s year=%s): "
                "crop_stock=%.6e legacy_crop=%.6e grass_stock=%.6e legacy_grass=%.6e",
                r,
                target_year,
                crop_val,
                legacy_crop,
                grass_val,
                legacy_grass,
            )

        for label, val in (
            ('crop', crop_val),
            ('grass', grass_val),
            ('forest', forest_val),
        ):
            if val < -1e-3:
                logger.warning(
                    "[LINEAR] rolling land state negative after stock update: region=%s year=%s %s=%.6e "
                    "(base_crop=%.6e base_grass=%.6e base_forest=%.6e crop_net=%.6e grass_net=%.6e forest_net=%.6e)",
                    r,
                    target_year,
                    label,
                    val,
                    base_crop,
                    base_grass,
                    base_forest,
                    crop_net,
                    grass_net,
                    forest_net,
                )

        if -1e-6 < crop_val < 0:
            crop_val = 0.0
        if -1e-6 < grass_val < 0:
            grass_val = 0.0
        if -1e-6 < forest_val < 0:
            forest_val = 0.0

        crop_area[r] = crop_val
        grass_area[r] = grass_val
        forest_area[r] = forest_val

    return crop_area, grass_area, forest_area


def _build_regional_idx_from_nodes(nodes: List[Any],
                                   *,
                                   commodities: List[str],
                                   years: List[int],
                                   dict_v3_path: Optional[str],
                                   population_by_country_year: Optional[Dict[Tuple[str, int], float]],
                                   income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]],
                                   hist_end_year: int) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], List[str], List[int]]:
    regional_df = aggregate_nodes_to_regions(
        nodes,
        dict_v3_path=dict_v3_path,
        population_by_country_year=population_by_country_year,
        income_mult_by_country_year=income_mult_by_country_year,
        hist_end_year=hist_end_year,
    )
    if regional_df.empty:
        return {}, [], sorted({int(y) for y in years})
    keep_years = {int(y) for y in years}
    regional_df = regional_df[
        regional_df['commodity'].isin(commodities)
        & regional_df['year'].astype(int).isin(keep_years)
    ].copy()
    idx: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for _, row in regional_df.iterrows():
        idx[(row['region'], row['commodity'], int(row['year']))] = row.to_dict()
    regions = sorted({key[0] for key in idx.keys()})
    return idx, regions, sorted(keep_years)


def _solve_linear_regional_rolling(logger: logging.Logger,
                                   solver_kwargs: Dict[str, Any],
                                   *,
                                   objective_strategy: str) -> Dict[str, Any]:
    years = sorted({int(y) for y in (solver_kwargs.get('years') or [])})
    base_hist_end_year = int(solver_kwargs.get('hist_end_year', 2020) or 2020)
    future_years = _future_years_after_hist(years, base_hist_end_year)
    working_nodes = copy.deepcopy(solver_kwargs.get('nodes') or [])

    base_cropland_by_region = dict(solver_kwargs.get('base_cropland_by_region') or {})
    base_grassland_by_region = dict(solver_kwargs.get('base_grassland_by_region') or {})
    base_forest_by_region = dict(solver_kwargs.get('base_forest_by_region') or {})
    forest_area_by_region_year = dict(solver_kwargs.get('forest_area_by_region_year') or {})

    merge_year_pos = {
        'Pc': -1,
        'Pc_rel': -1,
        'Qs': -1,
        'Qd': -1,
        'supply_curtailment': -1,
        'net_import': -1,
        'excess': -1,
        'shortage': -1,
        'Eij': -1,
        'Cij': -1,
        'grassland_to_cropland': -1,
        'grassland_to_forest': -1,
        'cropland_to_grassland': -1,
        'cropland_to_forest': -1,
        'forest_to_cropland': -1,
        'forest_to_grassland': -1,
        'nonforest_to_cropland': -1,
        'nonforest_to_grassland': -1,
        'luc_qs_forest_cap_lhs': -1,
        'luc_qs_forest_cap_rhs': -1,
        'abatement': 2,
        'strategy_abatement': 1,
    }
    combined: Dict[str, Any] = {
        'status': gp.GRB.OPTIMAL,
        'model': None,
        'rolling_mode': True,
        'rolling_steps': [],
        'objective_strategy': objective_strategy,
    }
    for name in merge_year_pos:
        combined[name] = {}

    # Preserve the v2.0 cost audit trail across rolling steps. These mappings
    # use a different tuple layout from the standard solution blocks.
    rolling_cost_year_pos = {
        'abatement_costs': 2,
        'abatement_database_keys': 2,
        'strategy_abatement_costs': 1,
    }
    for name in rolling_cost_year_pos:
        combined[name] = {}

    prev_year = base_hist_end_year
    slack_total = 0.0
    objective_total = 0.0
    excluded_items: set = set()

    for target_year in future_years:
        step_years = sorted({int(prev_year), int(target_year)})
        step_nodes = [n for n in working_nodes if int(getattr(n, 'year', -1)) in step_years]
        step_kwargs = dict(solver_kwargs)
        step_kwargs['nodes'] = step_nodes
        step_kwargs['years'] = step_years
        step_kwargs['hist_end_year'] = int(prev_year)
        step_kwargs['hist_max_production'] = _scale_hist_max_for_rolling(
            solver_kwargs.get('hist_max_production'),
            base_hist_end_year=base_hist_end_year,
            step_hist_end_year=int(prev_year),
            max_growth_rate_per_period=solver_kwargs.get('max_growth_rate_per_period'),
        )
        step_kwargs['rolling_future_solve'] = False
        step_kwargs['objective_strategy'] = objective_strategy
        step_kwargs['base_cropland_by_region'] = dict(base_cropland_by_region) if base_cropland_by_region else None
        step_kwargs['base_grassland_by_region'] = dict(base_grassland_by_region) if base_grassland_by_region else None
        step_kwargs['base_forest_by_region'] = dict(base_forest_by_region) if base_forest_by_region else None
        step_kwargs['forest_area_by_region_year'] = dict(forest_area_by_region_year) if forest_area_by_region_year else None

        logger.info(
            "[LINEAR] rolling step %s -> %s (years=%s, objective=%s)",
            prev_year,
            target_year,
            step_years,
            objective_strategy,
        )
        step_result = solve_linear_regional(**step_kwargs)
        try:
            step_status = int(step_result.get('status', -1))
        except Exception:
            step_status = -1
        step_meta = {
            'hist_end_year': int(prev_year),
            'target_year': int(target_year),
            'status': step_status,
            'slack_objective': float(step_result.get('slack_objective', 0.0) or 0.0),
            'objective': float(step_result.get('objective', 0.0) or 0.0),
        }
        combined['rolling_steps'].append(step_meta)
        if step_status != gp.GRB.OPTIMAL:
            step_result['rolling_mode'] = True
            step_result['rolling_steps'] = combined['rolling_steps']
            step_result['rolling_failed_year'] = int(target_year)
            step_result['rolling_failed_hist_end_year'] = int(prev_year)
            return step_result

        slack_total += step_meta['slack_objective']
        objective_total += step_meta['objective']
        if step_result.get('excluded_commodities'):
            excluded_items.update(step_result.get('excluded_commodities') or [])

        for block, year_pos in merge_year_pos.items():
            src = step_result.get(block, {}) or {}
            if not isinstance(src, dict):
                continue
            for key, val in src.items():
                if not isinstance(key, tuple):
                    continue
                idx_pos = year_pos if year_pos >= 0 else len(key) + year_pos
                if idx_pos < 0 or idx_pos >= len(key):
                    continue
                try:
                    key_year = int(key[idx_pos])
                except Exception:
                    continue
                if key_year == int(target_year):
                    combined[block][key] = val

        for block, year_pos in rolling_cost_year_pos.items():
            src = step_result.get(block, {}) or {}
            if not isinstance(src, dict):
                continue
            for key, val in src.items():
                if not isinstance(key, tuple) or year_pos >= len(key):
                    continue
                try:
                    key_year = int(key[year_pos])
                except Exception:
                    continue
                if key_year == int(target_year):
                    combined[block][key] = val

        for metadata_name in ('strategy_cost_metadata', 'cost_database_metadata'):
            metadata_value = step_result.get(metadata_name)
            if isinstance(metadata_value, dict):
                combined[metadata_name] = dict(metadata_value)

        country_result = disaggregate_to_countries(working_nodes, step_result)
        apply_results_to_nodes(working_nodes, country_result)
        updated = _apply_country_result_as_next_baseline(working_nodes, country_result, int(target_year))
        logger.info(
            "[LINEAR] rolling step %s -> %s baseline advanced on %d country nodes",
            prev_year,
            target_year,
            updated,
        )

        try:
            crop_state, grass_state, forest_state = _compute_step_land_state(
                step_result,
                target_year=int(target_year),
                hist_end_year=int(prev_year),
                land_carbon_price_by_year=solver_kwargs.get('land_carbon_price_by_year'),
                grassland_method=str(solver_kwargs.get('grassland_method', 'dynamic') or 'dynamic'),
                grass_area_by_region_year=solver_kwargs.get('grass_area_by_region_year'),
            )
            if crop_state:
                base_cropland_by_region = crop_state
            if grass_state:
                base_grassland_by_region = grass_state
            if forest_state:
                base_forest_by_region = forest_state
                for region, val in forest_state.items():
                    forest_area_by_region_year[(region, int(target_year))] = float(val)
        except Exception as exc:
            logger.warning("[LINEAR] rolling step %s land-state update failed: %s", target_year, exc)

        prev_year = int(target_year)

    idx, regions, years_out = _build_regional_idx_from_nodes(
        working_nodes,
        commodities=solver_kwargs.get('commodities') or [],
        years=years,
        dict_v3_path=solver_kwargs.get('dict_v3_path'),
        population_by_country_year=solver_kwargs.get('population_by_country_year'),
        income_mult_by_country_year=solver_kwargs.get('income_mult_by_country_year'),
        hist_end_year=base_hist_end_year,
    )
    combined['idx'] = idx
    combined['regions'] = regions
    combined['years'] = years_out
    combined['slack_objective'] = slack_total
    combined['objective'] = objective_total
    combined['process_abatement_cost_usd'] = float(sum(combined['Cij'].values()))
    combined['strategy_cost_contributions'] = {
        key: float(quantity) * float(combined['strategy_abatement_costs'].get(key, 0.0))
        for key, quantity in combined['strategy_abatement'].items()
        if key in combined['strategy_abatement_costs']
    }
    combined['strategy_abatement_cost_usd'] = float(
        sum(combined['strategy_cost_contributions'].values())
    )
    combined['total_abatement_cost_usd'] = (
        combined['process_abatement_cost_usd']
        + combined['strategy_abatement_cost_usd']
    )
    if excluded_items:
        combined['excluded_commodities'] = sorted(excluded_items)
    return combined


def solve_linear_regional(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    time_limit: float = 300.0,
    dict_v3_path: Optional[str] = None,
    output_dir: Optional[str] = None,  # IIS output directory
    gurobi_log_path: Optional[str] = None,  # Gurobi log file path
    solver_method: Optional[int] = None,
    solver_threads: Optional[int] = None,
    use_relative_price: bool = False,
    relative_price_bounds: Tuple[float, float] = (0.1, 10.0),
    price_bounds_mode: str = 'absolute',
    price_bounds_p0_mult: Tuple[float, float] = (0.1, 10.0),
    price_bounds: Tuple[float, float] = (1e-6, 1e6),
    price_wedge_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    price_wedge_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    price_wedge_by_region: Optional[Dict[str, float]] = None,
    market_clearing_mode: str = 'country_trade',
    armington_sigma_by_comm: Optional[Dict[Any, float]] = None,
    trade_base_net_import: Optional[Dict[Tuple[str, str], float]] = None,
    trade_base_volume: Optional[Dict[Tuple[str, str], float]] = None,
    trade_cap_region_volume: Optional[Dict[Tuple[str, str], float]] = None,
    trade_cap_region_map: Optional[Dict[Any, str]] = None,
    trade_cap_ratio: Any = None,
    trade_cap_exempt_pairs: Optional[set] = None,
    armington_trade_scale: Optional[float] = None,
    armington_trade_slack_penalty: Optional[float] = None,
    supply_curtailment_enabled: bool = False,
    supply_curtailment_penalty: Optional[float] = None,
    zero_price_shutdown_enabled: bool = False,
    zero_demand_production_shutdown: bool = False,
    qty_scale: float = 1.0,
    land_scale: float = 1.0,
    # Population and income
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    # Emissions and abatement parameters
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    # LUC in optimization
    luc_opt_mode: str = 'none',  # 'none' | 'explicit' | 'iterative'
    luc_params: Optional[Dict[str, Any]] = None,
    luc_shift_area_mode: str = 'abs',
    luc_penalty_by_region_year: Optional[Dict[Tuple[str, int], Dict[str, float]]] = None,
    # Constraint parameters
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    land_soft_constraints_enabled: bool = False,
    land_slack_max_rate: Optional[float] = None,
    land_slack_penalty: Optional[float] = None,
    land_delta_anchor_to_available_stock: bool = False,
    grass_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,  # Grassland area {(region, year): ha}
    forest_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,  # Forest area {(region, year): ha}
    forest_global_target_slack_enabled: bool = False,
    forest_global_target_slack_penalty: Optional[float] = None,
    forest_global_target_slack_max_rate: Optional[float] = None,
    forest_nonneg_ratio: float = 1.0,
    cropland_nonforest_expand_ratio: float = 1.0,
    pasture_nonforest_expand_ratio: float = 1.0,
    base_cropland_by_region: Optional[Dict[str, float]] = None,  # LUH2 base-period cropland area {(region): ha}
    base_grassland_by_region: Optional[Dict[str, float]] = None,  # LUH2 base-period grassland area {(region): ha}
    base_forest_by_region: Optional[Dict[str, float]] = None,  # LUH2/FAO base-period forest area {(region): ha}
    background_forest_to_cropland_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_forest_to_grassland_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_cropland_to_forest_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_grassland_to_forest_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    land_demand_calibration_mode: str = 'none',
    yield_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    grassland_method: str = 'dynamic',  # 'dynamic' (approach A) or 'static' (approach B)
    grassland_conversion_penalty: float = 0.0,
    grassland_to_cropland_cost_mode: str = 'per_ha_cost',
    cropland_to_grassland_penalty: float = 0.0,
    land_conversion_allocation_mode: str = 'priority_nonforest_pasture_forest',
    land_conversion_priority_penalty_per_ha: float = 1e6,
    land_priority_weight_grassland_to_cropland: float = 1.0,
    land_priority_weight_forest_to_cropland: float = 100.0,
    land_priority_weight_forest_to_grassland: float = 100.0,
    luc_direct_carbon_price: bool = False,
    limit_reforestation_to_released_ag_land: bool = True,
    prevent_land_conversion_cycles: bool = True,
    reforestation_physical_cap_enabled: bool = True,
    reforestation_max_forest_increase_ratio: Optional[float] = 0.30,
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    hist_max_production: Optional[Dict[Tuple[str, str], float]] = None,
    future_last_only: bool = True,
    hist_max_small_prod_exempt_t: Optional[float] = None,
    hist_max_small_prod_floor_t: Optional[float] = None,
    rolling_future_solve: Optional[bool] = None,
    objective_strategy: str = 'auto',
    # Scenario parameters (Phase 2)
    tax_unit_adder: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    waste_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    losses_ratio_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_mode: Optional[str] = None,
    feed_crop_link_base: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_coeff: Optional[Dict[Tuple[str, str, str, int], float]] = None,
    feed_background_crop: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_background_pasture: Optional[Dict[Tuple[str, int], float]] = None,
    feed_crop_link_livestock: Optional[List[str]] = None,
    feed_crop_link_override: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_credit: Optional[Dict[Tuple[str, str, int], float]] = None,
    ruminant_intake_cap: Optional[Dict[Tuple[str, int], float]] = None,
    ruminant_commodities: Optional[List[str]] = None,
    # Market imbalance limits
    max_slack_rate: Optional[float] = 0.1,
    max_shortage_slack_rate: Any = "inherit",
    max_excess_slack_rate: Any = "inherit",
    slack_penalty: Optional[float] = 1e6,
    disable_production_cost_term: bool = True,
    production_cost_weight: float = 1.0,
    # Cross-elasticity term clipping
    cross_terms_top_n: Optional[int] = None,
    cross_terms_scale: Optional[float] = None,
    exclude_commodities: Optional[List[str]] = None,
    # Unit-cost method parameters
    unit_cost_data: Optional[Dict[Tuple[str, str], float]] = None,
    baseline_scenario_result: Optional[Dict[str, Any]] = None,
    process_cost_mapping: Optional[Dict[str, str]] = None,
    cost_calculation_method: str = 'MACC',
    active_strategy_cost_keys: Optional[Sequence[str]] = None,
    strategy_cost_regions: Optional[Sequence[str]] = None,
    cost_database_metadata: Optional[Mapping[str, Any]] = None,
    cost_strategy_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    # Demand projection method
    demand_method: str = 'elasticity',  # 'elasticity' | 'nutrition' | 'nutrition_band' | 'nutrition_anchor'
    nutrition_profile_xlsx: Optional[str] = None,
    nutrition_profile_sheet: Any = 0,
    nutrition_indicator: str = 'energy',
    nutrition_use_baseyear_for_future: bool = True,
    nutrition_band_epsilon: float = 0.1,
    nutrition_feed_t_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    nutrition_residual_demand_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    bioenergy_crop_demand_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    energy_crop_land_requirement_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    post_solve_violation_tol: Optional[float] = 1e-6,
    post_solve_violation_top_n: int = 20,
    enable_infeasible_iis: bool = True,
    enable_violation_iis: bool = True,
    enable_output_diagnostics: bool = False,
    enable_verbose_logging: bool = True,
) -> Dict[str, Any]:
    """
    Solve the full linear regional model.
    
    Returns:
    - status: Gurobi status code.
    - Pc: Prices {(j, t): value}.
    - Qs: Regional supply {(r, j, t): value}.
    - Qd: Regional demand {(r, j, t): value}.
    - net_import: Regional net imports {(r, j, t): value}.
    - Eij: Regional emissions {(r, j, t): value}.
    - Cij: Regional abatement costs {(r, j, t): value}.
    - abatement: Abatement {(r, j, t, proc, seg): value}.
    """
    logger = logging.getLogger(__name__)
    
    # Build the full model.
    exclude_norm = {str(c).strip().lower() for c in (exclude_commodities or []) if str(c).strip()}
    if exclude_norm:
        model_commodities = [c for c in commodities if str(c).strip().lower() not in exclude_norm]
        excluded_items = [c for c in commodities if str(c).strip().lower() in exclude_norm]
        if not model_commodities:
            raise ValueError("[LINEAR] exclude_commodities removes all commodities from model")
        if not excluded_items:
            logger.info("[LINEAR] exclude_commodities specified but no match in commodities list")
    else:
        model_commodities = commodities
        excluded_items = []

    future_years_in_run = _future_years_after_hist(years, hist_end_year)
    objective_strategy_norm = _normalize_linear_objective_strategy(
        objective_strategy,
        future_years_in_run,
    )
    rolling_enabled = (
        len(future_years_in_run) > 1
        and (
            bool(rolling_future_solve)
            if rolling_future_solve is not None
            else True
        )
    )
    if rolling_enabled:
        logger.info(
            "[LINEAR] multi-future run detected (%s); use rolling solve with objective=%s",
            future_years_in_run,
            objective_strategy_norm,
        )
        rolling_kwargs = locals().copy()
        for drop_key in (
            'logger',
            'exclude_norm',
            'model_commodities',
            'excluded_items',
            'future_years_in_run',
            'objective_strategy_norm',
            'rolling_enabled',
        ):
            rolling_kwargs.pop(drop_key, None)
        return _solve_linear_regional_rolling(
            logger,
            rolling_kwargs,
            objective_strategy=objective_strategy_norm,
        )

    m = build_linear_regional_model(
        nodes=nodes,
        commodities=model_commodities,
        years=years,
        dict_v3_path=dict_v3_path,
        output_dir=output_dir,
        gurobi_log_path=gurobi_log_path,
        solver_method=solver_method,
        solver_threads=solver_threads,
        use_relative_price=use_relative_price,
        relative_price_bounds=relative_price_bounds,
        price_bounds_mode=price_bounds_mode,
        price_bounds_p0_mult=price_bounds_p0_mult,
        price_bounds=price_bounds,
        price_wedge_by_region_comm_year=price_wedge_by_region_comm_year,
        price_wedge_by_region_comm=price_wedge_by_region_comm,
        price_wedge_by_region=price_wedge_by_region,
        market_clearing_mode=market_clearing_mode,
        armington_sigma_by_comm=armington_sigma_by_comm,
        trade_base_net_import=trade_base_net_import,
        trade_base_volume=trade_base_volume,
        trade_cap_region_volume=trade_cap_region_volume,
        trade_cap_region_map=trade_cap_region_map,
        trade_cap_ratio=trade_cap_ratio,
        trade_cap_exempt_pairs=trade_cap_exempt_pairs,
        armington_trade_scale=armington_trade_scale,
        armington_trade_slack_penalty=armington_trade_slack_penalty,
        qty_scale=qty_scale,
        land_scale=land_scale,
        population_by_country_year=population_by_country_year,
        income_mult_by_country_year=income_mult_by_country_year,
        macc_path=macc_path,
        land_carbon_price_by_year=land_carbon_price_by_year,
        luc_opt_mode=luc_opt_mode,
        luc_params=luc_params,
        luc_shift_area_mode=luc_shift_area_mode,
        luc_penalty_by_region_year=luc_penalty_by_region_year,
        nutrition_rhs=nutrition_rhs,
        nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
        land_area_limits=land_area_limits,
        land_soft_constraints_enabled=land_soft_constraints_enabled,
        land_slack_max_rate=land_slack_max_rate,
        land_slack_penalty=land_slack_penalty,
        grass_area_by_region_year=grass_area_by_region_year,  # Pass grassland area.
        forest_area_by_region_year=forest_area_by_region_year,  # Pass forest area.
        forest_global_target_slack_enabled=forest_global_target_slack_enabled,
        forest_global_target_slack_penalty=forest_global_target_slack_penalty,
        forest_global_target_slack_max_rate=forest_global_target_slack_max_rate,
        forest_nonneg_ratio=forest_nonneg_ratio,
        cropland_nonforest_expand_ratio=cropland_nonforest_expand_ratio,
        pasture_nonforest_expand_ratio=pasture_nonforest_expand_ratio,
        base_cropland_by_region=base_cropland_by_region,
        base_grassland_by_region=base_grassland_by_region,
        base_forest_by_region=base_forest_by_region,
        background_forest_to_cropland_by_region_year=background_forest_to_cropland_by_region_year,
        background_forest_to_grassland_by_region_year=background_forest_to_grassland_by_region_year,
        background_cropland_to_forest_by_region_year=background_cropland_to_forest_by_region_year,
        background_grassland_to_forest_by_region_year=background_grassland_to_forest_by_region_year,
        land_demand_calibration_mode=land_demand_calibration_mode,
        yield_by_region_comm_year=yield_by_region_comm_year,
        yield_t_per_ha_default=yield_t_per_ha_default,
        grassland_method=grassland_method,  # Pass the grassland handling method.
        grassland_conversion_penalty=grassland_conversion_penalty,
        grassland_to_cropland_cost_mode=grassland_to_cropland_cost_mode,
        cropland_to_grassland_penalty=cropland_to_grassland_penalty,
        land_conversion_allocation_mode=land_conversion_allocation_mode,
        land_conversion_priority_penalty_per_ha=land_conversion_priority_penalty_per_ha,
        land_priority_weight_grassland_to_cropland=land_priority_weight_grassland_to_cropland,
        land_priority_weight_forest_to_cropland=land_priority_weight_forest_to_cropland,
        land_priority_weight_forest_to_grassland=land_priority_weight_forest_to_grassland,
        luc_direct_carbon_price=luc_direct_carbon_price,
        limit_reforestation_to_released_ag_land=limit_reforestation_to_released_ag_land,
        prevent_land_conversion_cycles=prevent_land_conversion_cycles,
        reforestation_physical_cap_enabled=reforestation_physical_cap_enabled,
        reforestation_max_forest_increase_ratio=reforestation_max_forest_increase_ratio,
        max_growth_rate_per_period=max_growth_rate_per_period,
        max_decline_rate_per_period=max_decline_rate_per_period,
        hist_end_year=hist_end_year,
        hist_max_production=hist_max_production,
        future_last_only=future_last_only,
        hist_max_small_prod_exempt_t=hist_max_small_prod_exempt_t,
        hist_max_small_prod_floor_t=hist_max_small_prod_floor_t,
        # Phase 2 scenario parameters
        tax_unit_adder=tax_unit_adder,
        feed_reduction_by=feed_reduction_by,
        waste_reduction_by=waste_reduction_by,
        losses_ratio_by=losses_ratio_by,
        feed_crop_link_mode=feed_crop_link_mode,
        feed_crop_link_base=feed_crop_link_base,
        feed_crop_link_coeff=feed_crop_link_coeff,
        feed_background_crop=feed_background_crop,
        feed_background_pasture=feed_background_pasture,
        feed_crop_link_livestock=feed_crop_link_livestock,
        feed_crop_link_override=feed_crop_link_override,
        feed_crop_link_credit=feed_crop_link_credit,
        ruminant_intake_cap=ruminant_intake_cap,
        ruminant_commodities=ruminant_commodities,
        # Market imbalance limits
        max_slack_rate=max_slack_rate,
        max_shortage_slack_rate=max_shortage_slack_rate,
        max_excess_slack_rate=max_excess_slack_rate,
        slack_penalty=slack_penalty,
        supply_curtailment_enabled=supply_curtailment_enabled,
        supply_curtailment_penalty=supply_curtailment_penalty,
        zero_price_shutdown_enabled=zero_price_shutdown_enabled,
        zero_demand_production_shutdown=zero_demand_production_shutdown,
        land_delta_anchor_to_available_stock=land_delta_anchor_to_available_stock,
        disable_production_cost_term=disable_production_cost_term,
        production_cost_weight=production_cost_weight,
        cross_terms_top_n=cross_terms_top_n,
        cross_terms_scale=cross_terms_scale,
        # Unit-cost method parameters
        unit_cost_data=unit_cost_data,
        baseline_scenario_result=baseline_scenario_result,
        process_cost_mapping=process_cost_mapping,
        cost_calculation_method=cost_calculation_method,
        active_strategy_cost_keys=active_strategy_cost_keys,
        strategy_cost_regions=strategy_cost_regions,
        cost_database_metadata=cost_database_metadata,
        cost_strategy_metadata=cost_strategy_metadata,
        demand_method=demand_method,
        nutrition_profile_xlsx=nutrition_profile_xlsx,
        nutrition_profile_sheet=nutrition_profile_sheet,
        nutrition_indicator=nutrition_indicator,
        nutrition_use_baseyear_for_future=nutrition_use_baseyear_for_future,
        nutrition_band_epsilon=nutrition_band_epsilon,
        nutrition_feed_t_by_country_comm_year=nutrition_feed_t_by_country_comm_year,
        nutrition_residual_demand_by_country_comm_year=nutrition_residual_demand_by_country_comm_year,
        bioenergy_crop_demand_by_country_comm_year=bioenergy_crop_demand_by_country_comm_year,
        energy_crop_land_requirement_by_region_year=energy_crop_land_requirement_by_region_year,
        enable_output_diagnostics=enable_output_diagnostics,
        enable_verbose_logging=enable_verbose_logging,
    )
    # Remove the 300.0-second solve limit as requested by the user.
    # m.setParam('TimeLimit', time_limit)

    result: Dict[str, Any] = {
        'status': None,
        'model': m,
        'objective_strategy': objective_strategy_norm,
    }
    if objective_strategy_norm == 'lexicographic_slack':
        cache = getattr(m, '_nzf_cache', {}) or {}
        slack_obj = cache.get('objective_feasibility')
        cost_obj = cache.get('objective_cost')
        if slack_obj is None or cost_obj is None:
            logger.warning("[LINEAR] lexicographic objective unavailable in cache; fallback to single_stage")
            objective_strategy_norm = 'single_stage'
            result['objective_strategy'] = objective_strategy_norm
        else:
            m.setObjective(slack_obj, gp.GRB.MINIMIZE)
            m.update()
            phase1_status = _optimize_model_with_retries(m, logger, phase_label="phase1-slack")
            result['status_phase1'] = phase1_status
            if phase1_status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL) and m.SolCount > 0:
                slack_opt = float(m.ObjVal)
                slack_tol = max(1e-8, abs(slack_opt) * 1e-6)
                result['slack_objective'] = slack_opt
                result['slack_objective_tol'] = slack_tol
                slack_fix = m.addConstr(
                    slack_obj <= slack_opt + slack_tol,
                    name="lexicographic_slack_fix",
                )
                m.update()
                m.setObjective(cost_obj, gp.GRB.MINIMIZE)
                m.update()
                status = _optimize_model_with_retries(m, logger, phase_label="phase2-cost")
                result['status_phase2'] = status
                result['slack_fix_constr'] = slack_fix.ConstrName
            else:
                status = phase1_status
    if objective_strategy_norm == 'single_stage':
        status = _optimize_model_with_retries(m, logger)

    if status == gp.GRB.OPTIMAL:
        result['land_source_selection'] = _select_land_sources_on_optimal_face(m, logger)
        status = int(m.Status)
        if output_dir and result['land_source_selection']:
            selection_path = Path(output_dir) / 'Diagnostics' / 'land_source_selection.json'
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(json.dumps(result['land_source_selection'], indent=2), encoding='utf-8')
    result['status'] = status
    if getattr(m, 'SolCount', 0) > 0:
        try:
            result['objective'] = float(m.ObjVal)
        except Exception:
            pass

    if status in (gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED) and enable_infeasible_iis:
        try:
            if output_dir:
                model_path = Path(output_dir) / "linear_model_inf_or_unbd.lp"
            else:
                model_path = Path("linear_model_inf_or_unbd.lp")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            m.write(str(model_path))
            logger.info(f"[LINEAR] 状态{status}模型已保存到: {model_path}")
        except Exception as e:
            logger.warning(f"[LINEAR] 无法保存状态{status}模型文件: {e}")
    
    # IIS analysis when the model is infeasible
    if status == gp.GRB.INFEASIBLE:
        if not enable_infeasible_iis:
            logger.warning("[LINEAR] 模型不可行；batch mode 已关闭 LP/IIS 输出")
        else:
            logger.warning("[LINEAR] 模型不可行，开始 IIS 分析...")
            
            # Save the model for manual inspection before IIS analysis, which may fail due to numerical issues.
            try:
                if output_dir:
                    model_path = Path(output_dir) / "linear_model_infeasible.lp"
                else:
                    model_path = Path("linear_model_infeasible.lp")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                m.write(str(model_path))
                logger.info(f"[LINEAR] 不可行模型已保存到: {model_path}")
            except Exception as e:
                logger.warning(f"[LINEAR] 无法保存模型文件: {e}")
            
            try:
                m.computeIIS()

                # Collect IIS constraints.
                iis_constrs = []
                iis_bounds = []

                for c in m.getConstrs():
                    if c.IISConstr:
                        iis_constrs.append(c.ConstrName)

                for v in m.getVars():
                    if v.IISLB:
                        iis_bounds.append(f"{v.VarName} (lower bound)")
                    if v.IISUB:
                        iis_bounds.append(f"{v.VarName} (upper bound)")

                logger.error(f"[LINEAR] IIS 包含 {len(iis_constrs)} 个约束, {len(iis_bounds)} 个变量边界")

                # Count IIS constraints by category.
                iis_by_type = {}
                for cname in iis_constrs:
                    # Extract the constraint type before the opening bracket.
                    ctype = cname.split('[')[0] if '[' in cname else cname
                    iis_by_type[ctype] = iis_by_type.get(ctype, 0) + 1

                logger.error("[LINEAR] IIS 约束类型统计:")
                for ctype, count in sorted(iis_by_type.items(), key=lambda x: -x[1]):
                    logger.error(f"  {ctype}: {count} 个")

                # Show details of the first ten IIS constraints.
                logger.error("[LINEAR] IIS 约束示例 (前20个):")
                for cname in iis_constrs[:20]:
                    logger.error(f"  - {cname}")

                # Show IIS variable bounds.
                if iis_bounds:
                    logger.error("[LINEAR] IIS 变量边界 (前10个):")
                    for vname in iis_bounds[:10]:
                        logger.error(f"  - {vname}")

                # Check IIS direction: excessive demand with limited net imports versus excessive supply with limited net exports.
                def _parse_iis_key(name: str) -> Optional[Tuple[str, str, str]]:
                    if '[' not in name or ']' not in name:
                        return None
                    content = name[name.find('[') + 1:name.rfind(']')]
                    if ',' not in content:
                        return None
                    first = content.find(',')
                    last = content.rfind(',')
                    if first == -1 or last == -1 or first == last:
                        return None
                    r = content[:first].strip()
                    j = content[first + 1:last].strip()
                    t = content[last + 1:].strip()
                    return (r, j, t)

                trade_pos = set()
                trade_neg = set()
                supply_keys = set()
                demand_keys = set()
                balance_keys = set()
                pc_ub = set()
                pc_lb = set()
                for cname in iis_constrs:
                    ctype = cname.split('[')[0] if '[' in cname else cname
                    key = _parse_iis_key(cname)
                    if not key:
                        continue
                    if ctype == 'trade_cap_region_pos':
                        trade_pos.add(key)
                    elif ctype == 'trade_cap_region_neg':
                        trade_neg.add(key)
                    elif ctype == 'supply':
                        supply_keys.add(key)
                    elif ctype.startswith('demand'):
                        demand_keys.add(key)
                    elif ctype == 'balance':
                        balance_keys.add(key)
                for vname in iis_bounds:
                    base_name = vname.split(' (', 1)[0]
                    key = _parse_iis_key(base_name)
                    if not key:
                        continue
                    if 'upper bound' in vname:
                        pc_ub.add(key)
                    if 'lower bound' in vname:
                        pc_lb.add(key)

                # Aggregate by commodity-year.
                comm_years = {(j, t) for _, j, t in (supply_keys | demand_keys | balance_keys | trade_pos | trade_neg | pc_ub | pc_lb)}
                if comm_years:
                    logger.error("[IIS-DIRECTION] 方向性诊断(供给过大/需求过大):")
                    for j, t in sorted(comm_years):
                        pos_r = {r for (r, jj, tt) in trade_pos if jj == j and tt == t}
                        neg_r = {r for (r, jj, tt) in trade_neg if jj == j and tt == t}
                        ub_r = {r for (r, jj, tt) in pc_ub if jj == j and tt == t}
                        lb_r = {r for (r, jj, tt) in pc_lb if jj == j and tt == t}
                        if pos_r and not neg_r:
                            reason = "需求过大/净进口受限"
                        elif neg_r and not pos_r:
                            reason = "供给过大/净出口受限"
                        elif pos_r and neg_r:
                            reason = "进出口上限同时触发(方向混合)"
                        else:
                            if ub_r and not lb_r:
                                reason = "供给不足(价格上界限制供给扩张)"
                            elif lb_r and not ub_r:
                                reason = "供给过剩(价格下界限制供给下降)"
                            else:
                                reason = "方向不明(无贸易上限、价格边界不明确)"
                        logger.error(
                            "  - item=%s year=%s | import_cap=%d export_cap=%d Pc_ub=%d Pc_lb=%d => %s",
                            j, t, len(pos_r), len(neg_r), len(ub_r), len(lb_r), reason
                        )

                # Check nutrition conflicts implicated by the IIS.
                method = str(demand_method).lower()
                if method in {'nutrition', 'nutrition_band', 'nutrition_anchor'} and nutrition_rhs:
                    cache = getattr(m, '_nzf_cache', {})
                    nutrition_map = cache.get('nutrition_demand_map', {}) or {}
                    nutrient_dict = cache.get('nutrient_per_unit_by_comm', {}) or {}
                    if nutrition_map and nutrient_dict:
                        iis_rt = set()
                        iis_rt_comms: Dict[Tuple[str, int], set] = {}
                        for cname in iis_constrs:
                            if cname.startswith('nutri[') and cname.endswith(']'):
                                content = cname[cname.find('[') + 1:cname.rfind(']')]
                                if ',' in content:
                                    r_str, t_str = content.rsplit(',', 1)
                                    try:
                                        iis_rt.add((r_str.strip(), int(t_str.strip())))
                                    except Exception:
                                        continue
                            elif (
                                cname.startswith('demand_nutrition[')
                                or cname.startswith('demand_nutrition_lb[')
                                or cname.startswith('demand_nutrition_ub[')
                            ) and cname.endswith(']'):
                                content = cname[cname.find('[') + 1:cname.rfind(']')]
                                if ',' in content:
                                    r_part, rest = content.split(',', 1)
                                    if ',' in rest:
                                        j_part, t_str = rest.rsplit(',', 1)
                                        try:
                                            key = (r_part.strip(), int(t_str.strip()))
                                            iis_rt.add(key)
                                            iis_rt_comms.setdefault(key, set()).add(j_part.strip())
                                        except Exception:
                                            continue
                        for r, t in sorted(iis_rt):
                            rhs_val = nutrition_rhs.get((r, t))
                            if rhs_val is None:
                                continue
                            comms = iis_rt_comms.get((r, t))
                            total = 0.0
                            missing_map = 0
                            missing_kcal = 0
                            if comms:
                                for j in comms:
                                    demand_val = nutrition_map.get((r, j, t))
                                    if demand_val is None:
                                        missing_map += 1
                                        continue
                                    kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                                    if kcal_per_ton <= 0:
                                        missing_kcal += 1
                                        continue
                                    total += float(demand_val) * kcal_per_ton
                            else:
                                for (rr, j, tt), demand_val in nutrition_map.items():
                                    if rr == r and tt == t:
                                        kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                                        if kcal_per_ton <= 0:
                                            continue
                                        total += float(demand_val) * kcal_per_ton
                            nutri_gap = float(rhs_val) - total
                            logger.error(
                                "[NUTRI_GAP] region=%s year=%s nutri_gap=%.6e RHS=%.6e -sum_kcal=%.6e comms=%s missing_map=%s missing_kcal=%s",
                                r, t, nutri_gap, float(rhs_val), -total,
                                len(comms) if comms else 0, missing_map, missing_kcal
                            )

                # Save IIS information in the results.
                result['iis_constrs'] = iis_constrs
                result['iis_bounds'] = iis_bounds
                result['iis_by_type'] = iis_by_type

                # Write the IIS file.
                try:
                    if output_dir:
                        iis_path = Path(output_dir) / "linear_model_iis.ilp"
                    else:
                        iis_path = Path("linear_model_iis.ilp")
                    iis_path.parent.mkdir(parents=True, exist_ok=True)
                    m.write(str(iis_path))
                    logger.info(f"[LINEAR] IIS 已保存到: {iis_path}")
                    result['iis_path'] = str(iis_path)
                except Exception as e:
                    logger.warning(f"[LINEAR] 无法保存 IIS 文件: {e}")

            except Exception as e:
                logger.error(f"[LINEAR] IIS 分析失败: {e}")
    
    if solver_result_is_extractable(
        status,
        getattr(m, "SolCount", 0),
    ):
        cache = m._nzf_cache
        try:
            qty_scale = float(cache.get('qty_scale', 1.0) or 1.0)
        except Exception:
            qty_scale = 1.0
        if not np.isfinite(qty_scale) or qty_scale <= 0:
            qty_scale = 1.0
        try:
            land_scale = float(cache.get('land_scale', 1.0) or 1.0)
        except Exception:
            land_scale = 1.0
        if not np.isfinite(land_scale) or land_scale <= 0:
            land_scale = 1.0

        def _land_out_value(value: Any) -> float:
            try:
                return float(value or 0.0) * land_scale
            except Exception:
                return 0.0

        def _land_expr_out(expr: Any) -> float:
            try:
                val = float(expr.getValue()) if hasattr(expr, 'getValue') else float(expr)
            except Exception:
                val = 0.0
            return val * land_scale

        def _land_var_out(var: Any) -> float:
            try:
                val = float(var.X) if hasattr(var, 'X') else float(var or 0.0)
            except Exception:
                val = 0.0
            return val * land_scale

        def _land_dict_out(mapping: Mapping[Any, Any]) -> Dict[Any, float]:
            return {k: _land_out_value(v) for k, v in (mapping or {}).items()}
        
        # Basic results
        pc_rel = {k: v.X for k, v in cache['Pc'].items()}
        use_rel = bool(cache.get('use_relative_price'))
        pc_by_region = bool(cache.get('pc_by_region', False))
        if use_rel:
            price_ref = cache.get('price_ref_by_comm', {}) or {}
            if pc_by_region:
                pc_abs: Dict[Tuple[str, str, int], float] = {}
                for (r, j, t), val in pc_rel.items():
                    ref = float(price_ref.get(j, 1.0) or 1.0)
                    if not np.isfinite(ref) or ref <= 0:
                        ref = 1.0
                    pc_abs[(r, j, t)] = float(val) * ref
            else:
                pc_abs = {}
                for (j, t), val in pc_rel.items():
                    ref = float(price_ref.get(j, 1.0) or 1.0)
                    if not np.isfinite(ref) or ref <= 0:
                        ref = 1.0
                    pc_abs[(j, t)] = float(val) * ref
            result['Pc'] = pc_abs
            result['Pc_rel'] = pc_rel
        else:
            result['Pc'] = pc_rel
        result['Qs'] = {k: v.X * qty_scale for k, v in cache['Qs'].items()}
        result['Qd'] = {k: v.X * qty_scale for k, v in cache['Qd'].items()}
        result['seed_demand'] = {k: float(v.getValue()) * qty_scale for k,v in cache.get('seed_demand_expr_by_key',{}).items()}
        if output_dir and result['seed_demand']:
            pd.DataFrame([dict(region=k[0],commodity=k[1],year=k[2],seed_t=v)
                          for k,v in result['seed_demand'].items()]).to_csv(
                Path(output_dir)/'seed_demand.csv',index=False)

        ces_state = cache.get('armington_ces', {})
        if ces_state.get('records'):
            if ces_state.get('method') == 'analytic_price_elimination':
                recover_projected_ces(ces_state)
                for key,rec in ces_state['records'].items():
                    ref=(price_ref.get(key[1],1.) if use_rel else 1.)
                    result['Pc'][key]=ces_value(rec,'pd')*ref
                    if use_rel: result['Pc_rel'][key]=ces_value(rec,'pd')
            result['armington_method'] = ces_state.get('method','sequential_linearization')
            result['armington_converged'] = bool(ces_state.get('converged'))
            result['armington_iterations'] = ces_state['history']
            for label, field in [('domestic_sales', 'domestic'), ('gross_imports','imports'),
                                 ('gross_exports','exports'), ('ces_composite_quantity','composite')]:
                result[label] = {key: ces_value(rec,field) * qty_scale for key,rec in ces_state['records'].items()}
            result['consumer_price'] = {key: ces_value(rec,'price') * (price_ref.get(key[1],1.) if use_rel else 1.)
                                        for key,rec in ces_state['records'].items()}
            if output_dir:
                rows = [dict(region=key[0], commodity=key[1], year=key[2],
                             physical_demand_t=result['Qd'][key] + rec['bio']*qty_scale,
                             domestic_sales_t=rec['domestic'].X*qty_scale,
                             gross_imports_t=rec['imports'].X*qty_scale,
                             gross_exports_t=rec['exports'].X*qty_scale,
                             ces_composite_quantity=ces_value(rec,'composite')*qty_scale,
                             consumer_price=result['consumer_price'][key], reference_status=rec['reference_status'],
                             domestic_price=ces_value(rec,'pd')*(price_ref.get(key[1],1.) if use_rel else 1.),
                             import_price=ces_value(rec,'pm')*(price_ref.get(key[1],1.) if use_rel else 1.),
                             ces_method=ces_state.get('method','sequential_linearization'))
                        for key,rec in ces_state['records'].items()]
                pd.DataFrame(rows).to_csv(Path(output_dir)/'Diagnostics'/'armington_sources.csv',index=False)
                (Path(output_dir)/'Diagnostics'/'armington_iterations.json').write_text(
                    json.dumps(ces_state['history'],indent=2),encoding='utf-8')

        result['bioenergy_crop_demand'] = dict(cache.get('bioenergy_crop_demand_map', {}) or {})
        result['energy_crop_land_requirement'] = _land_dict_out(
            cache.get('energy_crop_land_requirement_map', {}) or {}
        )
        feed_expr_map = cache.get('feed_demand_expr_by_key', {}) or {}
        if feed_expr_map:
            feed_result: Dict[Tuple[str, str, int], float] = {}
            feed_credit_result: Dict[Tuple[str, str, int], float] = {}
            feed_net_result: Dict[Tuple[str, str, int], float] = {}
            feed_credit_scaled = cache.get('feed_credit_scaled_by_key', {}) or {}
            for k, expr in feed_expr_map.items():
                try:
                    if hasattr(expr, 'getValue'):
                        val = float(expr.getValue())
                    else:
                        val = float(expr)
                except Exception:
                    continue
                if np.isfinite(val) and val > 0.0:
                    feed_result[k] = val * qty_scale
                    credit_item = feed_credit_scaled.get(k, 0.0)
                    credit_val = float(credit_item.X if hasattr(credit_item,'X') else credit_item) * qty_scale
                    if credit_val > 0.0:
                        feed_credit_result[k] = credit_val
                        feed_net_result[k] = max(0.0, val * qty_scale - credit_val)
            result['feed_demand'] = feed_result
            if feed_credit_result:
                result['feed_credit'] = feed_credit_result
                result['feed_demand_net_of_bioenergy_credit'] = feed_net_result
        if cache.get('supply_curtailment'):
            result['supply_curtailment'] = {
                k: v.X * qty_scale for k, v in cache.get('supply_curtailment', {}).items()
            }
        result['net_import'] = {k: v.X * qty_scale for k, v in cache.get('net_import', {}).items()}
        result['excess'] = {k: v.X * qty_scale for k, v in cache['excess'].items()}
        result['shortage'] = {k: v.X * qty_scale for k, v in cache['shortage'].items()}
        result['land_slack'] = {
            k: _land_var_out(v) for k, v in cache.get('land_slack', {}).items()
        }
        
        # Emissions results
        result['Eij'] = {k: v.X for k, v in cache.get('Eij', {}).items()}
        result['Cij'] = {k: v.X for k, v in cache.get('Cij', {}).items()}
        result['base_cropland_demand'] = _land_dict_out(cache.get('base_cropland_demand', {}) or {})
        result['base_grassland_demand'] = _land_dict_out(cache.get('base_grassland_demand', {}) or {})
        result['base_cropland_demand_raw'] = _land_dict_out(cache.get('base_cropland_demand_raw', {}) or {})
        result['base_grassland_demand_raw'] = _land_dict_out(cache.get('base_grassland_demand_raw', {}) or {})
        result['land_demand_calibration_mode'] = cache.get('land_demand_calibration_mode', 'none')
        result['land_demand_crop_scale_by_region'] = dict(cache.get('land_demand_crop_scale_by_region', {}) or {})
        result['land_demand_grass_scale_by_region'] = dict(cache.get('land_demand_grass_scale_by_region', {}) or {})
        result['base_cropland_area'] = _land_dict_out(cache.get('base_cropland_area', {}) or {})
        result['base_grassland_area'] = _land_dict_out(cache.get('base_grassland_area', {}) or {})
        result['base_forest_area'] = _land_dict_out(cache.get('base_forest_area', {}) or {})
        result['land_demand_expansion_need'] = {
            k: _land_var_out(v)
            for k, v in (cache.get('land_demand_expansion_need', {}) or {}).items()
        }
        result['land_demand_contraction_need'] = {
            k: _land_var_out(v)
            for k, v in (cache.get('land_demand_contraction_need', {}) or {}).items()
        }
        result['cropland_actual_by_region_year'] = {
            k: _land_expr_out(expr)
            for k, expr in (cache.get('cropland_actual_expr_by_region_year', {}) or {}).items()
        }
        result['grassland_actual_by_region_year'] = {
            k: _land_expr_out(expr)
            for k, expr in (cache.get('grassland_actual_expr_by_region_year', {}) or {}).items()
        }
        result['forest_actual_by_region_year'] = {
            k: _land_expr_out(expr)
            for k, expr in (cache.get('forest_actual_expr_by_region_year', {}) or {}).items()
        }
        result['land_conversion_guard_counts'] = dict(cache.get('land_conversion_guard_counts', {}) or {})
        result['limit_reforestation_to_released_ag_land'] = bool(
            cache.get('limit_reforestation_to_released_ag_land', False)
        )
        result['prevent_land_conversion_cycles'] = bool(
            cache.get('prevent_land_conversion_cycles', False)
        )
        result['reforestation_physical_cap_enabled'] = bool(
            cache.get('reforestation_physical_cap_enabled', False)
        )
        result['reforestation_max_forest_increase_ratio'] = float(
            cache.get('reforestation_max_forest_increase_ratio', 0.0) or 0.0
        )
        result['cropland_demand_effective_by_region_year'] = {
            k: _land_expr_out(expr)
            for k, expr in (cache.get('cropland_demand_effective_expr_by_region_year', {}) or {}).items()
        }
        result['grassland_demand_effective_by_region_year'] = {
            k: _land_expr_out(expr)
            for k, expr in (cache.get('grassland_demand_effective_expr_by_region_year', {}) or {}).items()
        }
        result['luc_qs_forest_cap_lhs'] = {
            k: _land_expr_out(expr)
            for k, expr in (cache.get('luc_qs_forest_cap_expr_by_region_year', {}) or {}).items()
        }
        result['luc_qs_forest_cap_rhs'] = _land_dict_out(
            cache.get('luc_qs_forest_cap_rhs_by_region_year', {}) or {}
        )
        forest_target_rows: List[Dict[str, float]] = []
        target_map = cache.get('forest_global_target_by_year', {}) or {}
        actual_expr_map = cache.get('forest_global_actual_expr_by_year', {}) or {}
        shortfall_map = cache.get('forest_global_target_shortfall', {}) or {}
        surplus_map = cache.get('forest_global_target_surplus', {}) or {}
        forest_years = sorted(set(target_map) | set(actual_expr_map) | set(shortfall_map) | set(surplus_map))
        for t in forest_years:
            try:
                year_val = int(t)
            except Exception:
                year_val = t
            try:
                target_val = _land_out_value(target_map.get(t, 0.0) or 0.0)
            except Exception:
                target_val = np.nan
            try:
                actual_obj = actual_expr_map.get(t)
                actual_val = _land_expr_out(actual_obj)
            except Exception:
                actual_val = np.nan
            try:
                shortfall_obj = shortfall_map.get(t)
                shortfall_val = _land_var_out(shortfall_obj)
            except Exception:
                shortfall_val = 0.0
            try:
                surplus_obj = surplus_map.get(t)
                surplus_val = _land_var_out(surplus_obj)
            except Exception:
                surplus_val = 0.0
            abs_slack_val = max(0.0, shortfall_val) + max(0.0, surplus_val)
            denom = abs(target_val) if np.isfinite(target_val) and abs(target_val) > 1.0 else 1.0
            forest_target_rows.append({
                'year': year_val,
                'target_year': year_val,
                'target_forest_ha': target_val,
                'actual_forest_ha': actual_val,
                'shortfall_ha': shortfall_val,
                'surplus_ha': surplus_val,
                'abs_slack_ha': abs_slack_val,
                'slack_rate': abs_slack_val / denom,
            })
        if forest_target_rows:
            result['forest_global_target_slack'] = forest_target_rows
            if output_dir:
                try:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(forest_target_rows).to_csv(
                        Path(output_dir) / "forest_global_target_slack.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                except Exception:
                    pass
        result['grassland_to_cropland'] = {
            k: _land_var_out(v) for k, v in cache.get('grassland_to_cropland', {}).items()
        }
        result['grassland_to_forest'] = {
            k: _land_var_out(v) for k, v in cache.get('grassland_to_forest', {}).items()
        }
        result['grassland_to_othernat'] = {
            k: _land_var_out(v) for k, v in cache.get('grassland_to_othernat', {}).items()
        }
        result['cropland_to_grassland'] = {
            k: _land_var_out(v) for k, v in cache.get('cropland_to_grassland', {}).items()
        }
        result['cropland_to_forest'] = {
            k: _land_var_out(v) for k, v in cache.get('cropland_to_forest', {}).items()
        }
        result['cropland_to_othernat'] = {
            k: _land_var_out(v) for k, v in cache.get('cropland_to_othernat', {}).items()
        }
        result['forest_to_cropland'] = {
            k: _land_var_out(v) for k, v in cache.get('forest_to_cropland', {}).items()
        }
        result['forest_to_grassland'] = {
            k: _land_var_out(v) for k, v in cache.get('forest_to_grassland', {}).items()
        }
        result['background_forest_to_cropland'] = _land_dict_out(
            cache.get('background_forest_to_cropland', {}) or {}
        )
        result['background_forest_to_grassland'] = _land_dict_out(
            cache.get('background_forest_to_grassland', {}) or {}
        )
        result['background_cropland_to_forest'] = _land_dict_out(
            cache.get('background_cropland_to_forest', {}) or {}
        )
        result['background_grassland_to_forest'] = _land_dict_out(
            cache.get('background_grassland_to_forest', {}) or {}
        )
        result['nonforest_to_cropland'] = {
            k: _land_var_out(v) for k, v in cache.get('nonforest_to_cropland', {}).items()
        }
        result['nonforest_to_grassland'] = {
            k: _land_var_out(v) for k, v in cache.get('nonforest_to_grassland', {}).items()
        }
        result['land_anchor_clip_cropland_by_region'] = _land_dict_out(
            cache.get('land_anchor_clip_cropland_by_region', {}) or {}
        )
        result['land_anchor_clip_grassland_by_region'] = _land_dict_out(
            cache.get('land_anchor_clip_grassland_by_region', {}) or {}
        )
        
        # Abatement results
        abat_vars = cache.get('abatement_vars', {})
        result['abatement'] = {k: v.X for k, v in abat_vars.items()}
        for key, spec in (cache.get('zero_cost_abatement_specs', {}) or {}).items():
            baseline_value, current_expr = spec
            try:
                current_value = float(current_expr.getValue())
            except Exception:
                try:
                    current_value = float(current_expr)
                except Exception:
                    continue
            result['abatement'][key] = max(0.0, float(baseline_value) - current_value)
        for key, value in (cache.get('no_opportunity_abatement_specs', {}) or {}).items():
            result['abatement'][key] = float(value)

        strategy_abat_vars = cache.get('strategy_abatement_vars', {}) or {}
        result['strategy_abatement'] = {
            key: float(value.X) for key, value in strategy_abat_vars.items()
        }
        for key, spec in (cache.get('zero_cost_strategy_abatement_specs', {}) or {}).items():
            baseline_value, current_expr = spec
            try:
                current_value = float(current_expr.getValue())
            except Exception:
                try:
                    current_value = float(current_expr)
                except Exception:
                    continue
            result['strategy_abatement'][key] = max(
                0.0, float(baseline_value) - current_value
            )
        result['strategy_abatement_costs'] = dict(
            cache.get('strategy_abatement_costs', {}) or {}
        )
        result['abatement_costs'] = dict(cache.get('abatement_costs', {}) or {})
        result['abatement_database_keys'] = dict(
            cache.get('abatement_database_keys', {}) or {}
        )
        result['strategy_cost_metadata'] = dict(
            cache.get('strategy_cost_metadata', {}) or {}
        )
        result['cost_database_metadata'] = dict(
            cache.get('cost_database_metadata', {}) or {}
        )
        priced_abat_vars = cache.get('abatement_cost_vars', {})
        result['abatement_cost_vars'] = {
            k: v.X for k, v in priced_abat_vars.items()
        }

        if excluded_items:
            sep = _predict_excluded_commodities(
                nodes=nodes,
                excluded_commodities=excluded_items,
                years=years,
                dict_v3_path=dict_v3_path,
                population_by_country_year=population_by_country_year,
                income_mult_by_country_year=income_mult_by_country_year,
                hist_end_year=hist_end_year,
                demand_method=demand_method,
                nutrition_profile_xlsx=nutrition_profile_xlsx,
                nutrition_profile_sheet=nutrition_profile_sheet,
                nutrition_indicator=nutrition_indicator,
                nutrition_use_baseyear_for_future=nutrition_use_baseyear_for_future,
                tax_unit_adder=tax_unit_adder,
                feed_reduction_by=feed_reduction_by,
                waste_reduction_by=waste_reduction_by,
                losses_ratio_by=losses_ratio_by,
                use_relative_price=use_rel,
                relative_price_bounds=relative_price_bounds,
                price_bounds=cache.get('price_bounds', (1e-6, 1e6)),
                price_bounds_mode=cache.get('price_bounds_mode', price_bounds_mode),
                price_bounds_p0_mult=cache.get('price_bounds_p0_mult', price_bounds_p0_mult),
            )
            if sep:
                result['excluded_commodities'] = excluded_items
                result['Pc'].update(sep.get('Pc', {}))
                if use_rel:
                    result.setdefault('Pc_rel', {}).update(sep.get('Pc_rel', {}))
                result['Qs'].update(sep.get('Qs', {}))
                result['Qd'].update(sep.get('Qd', {}))
        
        # Summary statistics
        total_supply = sum(result['Qs'].values())
        total_demand = sum(result['Qd'].values())
        total_excess_slack = sum(result['excess'].values())
        total_shortage_slack = sum(result['shortage'].values())
        total_emissions = sum(result['Eij'].values())
        total_process_abat_cost = float(sum(result['Cij'].values()))
        result['strategy_cost_contributions'] = {
            key: float(quantity) * float(result['strategy_abatement_costs'].get(key, 0.0))
            for key, quantity in result.get('strategy_abatement', {}).items()
            if key in result['strategy_abatement_costs']
        }
        total_strategy_abat_cost = float(
            sum(result['strategy_cost_contributions'].values())
        )
        total_abat_cost = total_process_abat_cost + total_strategy_abat_cost
        result['process_abatement_cost_usd'] = total_process_abat_cost
        result['strategy_abatement_cost_usd'] = total_strategy_abat_cost
        result['total_abatement_cost_usd'] = total_abat_cost
        total_abatement = (
            sum(result['abatement'].values())
            + sum(result.get('strategy_abatement', {}).values())
        )

        solver_qs = {
            key: result['Qs'].get(key, 0.0)
            for key in (cache.get('Qs', {}) or {})
        }
        solver_qd = {
            key: result['Qd'].get(key, 0.0)
            for key in (cache.get('Qd', {}) or {})
        }
        market_balance_rows = build_market_balance_diagnostics(
            qs=solver_qs,
            qd=solver_qd,
            net_import=result.get('net_import', {}),
            bioenergy=result.get('bioenergy_crop_demand', {}),
            shortage=result.get('shortage', {}),
            excess=result.get('excess', {}),
            hist_end_year=hist_end_year,
        )
        result['market_balance_diagnostics'] = market_balance_rows
        result['market_shortage'] = total_shortage_slack
        result['market_excess'] = total_excess_slack
        # Compatibility keys now reflect the actual solver slack.  The raw
        # Qd-Qs difference is retained under an explicit, non-shortage name.
        result['implied_shortage'] = total_shortage_slack
        result['implied_excess'] = total_excess_slack

        raw_qd_minus_qs_positive = 0.0
        raw_qs_minus_qd_positive = 0.0
        shortage_by_comm_year: Dict[Tuple[str, int], float] = {}
        gap_samples: List[Tuple[float, str, int, float]] = []
        for row in market_balance_rows:
            j = str(row['commodity'])
            t = int(row['year'])
            raw_gap = float(row['raw_qd_minus_qs_t'])
            shortage_val = float(row['shortage_t'])
            shortage_by_comm_year[(j, t)] = shortage_val
            if raw_gap > 0.0:
                raw_qd_minus_qs_positive += raw_gap
            elif raw_gap < 0.0:
                raw_qs_minus_qd_positive += -raw_gap
            if raw_gap != 0.0:
                gap_samples.append((abs(raw_gap), j, t, raw_gap))
        result['raw_qd_minus_qs_positive_total'] = raw_qd_minus_qs_positive
        result['raw_qs_minus_qd_positive_total'] = raw_qs_minus_qd_positive
        
        # Calculate energy shortage when nutrient coefficients are available.
        nutrient_dict = cache.get('nutrient_per_unit_by_comm', {})
        total_energy_supply = 0.0
        total_energy_shortage = 0.0
        if nutrient_dict:
            for (r, j, t), val in result['Qs'].items():
                kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                if kcal_per_ton > 0:
                    total_energy_supply += kcal_per_ton * val
            
            for (j, t), shortage_val in shortage_by_comm_year.items():
                kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                if kcal_per_ton > 0 and shortage_val > 0:
                    total_energy_shortage += kcal_per_ton * shortage_val

        total_energy_demand = 0.0
        total_energy_demand_pos = 0.0
        total_energy_demand_neg = 0.0
        energy_demand_by_year: Dict[int, float] = {}
        energy_demand_pos_by_year: Dict[int, float] = {}
        energy_demand_neg_by_year: Dict[int, float] = {}
        if nutrient_dict:
            for (r, j, t), qd_val in result['Qd'].items():
                kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                if kcal_per_ton <= 0:
                    continue
                energy = kcal_per_ton * float(qd_val)
                energy_demand_by_year[t] = energy_demand_by_year.get(t, 0.0) + energy
                total_energy_demand += energy
                if qd_val >= 0:
                    energy_demand_pos_by_year[t] = energy_demand_pos_by_year.get(t, 0.0) + energy
                    total_energy_demand_pos += energy
                else:
                    energy_demand_neg_by_year[t] = energy_demand_neg_by_year.get(t, 0.0) + energy
                    total_energy_demand_neg += energy
            result['energy_demand_by_year'] = energy_demand_by_year
            result['energy_demand_pos_by_year'] = energy_demand_pos_by_year
            result['energy_demand_neg_by_year'] = energy_demand_neg_by_year
            result['energy_demand_total'] = total_energy_demand
            result['energy_demand_pos_total'] = total_energy_demand_pos
            result['energy_demand_neg_total'] = total_energy_demand_neg
        
        energy_shortage_ratio = (total_energy_shortage / total_energy_supply * 100) if total_energy_supply > 0 else 0
        
        if enable_verbose_logging:
            logger.info(f"[LINEAR] 总供给={total_supply:.2e} kt (跨{len(result['Qs'])}个区域-商品-年份), 总需求={total_demand:.2e} kt")
            logger.info(f"[LINEAR] 总过剩(slack)={total_excess_slack:.2e} kt (跨{len(result['excess'])}个商品-年份), 总短缺(slack)={total_shortage_slack:.2e} kt")
            logger.info(
                "[LINEAR] raw Qs-Qd positive=%.2e kt, raw Qd-Qs positive=%.2e kt "
                "(diagnostic only; includes trade/bioenergy effects)",
                raw_qs_minus_qd_positive,
                raw_qd_minus_qs_positive,
            )
            if gap_samples:
                gap_samples.sort(key=lambda x: x[0], reverse=True)
                for _, j, t, gap in gap_samples[:10]:
                    logger.info(
                        "[LINEAR] gap_sample commodity=%s year=%s gap=%.6e (Qd-Qs)",
                        j,
                        t,
                        gap,
                    )
            if nutrient_dict:
                logger.info(f"[LINEAR] 能量供给={total_energy_supply:.2e} kcal, 能量短缺={total_energy_shortage:.2e} kcal ({energy_shortage_ratio:.1f}%)")
                logger.info(
                    "[LINEAR] 能量需求(raw)=%.2e kcal, 能量需求(pos)=%.2e kcal, 能量需求(neg)=%.2e kcal",
                    total_energy_demand,
                    total_energy_demand_pos,
                    total_energy_demand_neg,
                )
                for t in sorted(energy_demand_by_year.keys()):
                    if t <= hist_end_year:
                        continue
                    raw = energy_demand_by_year.get(t, 0.0)
                    pos = energy_demand_pos_by_year.get(t, 0.0)
                    neg = energy_demand_neg_by_year.get(t, 0.0)
                    if raw <= 0 or pos <= 0:
                        logger.warning(
                            "[LINEAR] energy_demand_year t=%s raw=%.2e pos=%.2e neg=%.2e",
                            t,
                            raw,
                            pos,
                            neg,
                        )
            logger.info(f"[LINEAR] 总排放={total_emissions:.2e} tCO2e (跨{len(result['Eij'])}个区域-商品-年份), 总减排={total_abatement:.2e} tCO2e, 减排成本={total_abat_cost:.2e} USD")
            
        # If total shortage is unusually large, report the ten commodities with the largest shortages.
        if enable_verbose_logging and total_shortage_slack > 1e6:  # Shortage > 1 Mt
            # Calculate shortage as a fraction of total supply by mass.
            total_market_use = sum(
                float(row.get('market_total_use_t', 0.0) or 0.0)
                for row in market_balance_rows
            )
            shortage_ratio = (
                total_shortage_slack / total_market_use * 100
                if total_market_use > 0
                else 0
            )
            
            shortage_by_comm = {}
            supply_by_comm = {}
            energy_shortage_by_comm = {}
            energy_supply_by_comm = {}
            
            for (j, t), shortage_val in shortage_by_comm_year.items():
                if shortage_val > 0:
                    if j not in shortage_by_comm:
                        shortage_by_comm[j] = 0
                    shortage_by_comm[j] += shortage_val
                    
                    # Calculate energy shortage.
                    if nutrient_dict:
                        kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                        if kcal_per_ton > 0:
                            if j not in energy_shortage_by_comm:
                                energy_shortage_by_comm[j] = 0
                            energy_shortage_by_comm[j] += kcal_per_ton * shortage_val
            
            # Also total supply by commodity.
            for (r, j, t), val in result['Qs'].items():
                if j not in supply_by_comm:
                    supply_by_comm[j] = 0
                supply_by_comm[j] += val
                
                # Calculate energy supply.
                if nutrient_dict:
                    kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                    if kcal_per_ton > 0:
                        if j not in energy_supply_by_comm:
                            energy_supply_by_comm[j] = 0
                        energy_supply_by_comm[j] += kcal_per_ton * val
            
            if shortage_by_comm:
                top_shortages = sorted(shortage_by_comm.items(), key=lambda x: x[1], reverse=True)[:10]
                logger.warning(f"[LINEAR]  总短缺={total_shortage_slack:.2e} ({shortage_ratio:.1f}%重量, {energy_shortage_ratio:.1f}%能量) 异常大！前10大短缺商品:")
                for j, val in top_shortages:
                    supply_j = supply_by_comm.get(j, 0)
                    ratio_j = (val / supply_j * 100) if supply_j > 0 else 0
                    
                    # Energy share
                    energy_short_j = energy_shortage_by_comm.get(j, 0)
                    energy_supply_j = energy_supply_by_comm.get(j, 0)
                    energy_ratio_j = (energy_short_j / energy_supply_j * 100) if energy_supply_j > 0 else 0
                    
                    if nutrient_dict and energy_short_j > 0:
                        logger.warning(f"  - {j}: 短缺={val:.2e} kt ({ratio_j:.1f}%重量, {energy_ratio_j:.1f}%能量)")
                    else:
                        logger.warning(f"  - {j}: 短缺={val:.2e} kt, 供给={supply_j:.2e} kt, 占比={ratio_j:.1f}%")
    
    if status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT):
        # Post-solve violation check (unscaled)
        if post_solve_violation_tol is not None:
            try:
                tol_val = float(post_solve_violation_tol)
            except Exception:
                tol_val = None
            if tol_val is not None and tol_val > 0:
                max_violation, top_violations = _compute_max_violation(
                    m,
                    top_n=post_solve_violation_top_n,
                )
                result['max_violation'] = max_violation
                if top_violations:
                    result['violation_top'] = top_violations
                # Explicit fail on nutrition constraint violations
                nutri_max = 0.0
                nutri_top: List[Dict[str, Any]] = []
                for c in m.getConstrs():
                    name = c.ConstrName
                    if not name.startswith("nutri["):
                        continue
                    sense = c.Sense
                    rhs = float(c.RHS)
                    slack = float(c.Slack)
                    if sense == '=':
                        violation = abs(slack)
                        lhs = rhs - slack
                    elif sense == '<':
                        violation = max(-slack, 0.0)
                        lhs = rhs - slack
                    elif sense == '>':
                        violation = max(slack, 0.0)
                        lhs = rhs - slack
                    else:
                        continue
                    if violation > nutri_max:
                        nutri_max = violation
                    if violation > 0:
                        nutri_top.append({
                            'name': name,
                            'type': 'nutri',
                            'sense': sense,
                            'lhs': lhs,
                            'rhs': rhs,
                            'violation': violation,
                        })
                if nutri_top:
                    nutri_top.sort(key=lambda x: x['violation'], reverse=True)
                    result['nutrition_violation_top'] = nutri_top[:post_solve_violation_top_n]
                    result['nutrition_violation_max'] = nutri_max
                if nutri_max > tol_val:
                    result['status_raw'] = status
                    result['status'] = gp.GRB.INFEASIBLE
                    status = result['status']
                    logger.error(
                        "[LINEAR] Nutrition violation %.6e exceeds tol %.6e; mark as INFEASIBLE",
                        nutri_max,
                        tol_val,
                    )
                    for item in (nutri_top[:10] if nutri_top else []):
                        logger.error(
                            "[LINEAR] nutri violation: %s sense=%s lhs=%.6e rhs=%.6e viol=%.6e",
                            item.get('name'),
                            item.get('sense'),
                            float(item.get('lhs', 0.0)),
                            float(item.get('rhs', 0.0)),
                            float(item.get('violation', 0.0)),
                        )
                    if enable_violation_iis:
                        iis_path = _write_violation_iis(m, output_dir, tol_val)
                        if iis_path:
                            result['violation_iis_path'] = str(iis_path)
                if max_violation > tol_val and status != gp.GRB.INFEASIBLE:
                    result['status_raw'] = status
                    result['status'] = gp.GRB.NUMERIC
                    status = result['status']
                    logger.error(
                        "[LINEAR] Post-solve violation %.6e exceeds tol %.6e; mark as NUMERIC",
                        max_violation,
                        tol_val,
                    )
                    for item in top_violations[:10]:
                        logger.error(
                            "[LINEAR] violation top: %s type=%s sense=%s lhs=%.6e rhs=%.6e viol=%.6e",
                            item.get('name'),
                            item.get('type'),
                            item.get('sense'),
                            float(item.get('lhs', 0.0)),
                            float(item.get('rhs', 0.0)),
                            float(item.get('violation', 0.0)),
                        )
                    if enable_violation_iis:
                        iis_path = _write_violation_iis(m, output_dir, tol_val)
                        if iis_path:
                            result['violation_iis_path'] = str(iis_path)

    cache = getattr(m, '_nzf_cache', {}) or {}
    result['idx'] = cache.get('idx', {})
    result['regions'] = cache.get('regions', [])
    result['years'] = cache.get('years', [])

    if output_dir and enable_output_diagnostics:
        try:
            hist_diag_year = int(cache.get('hist_end_year', hist_end_year))
        except Exception:
            hist_diag_year = hist_end_year
        nutri_path = _write_nutrition_feasibility_diagnosis(
            cache,
            output_dir,
            nutrition_rhs=nutrition_rhs,
            nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
            feed_reduction_by=feed_reduction_by,
            nutrition_demand_map=cache.get('nutrition_demand_map'),
            hist_end_year=hist_diag_year,
        )
        if nutri_path:
            logger.info(f"[LINEAR] nutrition feasibility diagnosis written: {nutri_path}")
        gap_path = _write_nutrition_gap_by_model_coverage(
            cache,
            output_dir,
            nutrition_rhs=nutrition_rhs,
            nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
            nutrition_demand_map=cache.get('nutrition_demand_map'),
            hist_end_year=hist_diag_year,
        )
        if gap_path:
            logger.info(f"[LINEAR] nutrition gap by model coverage written: {gap_path}")
        forest_path = _write_forest_nonneg_feasibility_diagnosis(
            cache,
            output_dir,
            hist_end_year=hist_diag_year,
        )
        if forest_path:
            logger.info(f"[LINEAR] forest_nonneg feasibility diagnosis written: {forest_path}")
        try:
            if m.SolCount > 0:
                years = cache.get('years', []) or []
                diag_years = [max([y for y in years if int(y) > hist_diag_year])] if years else None
                resid_path = _write_constraint_residuals(m, output_dir)
                if resid_path:
                    logger.info(f"[LINEAR] constraint residuals written: {resid_path}")
                qd_path = _write_demand_equation_check(m, output_dir, years_filter=diag_years)
                if qd_path:
                    logger.info(f"[LINEAR] demand equation check written: {qd_path}")
                qs_path = _write_supply_equation_check(m, output_dir, years_filter=diag_years)
                if qs_path:
                    logger.info(f"[LINEAR] supply equation check written: {qs_path}")
                bc_path = _write_balance_clear_diagnosis(m, output_dir, years_filter=diag_years)
                if bc_path:
                    logger.info(f"[LINEAR] balance/clear diagnosis written: {bc_path}")
                sc_path = _write_supply_cross_terms_diagnosis(m, output_dir, years_filter=diag_years)
                if sc_path:
                    logger.info(f"[LINEAR] supply cross terms diagnosis written: {sc_path}")
        except Exception as exc:
            logger.warning(f"[LINEAR] equation diagnostics failed: {exc}")

    if result.get('Qd'):
        neg_qd = [(key, val) for key, val in result['Qd'].items() if float(val) < -1e-6]
        if neg_qd:
            neg_qd.sort(key=lambda x: x[1])
            result['neg_qd_count'] = len(neg_qd)
            result['neg_qd_min'] = float(neg_qd[0][1])
            logger.error(
                "[Qd-NEG] count=%s min=%.6e",
                len(neg_qd),
                float(neg_qd[0][1]),
            )
            total_qd_pos = sum(float(val) for val in result['Qd'].values() if float(val) > 0)
            total_qd_neg = sum(-float(val) for val in result['Qd'].values() if float(val) < 0)
            neg_ratio = (total_qd_neg / total_qd_pos * 100.0) if total_qd_pos > 0 else 0.0
            logger.error(
                "[Qd-NEG] total_neg=%.6e total_pos=%.6e neg_ratio=%.2f%%",
                total_qd_neg,
                total_qd_pos,
                neg_ratio,
            )
            neg_by_comm: Dict[str, float] = {}
            neg_by_region: Dict[str, float] = {}
            neg_by_year: Dict[int, float] = {}
            for (r, j, t), qd_val in neg_qd:
                neg_val = -float(qd_val)
                neg_by_comm[j] = neg_by_comm.get(j, 0.0) + neg_val
                neg_by_region[r] = neg_by_region.get(r, 0.0) + neg_val
                neg_by_year[t] = neg_by_year.get(t, 0.0) + neg_val
            top_comm = sorted(neg_by_comm.items(), key=lambda x: x[1], reverse=True)[:10]
            if top_comm:
                logger.error(
                    "[Qd-NEG] top_comm=%s",
                    ", ".join(f"{k}:{v:.2e}" for k, v in top_comm),
                )
            top_region = sorted(neg_by_region.items(), key=lambda x: x[1], reverse=True)[:10]
            if top_region:
                logger.error(
                    "[Qd-NEG] top_region=%s",
                    ", ".join(f"{k}:{v:.2e}" for k, v in top_region),
                )
            top_year = sorted(neg_by_year.items(), key=lambda x: x[1], reverse=True)[:10]
            if top_year:
                logger.error(
                    "[Qd-NEG] top_year=%s",
                    ", ".join(f"{k}:{v:.2e}" for k, v in top_year),
                )
            idx_map = cache.get('idx', {}) or {}
            alpha_d_map = cache.get('alpha_d', {}) or {}
            eps_d_map = cache.get('eps_d', {}) or {}
            d0_map = cache.get('D0', {}) or {}
            p0_map = cache.get('P0', {}) or {}
            cross_terms_top_n = cache.get('cross_terms_top_n')
            cross_terms_scale = cache.get('cross_terms_scale')
            for (r, j, t), qd_val in neg_qd[:10]:
                data = idx_map.get((r, j, t), {}) or {}
                epsD_row_raw = {k: v for k, v in (data.get('epsD_row', {}) or {}).items() if k in cache.get('commodities', [])}
                eps_d_val = float(eps_d_map.get((r, j, t), 0.0) or 0.0)
                epsD_row_scaled = _scale_cross_terms(epsD_row_raw, cross_terms_scale)
                epsD_row_limited = _limit_cross_terms(epsD_row_scaled, cross_terms_top_n)
                _, sum_cross_eps, _ = _normalize_cross_eps(epsD_row_limited, eps_d_val)
                coeff = 1.0 - eps_d_val - float(sum_cross_eps)
                logger.error(
                    "[Qd-NEG] r=%s j=%s t=%s Qd=%.6e a_d=%.6e eps_d=%.6e sum_cross_eps=%.6e coeff=%.6e D0=%.6e P0=%.6e",
                    r,
                    j,
                    t,
                    float(qd_val),
                    float(alpha_d_map.get((r, j, t), 0.0) or 0.0),
                    eps_d_val,
                    float(sum_cross_eps),
                    coeff,
                    float(d0_map.get((r, j, t), 0.0) or 0.0),
                    float(p0_map.get((r, j, t), 0.0) or 0.0),
                )


    try:
        sol_count = int(getattr(m, 'SolCount', 0) or 0)
    except Exception:
        sol_count = 0
    try:
        runtime_seconds = float(getattr(m, 'Runtime', 0.0) or 0.0)
    except Exception:
        runtime_seconds = 0.0
    solver_metadata = normalize_solver_status(
        result.get('status'),
        sol_count=sol_count,
        has_solution=sol_count > 0,
    )
    result['status_name'] = solver_metadata['status_name']
    result['sol_count'] = sol_count
    result['has_solution'] = bool(solver_metadata['has_solution'])
    result['runtime_seconds'] = runtime_seconds
    return result



# Monte Carlo simulation support


def build_linear_model_cache(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    dict_v3_path: Optional[str] = None,
    use_relative_price: bool = False,
    relative_price_bounds: Tuple[float, float] = (0.1, 10.0),
    price_bounds_mode: str = 'absolute',
    price_bounds_p0_mult: Tuple[float, float] = (0.1, 10.0),
    price_bounds: Tuple[float, float] = (1e-6, 1e6),
    price_wedge_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    price_wedge_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    price_wedge_by_region: Optional[Dict[str, float]] = None,
    qty_scale: float = 1.0,
    # Population and income
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    # Emissions and abatement parameters
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    # Constraint parameters
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    land_soft_constraints_enabled: bool = False,
    land_slack_max_rate: Optional[float] = None,
    land_slack_penalty: Optional[float] = None,
    yield_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    hist_max_production: Optional[Dict[Tuple[str, str], float]] = None,
    grassland_method: str = 'dynamic',
    grassland_conversion_penalty: float = 0.0,
    grassland_to_cropland_cost_mode: str = 'per_ha_cost',
    cropland_to_grassland_penalty: float = 0.0,
    land_conversion_allocation_mode: str = 'priority_nonforest_pasture_forest',
    land_conversion_priority_penalty_per_ha: float = 1e6,
    land_priority_weight_grassland_to_cropland: float = 1.0,
    land_priority_weight_forest_to_cropland: float = 100.0,
    land_priority_weight_forest_to_grassland: float = 100.0,
    limit_reforestation_to_released_ag_land: bool = True,
    prevent_land_conversion_cycles: bool = True,
    reforestation_physical_cap_enabled: bool = True,
    reforestation_max_forest_increase_ratio: Optional[float] = 0.30,
    cross_terms_top_n: Optional[int] = None,
    cross_terms_scale: Optional[float] = None,
    disable_production_cost_term: bool = True,
    production_cost_weight: float = 1.0,
    slack_penalty: Optional[float] = 1e6,
    supply_curtailment_enabled: bool = False,
    supply_curtailment_penalty: Optional[float] = None,
    zero_price_shutdown_enabled: bool = False,
) -> LinearModelCache:
    """
    Build the linear regional model and return its cache for Monte Carlo simulation.
    
    Similar to build_model_cache() in S3_0_ds_emis_mc_full.py:
    1. Build the full linear regional model.
    2. Extract all variables, constraints, and calibration parameters.
    3. Return a LinearModelCache object.
    
    Usage:
        cache = build_linear_model_cache(nodes, commodities, years, ...)
        apply_linear_sample_updates(cache, pop_mult=..., yield_mult=..., e0_mult=...)
        cache.model.optimize()
        # Read results: cache.Qs[key].X, cache.Qd[key].X, ...
    
    Parameters:
        nodes: List of country-level nodes.
        commodities: List of commodities.
        years: List of years.
        dict_v3_path: Path to dict_v3.xlsx.
        population_by_country_year: Population data.
        income_mult_by_country_year: Income multiplier data.
        macc_path: Path to MACC data.
        land_carbon_price_by_year: Land carbon price.
        nutrition_rhs: Nutrition constraint RHS.
        nutrient_per_unit_by_comm: Nutrition per commodity unit.
        land_area_limits: Land area caps.
        yield_t_per_ha_default: Default yield.
        max_growth_rate_per_period: Maximum growth rate.
        max_decline_rate_per_period: Maximum decline rate.
        hist_end_year: Last historical year.
        hist_max_production: Historical production anchor {(region, commodity): max_t}.
    
    Returns:
        LinearModelCache object.
    """
    logger = logging.getLogger(__name__)
    
    # Build the model.
    m = build_linear_regional_model(
        nodes=nodes,
        commodities=commodities,
        years=years,
        dict_v3_path=dict_v3_path,
        use_relative_price=use_relative_price,
        relative_price_bounds=relative_price_bounds,
        price_bounds_mode=price_bounds_mode,
        price_bounds_p0_mult=price_bounds_p0_mult,
        price_bounds=price_bounds,
        price_wedge_by_region_comm_year=price_wedge_by_region_comm_year,
        price_wedge_by_region_comm=price_wedge_by_region_comm,
        price_wedge_by_region=price_wedge_by_region,
        qty_scale=qty_scale,
        population_by_country_year=population_by_country_year,
        income_mult_by_country_year=income_mult_by_country_year,
        macc_path=macc_path,
        land_carbon_price_by_year=land_carbon_price_by_year,
        nutrition_rhs=nutrition_rhs,
        nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
        land_area_limits=land_area_limits,
        land_soft_constraints_enabled=land_soft_constraints_enabled,
        land_slack_max_rate=land_slack_max_rate,
        land_slack_penalty=land_slack_penalty,
        yield_by_region_comm_year=yield_by_region_comm_year,
        yield_t_per_ha_default=yield_t_per_ha_default,
        max_growth_rate_per_period=max_growth_rate_per_period,
        max_decline_rate_per_period=max_decline_rate_per_period,
        hist_end_year=hist_end_year,
        hist_max_production=hist_max_production,
        grassland_method=grassland_method,
        grassland_conversion_penalty=grassland_conversion_penalty,
        grassland_to_cropland_cost_mode=grassland_to_cropland_cost_mode,
        cropland_to_grassland_penalty=cropland_to_grassland_penalty,
        land_conversion_allocation_mode=land_conversion_allocation_mode,
        land_conversion_priority_penalty_per_ha=land_conversion_priority_penalty_per_ha,
        land_priority_weight_grassland_to_cropland=land_priority_weight_grassland_to_cropland,
        land_priority_weight_forest_to_cropland=land_priority_weight_forest_to_cropland,
        land_priority_weight_forest_to_grassland=land_priority_weight_forest_to_grassland,
        limit_reforestation_to_released_ag_land=limit_reforestation_to_released_ag_land,
        prevent_land_conversion_cycles=prevent_land_conversion_cycles,
        reforestation_physical_cap_enabled=reforestation_physical_cap_enabled,
        reforestation_max_forest_increase_ratio=reforestation_max_forest_increase_ratio,
        cross_terms_top_n=cross_terms_top_n,
        cross_terms_scale=cross_terms_scale,
        disable_production_cost_term=disable_production_cost_term,
        production_cost_weight=production_cost_weight,
        slack_penalty=slack_penalty,
        supply_curtailment_enabled=supply_curtailment_enabled,
        supply_curtailment_penalty=supply_curtailment_penalty,
        zero_price_shutdown_enabled=zero_price_shutdown_enabled,
    )
    
    # Extract the cache.
    c = m._nzf_cache
    
    # Construct LinearModelCache.
    cache = LinearModelCache(
        model=m,
        # Variables
        Pc=c['Pc'],
        Qs=c['Qs'],
        Qd=c['Qd'],
        supply_curtailment=c.get('supply_curtailment', {}),
        net_import=c.get('net_import', {}),
        armington_slack_pos=c.get('armington_slack_pos', {}),
        armington_slack_neg=c.get('armington_slack_neg', {}),
        Eij=c['Eij'],
        Cij=c['Cij'],
        excess=c['excess'],
        shortage=c['shortage'],
        # Constraint references
        constr_supply=c['constr_supply'],
        constr_demand=c['constr_demand'],
        constr_Edef=c['constr_Edef'],
        nutri_constr=c['nutri_constr'],
        land_constr=c['land_constr'],
        rumi_intake_constr=c.get('rumi_intake_constr', {}),  # Phase 2: ruminant demand cap constraints
        # MACC
        abatement_vars=c['abatement_vars'],
        abatement_caps=c['abatement_caps'],
        abatement_cost_vars=c.get('abatement_cost_vars', {}),
        abatement_cost_caps=c.get('abatement_cost_caps', {}),
        abatement_req_vars=c.get('abatement_req_vars', {}),
        abatement_req_constr=c.get('abatement_req_constr', {}),
        abatement_costs=c['abatement_costs'],
        proc_cap_basecoeff=c['proc_cap_basecoeff'],
        # Calibration parameters
        alpha_s=c['alpha_s'],
        alpha_d=c['alpha_d'],
        eps_s=c['eps_s'],
        eps_d=c['eps_d'],
        eps_pop=c['eps_pop'],
        eps_inc=c['eps_inc'],
        eta_y=c['eta_y'],
        eta_temp=c['eta_temp'],
        Q0=c['Q0'],
        D0=c['D0'],
        P0=c['P0'],
        Ymult0=c['Ymult0'],
        Tmult0=c['Tmult0'],
        pop_base=c['pop_base'],
        inc_base=c['inc_base'],
        disable_production_cost_term=bool(c.get('disable_production_cost_term', False)),
        production_cost_weight=float(c.get('production_cost_weight', 0.0) or 0.0),
        slack_penalty=float(c.get('slack_penalty', 1e6) or 1e6),
        armington_trade_slack_penalty=float(c.get('armington_trade_slack_penalty', 1e6) or 1e6),
        supply_curtailment_penalty=float(c.get('supply_curtailment_penalty', 0.0) or 0.0),
        supply_curtailment_enabled=bool(c.get('supply_curtailment_enabled', False)),
        # Emissions intensity
        e0_by_region=c['e0_by_region'],
        # Metadata
        regions=c['regions'],
        commodities=c['commodities'],
        years=c['years'],
        hist_end_year=int(c.get('hist_end_year', hist_end_year) or hist_end_year),
        idx=c['idx'],
        nutrition_import_pos=c.get('nutrition_import_pos', {}),
        nutrition_export_pos=c.get('nutrition_export_pos', {}),
        nutrition_supply_driven=bool(c.get('nutrition_supply_driven', False)),
        nutrition_trade_penalty=float(c.get('nutrition_trade_penalty', 0.0) or 0.0),
        nutrition_export_headroom_penalty=float(c.get('nutrition_export_headroom_penalty', 0.0) or 0.0),
        nutrition_export_headroom_weight_by_key=c.get('nutrition_export_headroom_weight_by_key', {}),
        tax_unit_adder=c.get('tax_unit_adder', {}),
    )
    
    logger.info(f"[LINEAR_MC] 模型缓存构建完成: 变量={m.NumVars}, 约束={m.NumConstrs}")
    logger.info(f"[LINEAR_MC] 区域数={len(c['regions'])}, 商品数={len(c['commodities'])}, 年份数={len(c['years'])}")
    
    return cache


def apply_linear_sample_updates(
    cache: LinearModelCache,
    *,
    pop_mult_by_region: Optional[Dict[str, float]] = None,
    income_mult_by_region: Optional[Dict[str, float]] = None,
    yield_mult_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    temp_mult_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    e0_mult_by_region_comm_proc: Optional[Dict[Tuple[str, str, int, str], float]] = None,
    land_cp_by_year: Optional[Dict[int, float]] = None,
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    land_limits: Optional[Dict[Tuple[str, int], float]] = None,
) -> None:
    """
    Update linear-model constraints in place for Monte Carlo sampling.
    
    Similar to apply_sample_updates() in S3_0_ds_emis_mc_full.py, but for the linear model.
    Linear model constraints take the form:
    
    Supply: Qs = a_s + b_s * Pc + sum(b_sj * Pc_j).
    Demand: Qd = a_d + b_d * Pc + sum(b_dj * Pc_j).
    
    Here a_s and a_d incorporate yield, temperature, population, and income effects.
    MC updates modify these intercepts.
    
    Update rules:
    1. Population multiplier: update demand intercept a_d (RHS).
    2. Income multiplier: update demand intercept a_d (RHS).
    3. Yield multiplier: update supply intercept a_s (RHS).
    4. Temperature multiplier: update supply intercept a_s (RHS).
    5. Emission-factor multiplier: update the Qs coefficient in E_def.
    
    Parameters:
        cache: LinearModelCache object.
        pop_mult_by_region: Population multipliers {region: mult}, relative to the base period.
        income_mult_by_region: Income multipliers {region: mult}, relative to the base period.
        yield_mult_by_region_comm: Yield multipliers {(region, commodity): mult}.
        temp_mult_by_region_comm: Temperature multipliers {(region, commodity): mult}.
        e0_mult_by_region_comm_proc: Emission-factor multipliers {(region, commodity, year, process): mult}.
        land_cp_by_year: Land carbon prices {year: price}.
        nutrition_rhs: Nutrition constraint RHS {(region, year): value}.
        land_limits: Land constraints {(region, year): value}.
    """
    from mc_sample_utils import validated_multipliers

    pop_mult_by_region = validated_multipliers(pop_mult_by_region, name='population')
    income_mult_by_region = validated_multipliers(income_mult_by_region, name='income')
    yield_mult_by_region_comm = validated_multipliers(yield_mult_by_region_comm, name='yield')
    temp_mult_by_region_comm = validated_multipliers(temp_mult_by_region_comm, name='temperature')
    e0_mult_by_region_comm_proc = validated_multipliers(e0_mult_by_region_comm_proc, name='EF')
    m = cache.model
    cache_meta = getattr(m, '_nzf_cache', {}) or {}
    _clear_land_source_selection(m)
    hist_end_year = int(getattr(cache, 'hist_end_year', cache_meta.get('hist_end_year', 2020) or 2020))
    cross_terms_top_n = cache_meta.get('cross_terms_top_n')
    cross_terms_scale = cache_meta.get('cross_terms_scale')
    try:
        qty_scale = float(cache_meta.get('qty_scale', 1.0) or 1.0)
    except Exception:
        qty_scale = 1.0
    if not np.isfinite(qty_scale) or qty_scale <= 0:
        qty_scale = 1.0
    inv_qty_scale = 1.0 / qty_scale
    use_relative_price = bool(cache_meta.get('use_relative_price'))
    price_ref_by_comm = cache_meta.get('price_ref_by_comm', {}) or {}
    price_wedge_by_region_comm_year = cache_meta.get('price_wedge_by_region_comm_year', {}) or {}
    price_wedge_by_region_comm = cache_meta.get('price_wedge_by_region_comm', {}) or {}
    price_wedge_by_region = cache_meta.get('price_wedge_by_region', {}) or {}

    def _price_wedge_local(region: Any, commodity: Any, year: Any) -> float:
        return _get_price_wedge(
            region,
            commodity,
            year,
            by_rcy=price_wedge_by_region_comm_year,
            by_rc=price_wedge_by_region_comm,
            by_r=price_wedge_by_region,
        )
    
    
    # 1. Update supply constraints using yield and temperature multipliers.
    
    # Linear supply: Qs = Q0 * (1 + eta_y*(Ymult-1) + eta_temp*(Tmult-1) - epsilon_s) + epsilon_s*Q0/P0*Pc.
    # + Σ[ε_sj*Q0/P0j*(Pj-P0j)]
    # MC yield update: Ymult_new = Ymult0 * yield_mult.
    # The new intercept subtracts baseline offsets only for cross terms actually included in the equation.
    
    if yield_mult_by_region_comm is not None or temp_mult_by_region_comm is not None:
        for key, con in cache.constr_supply.items():
            r, j, t = key
            
            Q0 = cache.Q0.get(key, 1.0)
            P0 = cache.P0.get(key, 1.0)
            eps_s = cache.eps_s.get(key, 0.0)
            eta_y = cache.eta_y.get(key, 0.0)
            eta_temp = cache.eta_temp.get(key, 0.0)
            Ymult0 = cache.Ymult0.get(key, 1.0)
            Tmult0 = cache.Tmult0.get(key, 1.0)
            
            # Get multipliers.
            y_mult = 1.0
            t_mult = 1.0
            if yield_mult_by_region_comm:
                y_mult = yield_mult_by_region_comm.get((r, j), 1.0)
            if temp_mult_by_region_comm:
                t_mult = temp_mult_by_region_comm.get((r, j), 1.0)
            
            # New yield and temperature multipliers
            Ymult_new = Ymult0 * y_mult
            Tmult_new = Tmult0 * t_mult
            
            # New intercept
            yield_adj_new = eta_y * (Ymult_new - 1.0)
            temp_adj_new = eta_temp * (Tmult_new - 1.0)
            
            # Get original cross-price elasticities.
            idx_data = cache.idx.get(key, {})
            epsS_row_raw = {k: v for k, v in (idx_data.get('epsS_row', {}) or {}).items() if k in cache.commodities}
            epsS_row_scaled = _scale_cross_terms(epsS_row_raw, cross_terms_scale)
            epsS_row = _limit_cross_terms(epsS_row_scaled, cross_terms_top_n)

            P0 = max(1e-6, float(P0 or 0.0))
            b_s_abs = Q0 * eps_s / P0
            b_s = b_s_abs
            if use_relative_price:
                price_ref_self = float(price_ref_by_comm.get(j, 1.0) or 1.0)
                if not np.isfinite(price_ref_self) or price_ref_self <= 0:
                    price_ref_self = 1.0
                b_s = b_s_abs * price_ref_self
            COEFF_THRESHOLD = 1e-6 * inv_qty_scale
            if abs(b_s) < COEFF_THRESHOLD:
                b_s = 0.0

            price_wedge_self = _price_wedge_local(r, j, t)
            cross_wedge_shift = 0.0
            cross_const_offset_s = 0.0
            if epsS_row:
                for other_comm, cross_eps in epsS_row.items():
                    if other_comm == j:
                        continue
                    other_base_key = (r, other_comm, hist_end_year)
                    P0_other = cache.idx.get(other_base_key, {}).get('P0')
                    if P0_other is None:
                        other_key = (r, other_comm, t)
                        P0_other = cache.idx.get(other_key, {}).get('P0', P0) or P0
                    P0_other = max(1e-6, float(P0_other))
                    b_cross_abs = Q0 * float(cross_eps) / P0_other
                    b_cross = b_cross_abs
                    pc_other_base_for_offset = P0_other
                    if use_relative_price:
                        price_ref_other = float(price_ref_by_comm.get(other_comm, 1.0) or 1.0)
                        if not np.isfinite(price_ref_other) or price_ref_other <= 0:
                            price_ref_other = 1.0
                        b_cross = b_cross_abs * price_ref_other
                        pc_other_base_for_offset = P0_other / price_ref_other
                    b_cross = _sanitize_cross_coef(
                        b_cross,
                        base_qty=Q0,
                        cross_eps=float(cross_eps),
                        use_relative_price=use_relative_price,
                        ratio_tol=CROSS_COEF_RATIO_TOL,
                    )
                    if abs(b_cross) < COEFF_THRESHOLD:
                        continue
                    cross_const_offset_s += b_cross * pc_other_base_for_offset
                    other_wedge = _price_wedge_local(r, other_comm, t)
                    if other_wedge:
                        cross_wedge_shift += b_cross * other_wedge
            
            # New intercept
            a_s_new = Q0 * (1.0 + yield_adj_new + temp_adj_new - eps_s) - cross_const_offset_s
            a_s_new += b_s * price_wedge_self + cross_wedge_shift
            
            # Update the constraint RHS for Qs - b_s*Pc - ... = a_s.
            # A Gurobi linear constraint's RHS is the intercept.
            con.RHS = a_s_new
    
    
    # 2. Update demand constraints using population and income multipliers.
    
    # Linear demand: Qd = D0 * pop_effect * inc_effect * (1 - epsilon_d) + epsilon_d*D0_adj/P0*Pc.
    # + Σ[ε_dj*D0_adj/P0j*(Pj-P0j)]
    # MC update: pop_effect_new = (pop_t * pop_mult / pop_base) ^ epsilon_pop.
    
    if pop_mult_by_region is not None or income_mult_by_region is not None:
        for key, con in cache.constr_demand.items():
            r, j, t = key
            
            D0 = cache.D0.get(key, 1.0)
            P0 = cache.P0.get(key, 1.0)
            eps_d = cache.eps_d.get(key, 0.0)
            eps_pop = cache.eps_pop.get(key, 0.0)
            eps_inc = cache.eps_inc.get(key, 0.0)
            pop_base = cache.pop_base.get(key, 1.0)
            inc_base = cache.inc_base.get(key, 1.0)
            
            # Get multipliers.
            p_mult = 1.0
            i_mult = 1.0
            if pop_mult_by_region:
                p_mult = pop_mult_by_region.get(r, 1.0)
            if income_mult_by_region:
                i_mult = income_mult_by_region.get(r, 1.0)
            
            # Get original population and income ratios.
            idx_data = cache.idx.get(key, {})
            pop_t = float(idx_data.get('pop_t', pop_base) or pop_base)
            inc_t = float(idx_data.get('inc_t', inc_base) or inc_base)
            
            # New population and income effects
            pop_ratio_new = (pop_t * p_mult) / max(1e-9, pop_base)
            inc_ratio_new = (inc_t * i_mult) / max(1e-9, inc_base)
            pop_effect_new = pop_ratio_new ** eps_pop if eps_pop != 0 else 1.0
            inc_effect_new = inc_ratio_new ** eps_inc if eps_inc != 0 else 1.0
            
            # Get cross-price elasticities.
            epsD_row_raw = {k: v for k, v in (idx_data.get('epsD_row', {}) or {}).items() if k in cache.commodities}
            epsD_row_scaled = _scale_cross_terms(epsD_row_raw, cross_terms_scale)
            epsD_row_limited = _limit_cross_terms(epsD_row_scaled, cross_terms_top_n)
            epsD_row, sum_cross_eps, _ = _normalize_cross_eps(epsD_row_limited, eps_d)
            
            # New adjusted D0
            D0_adjusted_new = D0 * pop_effect_new * inc_effect_new
            
            # New intercept
            P0 = max(1e-6, float(P0 or 0.0))
            b_d_abs = D0_adjusted_new * eps_d / P0
            b_d = b_d_abs
            if use_relative_price:
                price_ref_self = float(price_ref_by_comm.get(j, 1.0) or 1.0)
                if not np.isfinite(price_ref_self) or price_ref_self <= 0:
                    price_ref_self = 1.0
                b_d = b_d_abs * price_ref_self
            COEFF_THRESHOLD = 1e-6 * inv_qty_scale
            if abs(b_d) < COEFF_THRESHOLD:
                b_d = 0.0

            price_wedge_self = _price_wedge_local(r, j, t)
            cross_wedge_shift = 0.0
            cross_const_offset_d = 0.0
            if epsD_row:
                for other_comm, cross_eps in epsD_row.items():
                    if other_comm == j:
                        continue
                    other_base_key = (r, other_comm, hist_end_year)
                    P0_other = cache.idx.get(other_base_key, {}).get('P0')
                    if P0_other is None:
                        other_key = (r, other_comm, t)
                        P0_other = cache.idx.get(other_key, {}).get('P0', P0) or P0
                    P0_other = max(1e-6, float(P0_other))
                    b_cross_abs = D0_adjusted_new * float(cross_eps) / P0_other
                    b_cross = b_cross_abs
                    pc_other_base_for_offset = P0_other
                    if use_relative_price:
                        price_ref_other = float(price_ref_by_comm.get(other_comm, 1.0) or 1.0)
                        if not np.isfinite(price_ref_other) or price_ref_other <= 0:
                            price_ref_other = 1.0
                        b_cross = b_cross_abs * price_ref_other
                        pc_other_base_for_offset = P0_other / price_ref_other
                    b_cross = _sanitize_cross_coef(
                        b_cross,
                        base_qty=D0_adjusted_new,
                        cross_eps=float(cross_eps),
                        use_relative_price=use_relative_price,
                        ratio_tol=CROSS_COEF_RATIO_TOL,
                    )
                    if abs(b_cross) < COEFF_THRESHOLD:
                        continue
                    cross_const_offset_d += b_cross * pc_other_base_for_offset
                    other_wedge = _price_wedge_local(r, other_comm, t)
                    if other_wedge:
                        cross_wedge_shift += b_cross * other_wedge

            a_d_new = D0_adjusted_new * (1.0 - eps_d) - cross_const_offset_d
            a_d_new += b_d * price_wedge_self + cross_wedge_shift
            
            # Update the constraint RHS.
            con.RHS = a_d_new
    
    
    # 3. Update emissions definition constraints using emission-factor multipliers.
    
    # E_def: Eij = Σ(e0_p * Qs) - Σ(abatement)
    # MC update: e0_p_new = e0_p * mult.
    
    if e0_mult_by_region_comm_proc is not None:
        for key, con in cache.constr_Edef.items():
            r, j, t = key
            
            e0_map = cache.e0_by_region.get(key, {})
            if not e0_map:
                continue
            
            # Calculate the new total emissions intensity.
            new_sum_e0 = 0.0
            for proc, e0p in e0_map.items():
                mult = e0_mult_by_region_comm_proc.get((r, j, t, proc), 1.0)
                new_sum_e0 += float(e0p) * mult
            
            # Update the Qs coefficient.
            qs_var = cache.Qs.get(key)
            if qs_var is not None:
                m.chgCoeff(con, qs_var, -new_sum_e0 * qty_scale)
        
        # Also update coefficients in MACC abatement upper-bound constraints.
        for cap_key, cap_con in cache.abatement_caps.items():
            r, j, t, proc, seg = cap_key
            
            base_coeff = cache.proc_cap_basecoeff.get(cap_key)
            if base_coeff is None:
                continue
            mult = e0_mult_by_region_comm_proc.get((r, j, t, proc), 1.0)
            new_coeff = base_coeff * mult
            
            qs_var = cache.Qs.get((r, j, t))
            if qs_var is not None:
                # MACC constraint: a - coeff * Qs <= 0; the coefficient is negative.
                m.chgCoeff(cap_con, qs_var, -new_coeff)
        
        # Update emissions coefficients in baseline abatement constraints as e0_mult changes.
        for req_key, req_con in cache.abatement_req_constr.items():
            r, j, t, proc = req_key
            e0_map = cache.e0_by_region.get((r, j, t), {})
            e0p = e0_map.get(proc)
            if e0p is None:
                continue
            mult = e0_mult_by_region_comm_proc.get((r, j, t, proc), 1.0)
            new_coeff = float(e0p) * mult * qty_scale
            qs_var = cache.Qs.get((r, j, t))
            if qs_var is not None:
                m.chgCoeff(req_con, qs_var, new_coeff)
    
    
    # 4. Update nutrition constraint RHS values.
    
    if nutrition_rhs:
        for (r, t), con in cache.nutri_constr.items():
            rhs = nutrition_rhs.get((r, t))
            if rhs is not None:
                con.RHS = float(rhs)
    
    
    # 5. Update land constraint RHS values.
    
    if land_limits:
        for (r, t), con in cache.land_constr.items():
            lim = land_limits.get((r, t))
            if lim is not None:
                con.RHS = float(lim)
    
    
    # 5.5 Update ruminant demand cap RHS values (Phase 2).
    
    # MC sampling can update the RHS of rumi_intake_constr.
    # This assumes cache has a rumi_intake_constr field.
    # Pass ruminant_cap_by_region_year if updates are required.
    
    
    # 6. Update the objective for the land carbon price.
    
    if land_cp_by_year is not None:
        # Rebuild the objective.
        obj = gp.LinExpr(0.0)
        
        # Preserve the original slack penalty.
        SLACK_PENALTY = float(getattr(cache, 'slack_penalty', 1e6) or 1e6) * qty_scale
        for j in cache.commodities:
            for t in cache.years:
                if (j, t) in cache.excess:
                    obj += SLACK_PENALTY * cache.excess[j, t]
                if (j, t) in cache.shortage:
                    obj += SLACK_PENALTY * cache.shortage[j, t]
        nutrition_trade_pen = float(getattr(cache, 'nutrition_trade_penalty', 0.0) or 0.0)
        nutrition_imports = getattr(cache, 'nutrition_import_pos', None) or {}
        nutrition_exports = getattr(cache, 'nutrition_export_pos', None) or {}
        if nutrition_trade_pen > 0 and nutrition_imports:
            for key, var in nutrition_imports.items():
                obj += nutrition_trade_pen * var
                ex_var = nutrition_exports.get(key)
                if ex_var is not None:
                    obj += nutrition_trade_pen * ex_var
        headroom_pen = float(getattr(cache, 'nutrition_export_headroom_penalty', 0.0) or 0.0)
        headroom_weights = getattr(cache, 'nutrition_export_headroom_weight_by_key', {}) or {}
        if headroom_pen > 0 and headroom_weights and nutrition_exports:
            for key, weight in headroom_weights.items():
                ex_var = nutrition_exports.get(key)
                if ex_var is not None:
                    obj += headroom_pen * float(weight or 0.0) * ex_var
        armington_pen = getattr(cache, 'armington_trade_slack_penalty', None)
        if armington_pen is None:
            armington_pen = getattr(cache, 'slack_penalty', 1e6)
        ARMINGTON_TRADE_PENALTY = float(armington_pen or 1e6) * qty_scale
        if getattr(cache, 'armington_slack_pos', None):
            for key, var in cache.armington_slack_pos.items():
                obj += ARMINGTON_TRADE_PENALTY * var
            for key, var in cache.armington_slack_neg.items():
                obj += ARMINGTON_TRADE_PENALTY * var
        
        # Optional production cost term
        curtail_pen = getattr(cache, 'supply_curtailment_penalty', None)
        if curtail_pen is None:
            curtail_pen = getattr(cache, 'slack_penalty', 1e6)
        SUPPLY_CURTAIL_PENALTY = float(curtail_pen or 0.0) * qty_scale
        if getattr(cache, 'supply_curtailment', None) and SUPPLY_CURTAIL_PENALTY > 0:
            for var in cache.supply_curtailment.values():
                obj += SUPPLY_CURTAIL_PENALTY * var

        production_cost_obj, _effective_prod_weight = _build_production_cost_objective(
            qs=cache.Qs,
            p0_by_key=cache.P0,
            hist_end_year=int(getattr(cache, 'hist_end_year', 2020)),
            qty_scale=qty_scale,
            disable_production_cost_term=bool(
                getattr(cache, 'disable_production_cost_term', False)
            ),
            production_cost_weight=getattr(cache, 'production_cost_weight', 0.0),
            tax_unit_adder=getattr(cache, 'tax_unit_adder', {}) or {},
        )
        obj += production_cost_obj

        # Preserve abatement costs.
        for key, cij_var in cache.Cij.items():
            obj += cij_var
        
        # New land carbon price
        for key, qs_var in cache.Qs.items():
            r, j, t = key
            cp = float(land_cp_by_year.get(t, 0.0) or 0.0)
            if cp > 0:
                e0_map = cache.e0_by_region.get(key, {})
                e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
                if e_land > 0:
                    obj += cp * e_land * qty_scale * qs_var
        
        m.setObjective(obj, gp.GRB.MINIMIZE)
    
    # Update the model.
    m.update()


def _mc_draw_to_multiplier(draw: float, unit: str) -> float:
    """Legacy MC updates require multipliers; rate draws are relative changes."""
    unit_l = str(unit or '').strip().lower()
    if unit_l in {'rate', 'ratio', 'pct', 'percent', 'percentage'}:
        return max(0.0, 1.0 + float(draw))
    return max(0.0, float(draw))


def run_linear_mc(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    specs_df: pd.DataFrame,
    universe: Any,
    *,
    n_samples: int = 100,
    seed: int = 42,
    dict_v3_path: Optional[str] = None,
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    save_prefix: str = 'mc_linear_results',
    time_limit_per_sample: float = 60.0,
    grassland_method: str = 'dynamic',
    slack_penalty: Optional[float] = 1e6,
    supply_curtailment_enabled: bool = False,
    supply_curtailment_penalty: Optional[float] = None,
    zero_price_shutdown_enabled: bool = False,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation with the linear regional model.
    
    Similar to run_mc() in S3_0_ds_emis_mc_full.py, but using the linear regional model.
    
    Parameters:
        nodes: List of country-level nodes.
        commodities: List of commodities.
        years: List of years.
        specs_df: MC specification table from the MC sheet in Scenario_config.xlsx.
        universe: Universe object containing commodity classifications and related information.
        n_samples: Number of samples.
        seed: Random seed.
        dict_v3_path: Path to dict_v3.xlsx.
        macc_path: Path to MACC data.
        land_carbon_price_by_year: Land carbon price.
        population_by_country_year: Population data.
        income_mult_by_country_year: Income data.
        save_prefix: Output file prefix.
        time_limit_per_sample: Solve time limit per sample.
    
    Returns:
        DataFrame containing results for all samples.
    """
    from mc_bound_utils import draw_mc_multiplier
    from pathlib import Path
    if save_prefix:
        Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    
    logger.info(f"[LINEAR_MC] 开始蒙特卡洛模拟: n_samples={n_samples}, seed={seed}")
    
    # Build the model cache.
    cache = build_linear_model_cache(
        nodes=nodes,
        commodities=commodities,
        years=years,
        dict_v3_path=dict_v3_path,
        macc_path=macc_path,
        land_carbon_price_by_year=land_carbon_price_by_year,
        population_by_country_year=population_by_country_year,
        income_mult_by_country_year=income_mult_by_country_year,
        grassland_method=grassland_method,
        slack_penalty=slack_penalty,
        supply_curtailment_enabled=supply_curtailment_enabled,
        supply_curtailment_penalty=supply_curtailment_penalty,
        zero_price_shutdown_enabled=zero_price_shutdown_enabled,
    )
    cache_meta = getattr(cache.model, '_nzf_cache', {}) or {}
    try:
        qty_scale = float(cache_meta.get('qty_scale', 1.0) or 1.0)
    except Exception:
        qty_scale = 1.0
    if not np.isfinite(qty_scale) or qty_scale <= 0:
        qty_scale = 1.0
    
    m = cache.model
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit_per_sample)
    
    # Parse MC specifications.
    df = specs_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # Required columns
    c_elem = 'Element'
    c_proc = 'Process' if 'Process' in df.columns else None
    c_item = 'Item' if 'Item' in df.columns else None
    c_min = 'Min_bound'
    c_max = 'Max_bound'
    c_region = 'Region_cat' if 'Region_cat' in df.columns else None
    c_unit = 'Element unit' if 'Element unit' in df.columns else None
    
    if not all(c in df.columns for c in [c_elem, c_min, c_max]):
        logger.warning("[LINEAR_MC] 规格表缺少必要列")
        return pd.DataFrame()
    
    rng = np.random.default_rng(seed)
    recs = []
    
    for s in range(1, n_samples + 1):
        logger.info(f"[LINEAR_MC] 样本 {s}/{n_samples}")
        
        # Build multiplier dictionaries for this sample.
        yield_mult: Dict[Tuple[str, str], float] = {}
        e0_mult: Dict[Tuple[str, str, int, str], float] = {}
        
        for r in df.itertuples(index=False, name=None):
            row = dict(zip(df.columns, r))
            elem = str(row.get(c_elem, '')).lower()
            proc = str(row.get(c_proc, 'All')) if c_proc else 'All'
            item = str(row.get(c_item, 'All')) if c_item else 'All'
            unit = str(row.get(c_unit, '')) if c_unit else ''
            regc = str(row.get(c_region, 'All')) if c_region else 'All'
            mult = draw_mc_multiplier(rng, row.get(c_min, 1.0), row.get(c_max, 1.0), unit)
            
            # Determine target regions.
            if regc and regc.lower() != 'all':
                target_regions = [regc]
            else:
                target_regions = cache.regions
            
            # Determine target commodities.
            if item and item.lower() != 'all':
                # Check for a commodity category: crop/meat/dairy/other.
                if item.lower() in ('crop', 'meat', 'dairy', 'other'):
                    cat2 = getattr(universe, 'item_cat2_by_commodity', {}) or {}
                    target_comms = [c for c in cache.commodities 
                                   if cat2.get(c, '').lower() == item.lower()]
                else:
                    target_comms = [item] if item in cache.commodities else []
            else:
                target_comms = cache.commodities
            
            # Determine target processes.
            if proc and proc.lower() != 'all':
                target_procs = [proc]
            else:
                target_procs = list(getattr(universe, 'processes', []) or [])
            
            # Apply sampled values.
            if 'yield' in elem or 'productivity' in elem:
                for reg in target_regions:
                    for comm in target_comms:
                        yield_mult[(reg, comm)] = mult

            elif 'ef' in elem or 'emission' in elem:
                for reg in target_regions:
                    for comm in target_comms:
                        for p in target_procs:
                            for t in cache.years:
                                e0_mult[(reg, comm, t, p)] = mult
        
        # Apply updates.
        apply_linear_sample_updates(
            cache,
            yield_mult_by_region_comm=yield_mult if yield_mult else None,
            e0_mult_by_region_comm_proc=e0_mult if e0_mult else None,
        )
        
        # Solve.
        m.optimize()
        if m.Status == gp.GRB.OPTIMAL:
            _select_land_sources_on_optimal_face(m, logger)
        
        status = m.Status
        row = {'sample': s, 'status': status}
        
        if status == gp.GRB.OPTIMAL:
            # Summarize results.
            tot_E = sum(v.X for v in cache.Eij.values())
            tot_C = sum(v.X for v in cache.Cij.values())
            tot_Qs = sum(v.X for v in cache.Qs.values()) * qty_scale
            tot_Qd = sum(v.X for v in cache.Qd.values()) * qty_scale
            
            row['E_total'] = tot_E
            row['C_total'] = tot_C
            row['Qs_total'] = tot_Qs
            row['Qd_total'] = tot_Qd
            
            # Prices
            for (j, t), v in cache.Pc.items():
                row[f'{j}__Pc_{t}'] = v.X
            
            # Optionally save detailed regional results.
            # for (r, j, t), v in cache.Qs.items():
            # row[f'{r}::{j}::{t}__Qs'] = v.X
        
        recs.append(row)
        
        # Reset constraints to baseline values.
        # Simplification: ideally rebuild the model or save and restore original coefficients.
    
    out = pd.DataFrame(recs)
    
    # Save results.
    if save_prefix:
        out.to_csv(f'{save_prefix}__samples.csv', index=False, encoding='utf-8-sig')
        
        # Calculate statistics.
        num = out.drop(columns=['sample', 'status'], errors='ignore').apply(pd.to_numeric, errors='coerce')
        if num.shape[1] > 0 and num.shape[0] > 0:
            qs = num.quantile([0.5, 0.05, 0.95])
            qs.index = ['median', 'p05', 'p95']
            qs.to_csv(f'{save_prefix}__summary.csv', encoding='utf-8-sig')
    
    logger.info(f"[LINEAR_MC] 蒙特卡洛模拟完成: 成功样本={sum(1 for r in recs if r['status'] == gp.GRB.OPTIMAL)}/{n_samples}")
    
    return out



# Disaggregate regional results to countries.


def disaggregate_to_countries(
    nodes: List[Any],
    regional_result: Dict[str, Any],
    dict_v3_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Disaggregate regional results to country level.
    
    Disaggregation rules:
    - Pc[j,t]: the global price is assigned directly to every country.
    - Regional Qs[r,j,t] to country Qs[i,j,t]: allocate by base-period Q0 shares.
    - Regional Qd[r,j,t] to country Qd[i,j,t]: allocate by base-period D0 shares.
    - Regional E[r,j,t] to country E[i,j,t]: allocate by base-period Q0 shares.
    - Regional C[r,j,t] to country C[i,j,t]: allocate by base-period Q0 shares.
    
    Parameters:
        nodes: Original country-level node list.
        regional_result: Results returned by solve_linear_regional().
        dict_v3_path: Path to dict_v3.xlsx, used for regional mapping.
        
    Returns:
        country_result: {
            'Pc': {(country, commodity, year): price},
            'Qs': {(country, commodity, year): supply},
            'Qd': {(country, commodity, year): demand},
            'Eij': {(country, commodity, year): emissions},
            'Cij': {(country, commodity, year): abatement_cost},
        }
    """
    if regional_result.get('status') not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT):
        return {'status': regional_result.get('status'), 'Pc': {}, 'Qs': {}, 'Qd': {}, 'Eij': {}, 'Cij': {}}
    
    Pc_regional = regional_result.get('Pc', {})
    Qs_regional = regional_result.get('Qs', {})
    Qd_regional = regional_result.get('Qd', {})
    Eij_regional = regional_result.get('Eij', {})
    Cij_regional = regional_result.get('Cij', {})
    feed_regional = regional_result.get('feed_demand', {}) or {}
    feed_credit_regional = regional_result.get('feed_credit', {}) or {}
    feed_net_regional = regional_result.get('feed_demand_net_of_bioenergy_credit', {}) or {}
    
    # 1. Calculate base-period country shares within each region.
    supply_shares: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    demand_shares: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    
    for n in nodes:
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        region = get_region(n.country, m49=m49, dict_v3_path=dict_v3_path)
        key = (region, n.commodity, n.year)
        
        if key not in supply_shares:
            supply_shares[key] = {}
        q0 = getattr(n, 'Q0', 0.0) or 0.0
        supply_shares[key][n.country] = supply_shares[key].get(n.country, 0.0) + q0
        
        if key not in demand_shares:
            demand_shares[key] = {}
        d0 = getattr(n, 'D0', 0.0) or 0.0
        demand_shares[key][n.country] = demand_shares[key].get(n.country, 0.0) + d0
    
    # 2. Disaggregate to countries.
    Pc_country: Dict[Tuple[str, str, int], float] = {}
    Qs_country: Dict[Tuple[str, str, int], float] = {}
    Qd_country: Dict[Tuple[str, str, int], float] = {}
    Eij_country: Dict[Tuple[str, str, int], float] = {}
    Cij_country: Dict[Tuple[str, str, int], float] = {}
    feed_country: Dict[Tuple[str, str, int], float] = {}
    feed_credit_country: Dict[Tuple[str, str, int], float] = {}
    feed_net_country: Dict[Tuple[str, str, int], float] = {}
    
    for n in nodes:
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        region = get_region(n.country, m49=m49, dict_v3_path=dict_v3_path)
        key = (region, n.commodity, n.year)
        country_key = (n.country, n.commodity, n.year)
        
        # Regional or global prices
        price_key_reg = (region, n.commodity, n.year)
        if price_key_reg in Pc_regional:
            Pc_country[country_key] = Pc_regional[price_key_reg]
        else:
            price_key = (n.commodity, n.year)
            if price_key in Pc_regional:
                Pc_country[country_key] = Pc_regional[price_key]
        
        # Disaggregate supply.
        regional_key = (region, n.commodity, n.year)
        if regional_key in Qs_regional:
            regional_Qs = Qs_regional[regional_key]
            total_Q0 = sum(supply_shares.get(key, {}).values())
            if total_Q0 > 1e-9:
                country_Q0 = supply_shares.get(key, {}).get(n.country, 0.0)
                share = country_Q0 / total_Q0
                Qs_country[country_key] = regional_Qs * share
                
                # Disaggregate emissions and costs using supply shares as well.
                if regional_key in Eij_regional:
                    Eij_country[country_key] = Eij_regional[regional_key] * share
                if regional_key in Cij_regional:
                    Cij_country[country_key] = Cij_regional[regional_key] * share
            else:
                n_countries = len(supply_shares.get(key, {}))
                if n_countries > 0:
                    Qs_country[country_key] = regional_Qs / n_countries
                    if regional_key in Eij_regional:
                        Eij_country[country_key] = Eij_regional[regional_key] / n_countries
                    if regional_key in Cij_regional:
                        Cij_country[country_key] = Cij_regional[regional_key] / n_countries
        
        # Disaggregate demand.
        if regional_key in Qd_regional:
            regional_Qd = Qd_regional[regional_key]
            total_D0 = sum(demand_shares.get(key, {}).values())
            if total_D0 > 1e-9:
                country_D0 = demand_shares.get(key, {}).get(n.country, 0.0)
                share = country_D0 / total_D0
                Qd_country[country_key] = regional_Qd * share
                if regional_key in feed_regional:
                    feed_country[country_key] = float(feed_regional[regional_key]) * share
                if regional_key in feed_credit_regional:
                    feed_credit_country[country_key] = float(feed_credit_regional[regional_key]) * share
                if regional_key in feed_net_regional:
                    feed_net_country[country_key] = float(feed_net_regional[regional_key]) * share
            else:
                n_countries = len(demand_shares.get(key, {}))
                if n_countries > 0:
                    Qd_country[country_key] = regional_Qd / n_countries
                    if regional_key in feed_regional:
                        feed_country[country_key] = float(feed_regional[regional_key]) / n_countries
                    if regional_key in feed_credit_regional:
                        feed_credit_country[country_key] = float(feed_credit_regional[regional_key]) / n_countries
                    if regional_key in feed_net_regional:
                        feed_net_country[country_key] = float(feed_net_regional[regional_key]) / n_countries
    
    return {
        'status': regional_result.get('status'),
        'Pc': Pc_country,
        'Qs': Qs_country,
        'Qd': Qd_country,
        'Eij': Eij_country,
        'Cij': Cij_country,
        'feed_demand': feed_country,
        'feed_credit': feed_credit_country,
        'feed_demand_net_of_bioenergy_credit': feed_net_country,
    }


def apply_results_to_nodes(
    nodes: List[Any],
    country_result: Dict[str, Any],
) -> None:
    """
    Apply country-level results to nodes in place.
    
    Set the following node attributes:
    - P: Market-clearing price.
    - Q: Market-clearing supply.
    - D: Market-clearing demand.
    - E: Emissions.
    - abatement_cost: Abatement costs.
    """
    Pc = country_result.get('Pc', {})
    Qs = country_result.get('Qs', {})
    Qd = country_result.get('Qd', {})
    Eij = country_result.get('Eij', {})
    Cij = country_result.get('Cij', {})
    
    for n in nodes:
        key = (n.country, n.commodity, n.year)
        
        if key in Pc:
            n.P = Pc[key]
        if key in Qs:
            n.Q = Qs[key]
        if key in Qd:
            n.D = Qd[key]
        if key in Eij:
            n.E = Eij[key]
        if key in Cij:
            n.abatement_cost = Cij[key]



# Convenience function for the full workflow


def run_linear_regional_model(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    dict_v3_path: Optional[str] = None,
    time_limit: float = 300.0,
    # Emissions and abatement parameters
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    # Constraint parameters
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    apply_to_nodes: bool = True,
    *,
    use_relative_price: bool = False,
    relative_price_bounds: Tuple[float, float] = (0.1, 10.0),
    price_bounds_mode: str = 'absolute',
    price_bounds_p0_mult: Tuple[float, float] = (0.1, 10.0),
    price_bounds: Tuple[float, float] = (1e-6, 1e6),
    price_wedge_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    price_wedge_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    price_wedge_by_region: Optional[Dict[str, float]] = None,
    market_clearing_mode: str = 'country_trade',
    armington_sigma_by_comm: Optional[Dict[Any, float]] = None,
    trade_base_net_import: Optional[Dict[Tuple[str, str], float]] = None,
    trade_base_volume: Optional[Dict[Tuple[str, str], float]] = None,
    trade_cap_ratio: Any = None,
    armington_trade_scale: Optional[float] = None,
    qty_scale: float = 1.0,
    exclude_commodities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the complete linear regional model workflow.
    
    Steps:
    1. Build and solve the regional model.
    2. Disaggregate results to countries.
    3. Optionally apply results to nodes.
    
    Returns:
    - regional_result: Regional results.
    - country_result: Country results.
    - status: Solver status.
    """
    logger = logging.getLogger(__name__)
    
    # 1. Solve the regional model.
    logger.info("[LINEAR] 步骤 1: 求解区域模型...")
    regional_result = solve_linear_regional(
        nodes=nodes,
        commodities=commodities,
        years=years,
        time_limit=time_limit,
        dict_v3_path=dict_v3_path,
        use_relative_price=use_relative_price,
        relative_price_bounds=relative_price_bounds,
        price_bounds_mode=price_bounds_mode,
        price_bounds_p0_mult=price_bounds_p0_mult,
        price_bounds=price_bounds,
        price_wedge_by_region_comm_year=price_wedge_by_region_comm_year,
        price_wedge_by_region_comm=price_wedge_by_region_comm,
        price_wedge_by_region=price_wedge_by_region,
        market_clearing_mode=market_clearing_mode,
        armington_sigma_by_comm=armington_sigma_by_comm,
        trade_base_net_import=trade_base_net_import,
        trade_base_volume=trade_base_volume,
        trade_cap_ratio=trade_cap_ratio,
        armington_trade_scale=armington_trade_scale,
        qty_scale=qty_scale,
        exclude_commodities=exclude_commodities,
        macc_path=macc_path,
        land_carbon_price_by_year=land_carbon_price_by_year,
        nutrition_rhs=nutrition_rhs,
        nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
        land_area_limits=land_area_limits,
        yield_t_per_ha_default=yield_t_per_ha_default,
        max_growth_rate_per_period=max_growth_rate_per_period,
        max_decline_rate_per_period=max_decline_rate_per_period,
        hist_end_year=hist_end_year,
    )
    
    status = regional_result.get('status')
    if status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT):
        logger.warning(f"[LINEAR] 求解失败，状态={status}")
        return {'status': status, 'regional_result': regional_result, 'country_result': None}
    
    # 2. Disaggregate to countries.
    logger.info("[LINEAR] 步骤 2: 分解到国家级...")
    country_result = disaggregate_to_countries(
        nodes=nodes,
        regional_result=regional_result,
        dict_v3_path=dict_v3_path,
    )
    
    # 3. Apply results to nodes.
    if apply_to_nodes:
        logger.info("[LINEAR] 步骤 3: 应用结果到节点...")
        apply_results_to_nodes(nodes, country_result)
    
    logger.info("[LINEAR] 完成！")
    
    return {
        'status': status,
        'regional_result': regional_result,
        'country_result': country_result,
    }



# Approach B: grassland iteration framework (grassland_method='static')


def solve_with_grassland_iteration(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    time_limit: float = 300.0,
    dict_v3_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    gurobi_log_path: Optional[str] = None,
    solver_method: Optional[int] = None,
    solver_threads: Optional[int] = None,
    use_relative_price: bool = False,
    relative_price_bounds: Tuple[float, float] = (0.1, 10.0),
    price_bounds_mode: str = 'absolute',
    price_bounds_p0_mult: Tuple[float, float] = (0.1, 10.0),
    price_bounds: Tuple[float, float] = (1e-6, 1e6),
    price_wedge_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    price_wedge_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    price_wedge_by_region: Optional[Dict[str, float]] = None,
    market_clearing_mode: str = 'country_trade',
    armington_sigma_by_comm: Optional[Dict[Any, float]] = None,
    trade_base_net_import: Optional[Dict[Tuple[str, str], float]] = None,
    trade_base_volume: Optional[Dict[Tuple[str, str], float]] = None,
    trade_cap_region_volume: Optional[Dict[Tuple[str, str], float]] = None,
    trade_cap_region_map: Optional[Dict[Any, str]] = None,
    trade_cap_ratio: Any = None,
    trade_cap_exempt_pairs: Optional[set] = None,
    armington_trade_scale: Optional[float] = None,
    armington_trade_slack_penalty: Optional[float] = None,
    qty_scale: float = 1.0,
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    tax_unit_adder: Optional[Dict[Tuple[str, str, int], float]] = None,
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    land_soft_constraints_enabled: bool = False,
    land_slack_max_rate: Optional[float] = None,
    land_slack_penalty: Optional[float] = None,
    land_delta_anchor_to_available_stock: bool = False,
    grass_area_by_region_year_initial: Optional[Dict[Tuple[str, int], float]] = None,
    forest_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    forest_global_target_slack_enabled: bool = False,
    forest_global_target_slack_penalty: Optional[float] = None,
    forest_global_target_slack_max_rate: Optional[float] = None,
    forest_nonneg_ratio: float = 1.0,
    cropland_nonforest_expand_ratio: float = 1.0,
    pasture_nonforest_expand_ratio: float = 1.0,
    base_cropland_by_region: Optional[Dict[str, float]] = None,
    base_grassland_by_region: Optional[Dict[str, float]] = None,
    base_forest_by_region: Optional[Dict[str, float]] = None,
    background_forest_to_cropland_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_forest_to_grassland_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_cropland_to_forest_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    background_grassland_to_forest_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    land_demand_calibration_mode: str = 'none',
    yield_by_region_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    grassland_conversion_penalty: float = 0.0,
    grassland_to_cropland_cost_mode: str = 'per_ha_cost',
    cropland_to_grassland_penalty: float = 0.0,
    land_conversion_allocation_mode: str = 'priority_nonforest_pasture_forest',
    land_conversion_priority_penalty_per_ha: float = 1e6,
    land_priority_weight_grassland_to_cropland: float = 1.0,
    land_priority_weight_forest_to_cropland: float = 100.0,
    land_priority_weight_forest_to_grassland: float = 100.0,
    luc_direct_carbon_price: bool = False,
    limit_reforestation_to_released_ag_land: bool = True,
    prevent_land_conversion_cycles: bool = True,
    reforestation_physical_cap_enabled: bool = True,
    reforestation_max_forest_increase_ratio: Optional[float] = 0.30,
    ruminant_intake_cap: Optional[Dict[Tuple[str, int], float]] = None,
    ruminant_commodities: Optional[List[str]] = None,
    luc_opt_mode: str = 'explicit',
    luc_params: Optional[Dict[str, Any]] = None,
    luc_shift_area_mode: str = 'abs',
    luc_penalty_by_region_year: Optional[Dict[Tuple[str, int], Dict[str, float]]] = None,
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    hist_max_production: Optional[Dict[Tuple[str, str], float]] = None,
    max_iterations: int = 10,
    convergence_tolerance: float = 0.01,
    damping_factor: float = 0.5,
    # feed-crop link
    feed_crop_link_mode: Optional[str] = None,
    feed_crop_link_base: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_coeff: Optional[Dict[Tuple[str, str, str, int], float]] = None,
    feed_background_crop: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_background_pasture: Optional[Dict[Tuple[str, int], float]] = None,
    feed_crop_link_livestock: Optional[List[str]] = None,
    feed_crop_link_override: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_crop_link_credit: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_requirement_scheme: Optional[str] = None,
    feed_conversion_multiplier: Optional[
        Dict[Tuple[str, str, int], float]
    ] = None,
    # Demand projection method
    demand_method: str = 'elasticity',
    nutrition_profile_xlsx: Optional[str] = None,
    nutrition_profile_sheet: Any = 0,
    nutrition_indicator: str = 'energy',
    nutrition_use_baseyear_for_future: bool = True,
    nutrition_band_epsilon: float = 0.1,
    nutrition_feed_t_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    nutrition_residual_demand_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    bioenergy_crop_demand_by_country_comm_year: Optional[Dict[Tuple[str, str, int], float]] = None,
    energy_crop_land_requirement_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    waste_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    losses_ratio_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    # Market imbalance limits
    max_slack_rate: Optional[float] = 0.1,
    max_shortage_slack_rate: Any = "inherit",
    max_excess_slack_rate: Any = "inherit",
    slack_penalty: Optional[float] = 1e6,
    supply_curtailment_enabled: bool = False,
    supply_curtailment_penalty: Optional[float] = None,
    zero_price_shutdown_enabled: bool = False,
    zero_demand_production_shutdown: bool = False,
    disable_production_cost_term: bool = True,
    production_cost_weight: float = 1.0,
    # Cross-elasticity term clipping
    cross_terms_top_n: Optional[int] = None,
    cross_terms_scale: Optional[float] = None,
    exclude_commodities: Optional[List[str]] = None,
    # Unit-cost method parameters
    unit_cost_data: Optional[Dict[Tuple[str, str], float]] = None,
    baseline_scenario_result: Optional[Dict[str, Any]] = None,
    process_cost_mapping: Optional[Dict[str, str]] = None,
    cost_calculation_method: str = 'MACC',
    active_strategy_cost_keys: Optional[Sequence[str]] = None,
    strategy_cost_regions: Optional[Sequence[str]] = None,
    cost_database_metadata: Optional[Mapping[str, Any]] = None,
    cost_strategy_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    post_solve_violation_tol: Optional[float] = 1e-6,
    post_solve_violation_top_n: int = 20,
) -> Dict:
    """
    Approach B: iterative solving with grassland area as an exogenous parameter.
    
    Iteration workflow:
    1. Round 1: solve using initial grassland area to obtain optimized Qs.
    2. Calculate new grassland demand via livestock production -> stock -> feed demand -> grassland_ha.
    3. Round 2: solve again with updated grassland to obtain new Qs.
    4. Repeat steps 2-3 until grassland changes are below tolerance or max_iterations is reached.
    
    Convergence criteria:
    - max(Delta_grassland / grassland_prev) < convergence_tolerance (relative change < 1%).
    - max(Delta_objective / objective_prev) < convergence_tolerance (objective change < 1%).
    
    Damping strategy:
    - grassland_new = damping_factor * grassland_calculated + (1-damping) * grassland_old
    - Prevent oscillation and stabilize convergence.
    
    Args:
        nodes: List of country-level nodes.
        ... (other parameters match solve_linear_regional).
        grass_area_by_region_year_initial: Initial grassland area {(region, year): ha}.
        max_iterations: Maximum iterations (default 10).
        convergence_tolerance: Convergence tolerance (default 1%).
        damping_factor: Damping factor (default 0.5).
    
    Returns:
        Same result dictionary as solve_linear_regional, plus:
        - 'iterations': Actual iteration count.
        - 'converged': Whether convergence was achieved (True/False).
        - 'grassland_history': Grassland area at each iteration.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[GRASSLAND_ITER] 开始方案B迭代求解（最大迭代次数={max_iterations}，收敛容差={convergence_tolerance*100:.1f}%）")
    
    # Initialize grassland area.
    grass_area_current = dict(grass_area_by_region_year_initial or {})
    grass_area_history = [dict(grass_area_current)]
    
    objective_prev = None
    converged = False
    
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n[GRASSLAND_ITER] ===== 迭代 {iteration}/{max_iterations} =====")
        
        # Solve the linear model using current grassland area.
        result = solve_linear_regional(
            nodes=nodes,
            commodities=commodities,
            years=years,
            time_limit=time_limit,
            dict_v3_path=dict_v3_path,
            output_dir=output_dir,
            gurobi_log_path=gurobi_log_path,
            solver_method=solver_method,
            solver_threads=solver_threads,
            use_relative_price=use_relative_price,
            relative_price_bounds=relative_price_bounds,
            price_bounds_mode=price_bounds_mode,
            price_bounds_p0_mult=price_bounds_p0_mult,
            price_bounds=price_bounds,
            price_wedge_by_region_comm_year=price_wedge_by_region_comm_year,
            price_wedge_by_region_comm=price_wedge_by_region_comm,
            price_wedge_by_region=price_wedge_by_region,
            market_clearing_mode=market_clearing_mode,
            armington_sigma_by_comm=armington_sigma_by_comm,
            trade_base_net_import=trade_base_net_import,
            trade_base_volume=trade_base_volume,
            trade_cap_region_volume=trade_cap_region_volume,
            trade_cap_region_map=trade_cap_region_map,
            trade_cap_ratio=trade_cap_ratio,
            trade_cap_exempt_pairs=trade_cap_exempt_pairs,
            armington_trade_scale=armington_trade_scale,
            armington_trade_slack_penalty=armington_trade_slack_penalty,
            qty_scale=qty_scale,
            population_by_country_year=population_by_country_year,
            income_mult_by_country_year=income_mult_by_country_year,
            macc_path=macc_path,
            land_carbon_price_by_year=land_carbon_price_by_year,
            tax_unit_adder=tax_unit_adder,
            luc_opt_mode=luc_opt_mode,
            luc_params=luc_params,
            luc_shift_area_mode=luc_shift_area_mode,
            luc_penalty_by_region_year=luc_penalty_by_region_year,
            nutrition_rhs=nutrition_rhs,
            nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
            land_area_limits=land_area_limits,
            land_soft_constraints_enabled=land_soft_constraints_enabled,
            land_slack_max_rate=land_slack_max_rate,
            land_slack_penalty=land_slack_penalty,
            land_delta_anchor_to_available_stock=land_delta_anchor_to_available_stock,
            grass_area_by_region_year=grass_area_current,  # Use current grassland area.
            forest_area_by_region_year=forest_area_by_region_year,
            forest_global_target_slack_enabled=forest_global_target_slack_enabled,
            forest_global_target_slack_penalty=forest_global_target_slack_penalty,
            forest_global_target_slack_max_rate=forest_global_target_slack_max_rate,
            forest_nonneg_ratio=forest_nonneg_ratio,
            cropland_nonforest_expand_ratio=cropland_nonforest_expand_ratio,
            pasture_nonforest_expand_ratio=pasture_nonforest_expand_ratio,
            base_cropland_by_region=base_cropland_by_region,
            base_grassland_by_region=base_grassland_by_region,
            base_forest_by_region=base_forest_by_region,
            background_forest_to_cropland_by_region_year=background_forest_to_cropland_by_region_year,
            background_forest_to_grassland_by_region_year=background_forest_to_grassland_by_region_year,
            background_cropland_to_forest_by_region_year=background_cropland_to_forest_by_region_year,
            background_grassland_to_forest_by_region_year=background_grassland_to_forest_by_region_year,
            land_demand_calibration_mode=land_demand_calibration_mode,
            yield_by_region_comm_year=yield_by_region_comm_year,
            yield_t_per_ha_default=yield_t_per_ha_default,
            grassland_method='static',  # Force approach B.
            grassland_conversion_penalty=grassland_conversion_penalty,
            grassland_to_cropland_cost_mode=grassland_to_cropland_cost_mode,
            cropland_to_grassland_penalty=cropland_to_grassland_penalty,
            land_conversion_allocation_mode=land_conversion_allocation_mode,
            land_conversion_priority_penalty_per_ha=land_conversion_priority_penalty_per_ha,
            land_priority_weight_grassland_to_cropland=land_priority_weight_grassland_to_cropland,
            land_priority_weight_forest_to_cropland=land_priority_weight_forest_to_cropland,
            land_priority_weight_forest_to_grassland=land_priority_weight_forest_to_grassland,
            luc_direct_carbon_price=luc_direct_carbon_price,
            limit_reforestation_to_released_ag_land=limit_reforestation_to_released_ag_land,
            prevent_land_conversion_cycles=prevent_land_conversion_cycles,
            reforestation_physical_cap_enabled=reforestation_physical_cap_enabled,
            reforestation_max_forest_increase_ratio=reforestation_max_forest_increase_ratio,
            ruminant_intake_cap=ruminant_intake_cap,
            ruminant_commodities=ruminant_commodities,
            max_growth_rate_per_period=max_growth_rate_per_period,
            max_decline_rate_per_period=max_decline_rate_per_period,
            hist_end_year=hist_end_year,
            hist_max_production=hist_max_production,
            demand_method=demand_method,
            nutrition_profile_xlsx=nutrition_profile_xlsx,
            nutrition_profile_sheet=nutrition_profile_sheet,
            nutrition_indicator=nutrition_indicator,
            nutrition_use_baseyear_for_future=nutrition_use_baseyear_for_future,
            nutrition_band_epsilon=nutrition_band_epsilon,
            nutrition_feed_t_by_country_comm_year=nutrition_feed_t_by_country_comm_year,
            nutrition_residual_demand_by_country_comm_year=nutrition_residual_demand_by_country_comm_year,
            bioenergy_crop_demand_by_country_comm_year=bioenergy_crop_demand_by_country_comm_year,
            energy_crop_land_requirement_by_region_year=energy_crop_land_requirement_by_region_year,
            waste_reduction_by=waste_reduction_by,
            losses_ratio_by=losses_ratio_by,
            feed_crop_link_mode=feed_crop_link_mode,
            feed_crop_link_base=feed_crop_link_base,
            feed_crop_link_coeff=feed_crop_link_coeff,
        feed_background_crop=feed_background_crop,
        feed_background_pasture=feed_background_pasture,
            feed_crop_link_livestock=feed_crop_link_livestock,
            feed_crop_link_override=feed_crop_link_override,
            feed_crop_link_credit=feed_crop_link_credit,
            max_slack_rate=max_slack_rate,
            max_shortage_slack_rate=max_shortage_slack_rate,
            max_excess_slack_rate=max_excess_slack_rate,
            slack_penalty=slack_penalty,
            supply_curtailment_enabled=supply_curtailment_enabled,
            supply_curtailment_penalty=supply_curtailment_penalty,
            zero_price_shutdown_enabled=zero_price_shutdown_enabled,
            zero_demand_production_shutdown=zero_demand_production_shutdown,
            disable_production_cost_term=disable_production_cost_term,
            production_cost_weight=production_cost_weight,
            cross_terms_top_n=cross_terms_top_n,
            cross_terms_scale=cross_terms_scale,
            exclude_commodities=exclude_commodities,
            unit_cost_data=unit_cost_data,
            baseline_scenario_result=baseline_scenario_result,
            process_cost_mapping=process_cost_mapping,
            cost_calculation_method=cost_calculation_method,
            active_strategy_cost_keys=active_strategy_cost_keys,
            strategy_cost_regions=strategy_cost_regions,
            cost_database_metadata=cost_database_metadata,
            cost_strategy_metadata=cost_strategy_metadata,
            post_solve_violation_tol=post_solve_violation_tol,
            post_solve_violation_top_n=post_solve_violation_top_n,
        )
        
        if result['status'] != 2:  # Not OPTIMAL
            logger.warning(f"[GRASSLAND_ITER] 迭代{iteration}未达到最优状态（status={result['status']}），终止迭代")
            result['iterations'] = iteration
            result['converged'] = False
            result['grassland_history'] = grass_area_history
            return result
        
        objective_current = result.get('objective', 0.0)
        logger.info(f"[GRASSLAND_ITER] 迭代{iteration}完成，目标函数={objective_current:.2f}")
        
        # Check objective convergence.
        obj_change = None
        if objective_prev is not None:
            obj_change = abs(objective_current - objective_prev) / max(abs(objective_prev), 1e-6)
            logger.info(f"[GRASSLAND_ITER] 目标函数变化: {obj_change*100:.3f}%")
            if obj_change < convergence_tolerance:
                logger.info(f"[GRASSLAND_ITER]  目标函数收敛（变化 < {convergence_tolerance*100:.1f}%）")
                converged = True
        
        # Calculate new grassland demand from optimized results.
        # Requires optimized livestock Qs, livestock-to-stock conversion, and stock-to-grassland conversion.
        try:
            # Extract optimized livestock production.
            from S3_2_feed_demand import build_feed_demand_from_stock
            from gle_emissions_complete import calculate_stock_from_optimized_production
            from S2_0_load_data import DataPaths, load_emis_item_mappings
            from S1_0_schema import Universe
            
            # Construct a simplified universe.
            universe = Universe(
                countries=sorted(set(n.country for n in nodes)),
                iso3_by_country={n.country: getattr(n, 'iso3', '') for n in nodes},
                commodities=sorted(set(n.commodity for n in nodes)),
                years=sorted(set(n.year for n in nodes)),
            )
            
            paths = DataPaths()
            maps = load_emis_item_mappings(dict_v3_path or paths.dict_v3_path)
            
            # Calculate new stocks.
            country_result = result.get('country_result', {})
            qs_optimized = country_result.get('Qs', {})
            
            stock_df = calculate_stock_from_optimized_production(
                optimized_qs=qs_optimized,
                nodes=nodes,
                universe=universe,
                maps=maps,
                paths=paths
            )
            
            # Calculate feed demand, including grassland, from stocks.
            feed_outputs = build_feed_demand_from_stock(
                stock_df=stock_df,
                universe=universe,
                maps=maps,
                paths=paths,
                years=years,
                conversion_multiplier=feed_conversion_multiplier,
                feed_requirement_scheme=feed_requirement_scheme,
            )
            
            grass_req_df = feed_outputs.grass_requirement
            
            if grass_req_df.empty:
                logger.warning(f"[GRASSLAND_ITER] 迭代{iteration}：未计算到grassland需求，使用旧值")
                grass_area_new = dict(grass_area_current)
            else:
                # Aggregate to regions.
                # grass_req_df: [country, iso3, year, grass_tdm, grass_area_need_ha]
                grass_req_df['region'] = grass_req_df['country'].apply(
                    lambda c: get_region(c, dict_v3_path=dict_v3_path)
                )
                
                grass_area_new_df = grass_req_df.groupby(['region', 'year'])['grass_area_need_ha'].sum().reset_index()
                grass_area_new = {
                    (str(row['region']), int(row['year'])): float(row['grass_area_need_ha'])
                    for _, row in grass_area_new_df.iterrows()
                }
                
                # Apply damping to prevent oscillation.
                grass_area_damped = {}
                for key in set(grass_area_current.keys()) | set(grass_area_new.keys()):
                    old_val = grass_area_current.get(key, 0.0)
                    new_val = grass_area_new.get(key, 0.0)
                    damped_val = damping_factor * new_val + (1 - damping_factor) * old_val
                    grass_area_damped[key] = damped_val
                
                grass_area_new = grass_area_damped
            
            # Check grassland convergence.
            grass_changes = []
            for key in grass_area_current.keys():
                old_val = grass_area_current[key]
                new_val = grass_area_new.get(key, 0.0)
                if old_val > 1e-6:
                    rel_change = abs(new_val - old_val) / old_val
                    grass_changes.append(rel_change)
            
            if grass_changes:
                max_grass_change = max(grass_changes)
                avg_grass_change = sum(grass_changes) / len(grass_changes)
                logger.info(f"[GRASSLAND_ITER] Grassland变化: max={max_grass_change*100:.3f}%, avg={avg_grass_change*100:.3f}%")
                
                if max_grass_change < convergence_tolerance and converged:
                    logger.info(f"[GRASSLAND_ITER]  Grassland和目标函数均收敛，迭代完成")
                    result['iterations'] = iteration
                    result['converged'] = True
                    result['grassland_history'] = grass_area_history
                    return result
            else:
                max_grass_change = None
                avg_grass_change = None

            # Summarize iteration convergence.
            obj_pct = f"{obj_change*100:.3f}%" if obj_change is not None else "N/A"
            max_pct = f"{max_grass_change*100:.3f}%" if max_grass_change is not None else "N/A"
            avg_pct = f"{avg_grass_change*100:.3f}%" if avg_grass_change is not None else "N/A"
            logger.info(
                f"[GRASSLAND_ITER] 摘要: iter={iteration}, obj_change={obj_pct}, "
                f"grass_max={max_pct}, grass_avg={avg_pct}, converged={converged}"
            )
            
            # Update grassland area for the next iteration.
            grass_area_current = grass_area_new
            grass_area_history.append(dict(grass_area_current))
            objective_prev = objective_current
            
        except Exception as e:
            logger.error(f"[GRASSLAND_ITER] 迭代{iteration}：计算新grassland失败: {e}")
            import traceback
            traceback.print_exc()
            # Continue using previous grassland values.
            grass_area_history.append(dict(grass_area_current))
    
    # Maximum iterations reached
    logger.warning(f"[GRASSLAND_ITER]  达到最大迭代次数{max_iterations}，未完全收敛")
    result['iterations'] = max_iterations
    result['converged'] = False
    result['grassland_history'] = grass_area_history
    return result



# Test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Create test data.
    from dataclasses import dataclass
    
    @dataclass
    class MockNode:
        country: str
        commodity: str
        year: int
        Q0: float = 1000.0
        D0: float = 1000.0
        P0: float = 100.0
        eps_supply: float = 0.3
        eps_demand: float = -0.5
        eps_pop_demand: float = 0.0
        eps_income_demand: float = 0.0
        e0_by_proc: Dict = None
        
        def __post_init__(self):
            if self.e0_by_proc is None:
                self.e0_by_proc = {'Enteric fermentation': 0.5, 'Manure management': 0.2}
    
    # Generate test nodes.
    test_countries = ['United States of America', 'China', 'Brazil', 'Germany', 'India']
    test_commodities = ['Wheat', 'Rice', 'Maize (corn)']
    test_years = [2020, 2050]
    
    nodes = []
    for c in test_countries:
        for j in test_commodities:
            for t in test_years:
                nodes.append(MockNode(
                    country=c, commodity=j, year=t,
                    Q0=1000 + np.random.rand() * 500,
                    D0=1000 + np.random.rand() * 500,
                ))
    
    print(f"测试节点数: {len(nodes)}")
    
    # Solve.
    result = solve_linear_regional(nodes, test_commodities, test_years, time_limit=60)
    
    print(f"\n状态: {result['status']}")
    if 'Pc' in result:
        print("\n价格结果:")
        for (j, t), p in sorted(result['Pc'].items()):
            print(f"  {j}, {t}: {p:.2f}")
