# -*- coding: utf-8 -*-
"""Bioenergy demand and feedstock preprocessing for the food-land model.

The module deliberately keeps crop-based bioenergy separate from food demand.
Only feedstocks explicitly linked to an existing model commodity enter the
commodity-market balance. Residues, manure, forest biomass, waste, and
dedicated energy crops remain in the detailed resource table until dedicated
physical constraints are enabled for those feedstock classes. P1 resource
accounting adds explicit availability, land, and emissions handoff diagnostics
for those non-market feedstocks without changing the solver objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from model_run_status import normalize_solver_status
from S1_0_schema import Universe
try:
    from runtime_data_cache import read_csv_cached, read_excel_cached
except ImportError:
    # Keep the standalone bioenergy preprocessor usable when the optional
    # project-wide persistent cache module is unavailable.
    def read_csv_cached(path: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)

    def read_excel_cached(path: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_excel(path, **kwargs)


COUNTRY_COMM_YEAR = Tuple[str, str, int]

RESIDUE_QUALITY_COLUMNS = [
    "residue_carbon_pct",
    "residue_nitrogen_pct",
    "residue_phosphorus_pct",
    "residue_potassium_pct",
    "residue_calcium_pct",
    "residue_magnesium_pct",
    "residue_sulfur_pct",
    "residue_lignin_pct",
    "residue_polyphenols_pct",
    "residue_cellulose_pct",
    "residue_ash_pct",
    "residue_quality_n_obs",
]

RESIDUE_QUALITY_ALIASES = {
    "residue_carbon_pct": ["carbon_pct", "carbon_percent", "Carbon (%)"],
    "residue_nitrogen_pct": ["nitrogen_pct", "nitrogen_percent", "Nitrogen (%)"],
    "residue_phosphorus_pct": ["phosphorus_pct", "phosphorus_percent", "Phosphorus"],
    "residue_potassium_pct": ["potassium_pct", "potassium_percent", "Potassium (%)"],
    "residue_calcium_pct": ["calcium_pct", "calcium_percent", "Calcium (%)"],
    "residue_magnesium_pct": ["magnesium_pct", "magnesium_percent", "Magnesium (%)"],
    "residue_sulfur_pct": ["sulfur_pct", "sulfur_percent", "Sulfur"],
    "residue_lignin_pct": ["lignin_pct", "lignin_percent", "Lignin (%)"],
    "residue_polyphenols_pct": ["polyphenols_pct", "polyphenols_percent", "Polyphenols (%)"],
    "residue_cellulose_pct": ["cellulose_pct", "cellulose_percent", "Cellulose (%)"],
    "residue_ash_pct": ["ash_pct", "ash_percent", "Ash (%)"],
    "residue_quality_n_obs": ["quality_n_obs", "residue_quality_observations"],
}

DETAIL_COLUMNS = [
    "scenario",
    "M49_Country_Code",
    "country_name",
    "year",
    "carrier",
    "feedstock",
    "feedstock_category",
    "model_commodity",
    "market_link",
    "energy_basis",
    "energy_target_tj",
    "feedstock_demand_t",
    "feedstock_demand_tdm",
    "energy_supplied_tj",
    "energy_gap_tj",
    "parent_commodity",
    "resource_available_t",
    "resource_available_tdm",
    "competing_use_t",
    "competing_use_tdm",
    "sustainable_fraction",
    "sustainable_supply_tdm",
    "feasible_feedstock_demand_t",
    "feasible_feedstock_demand_tdm",
    "feasible_energy_supplied_tj",
    "resource_gap_t",
    "resource_gap_tdm",
    "unmet_energy_tj",
    "resource_use_ratio",
    "resource_status",
    "yield_tdm_per_ha",
    "eligible_land_area_ha",
    "energy_crop_area_target_ha",
    "land_requirement_ha",
    "cropland_reallocation_area_ha",
    "additional_land_expansion_ha",
    "feasible_additional_land_expansion_ha",
    "land_gap_ha",
    "ghg_direct_kgco2e_per_tdm",
    "ghg_soil_kgco2e_per_tdm",
    "ghg_avoided_kgco2e_per_tdm",
    "fossil_displacement_kgco2e_per_tj",
    "beccs_capture_kgco2_per_tdm",
    *RESIDUE_QUALITY_COLUMNS,
    "lhv_gj_per_tdm",
    "conversion_efficiency",
    "dry_matter_fraction",
    "source",
    "notes",
    "coproduct_feed_commodity",
    "coproduct_feed_credit_tdm",
    "residue_feed_competition_tdm",
    "allocation_status",
]

PARAMETER_COLUMNS = [
    "feedstock",
    "feedstock_category",
    "model_commodity",
    "market_link",
    "carrier",
    "parent_commodity",
    "sustainable_fraction",
    "yield_tdm_per_ha",
    "ghg_direct_kgco2e_per_tdm",
    "ghg_soil_kgco2e_per_tdm",
    "ghg_avoided_kgco2e_per_tdm",
    "fossil_displacement_kgco2e_per_tj",
    "beccs_capture_kgco2_per_tdm",
    *RESIDUE_QUALITY_COLUMNS,
    "lhv_gj_per_tdm",
    "conversion_efficiency",
    "dry_matter_fraction",
    "source",
    "notes",
]

P1_RESOURCE_COLUMNS = [
    "scenario",
    "M49_Country_Code",
    "country_name",
    "year",
    "feedstock",
    "feedstock_category",
    "parent_commodity",
    "resource_available_t",
    "resource_available_tdm",
    "competing_use_t",
    "competing_use_tdm",
    "sustainable_fraction",
    "yield_tdm_per_ha",
    "eligible_land_area_ha",
    "ghg_direct_kgco2e_per_tdm",
    "ghg_soil_kgco2e_per_tdm",
    "ghg_avoided_kgco2e_per_tdm",
    "fossil_displacement_kgco2e_per_tj",
    "beccs_capture_kgco2_per_tdm",
    *RESIDUE_QUALITY_COLUMNS,
    "source",
    "notes",
]

P1_NUMERIC_COLUMNS = [
    "resource_available_t",
    "resource_available_tdm",
    "competing_use_t",
    "competing_use_tdm",
    "sustainable_fraction",
    "sustainable_supply_tdm",
    "feasible_feedstock_demand_t",
    "feasible_feedstock_demand_tdm",
    "feasible_energy_supplied_tj",
    "resource_gap_t",
    "resource_gap_tdm",
    "unmet_energy_tj",
    "resource_use_ratio",
    "yield_tdm_per_ha",
    "eligible_land_area_ha",
    "land_requirement_ha",
    "cropland_reallocation_area_ha",
    "additional_land_expansion_ha",
    "feasible_additional_land_expansion_ha",
    "land_gap_ha",
    "ghg_direct_kgco2e_per_tdm",
    "ghg_soil_kgco2e_per_tdm",
    "ghg_avoided_kgco2e_per_tdm",
    "fossil_displacement_kgco2e_per_tj",
    "beccs_capture_kgco2_per_tdm",
    *RESIDUE_QUALITY_COLUMNS,
    "energy_crop_area_target_ha",
    "coproduct_feed_credit_tdm",
    "residue_feed_competition_tdm",
]

P1_RESOURCE_ABSOLUTE_COLUMNS = [
    "resource_available_t",
    "resource_available_tdm",
    "competing_use_t",
    "competing_use_tdm",
    "eligible_land_area_ha",
]

P1_RESOURCE_RATE_COLUMNS = [
    "parent_commodity",
    "feedstock_category",
    "sustainable_fraction",
    "yield_tdm_per_ha",
    "ghg_direct_kgco2e_per_tdm",
    "ghg_soil_kgco2e_per_tdm",
    "ghg_avoided_kgco2e_per_tdm",
    "fossil_displacement_kgco2e_per_tj",
    "beccs_capture_kgco2_per_tdm",
    *RESIDUE_QUALITY_COLUMNS,
    "source",
    "notes",
]

DEDICATED_ENERGY_CROP_CATEGORIES = {
    "dedicated_energy_crop",
    "energy_crop",
    "perennial_energy_crop",
    "short_rotation_coppice",
    "miscanthus",
    "switchgrass",
    "eucalypt",
    "poplar",
    "willow",
}


@dataclass
class BioenergyBundle:
    scenario: str
    historical_detail: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=DETAIL_COLUMNS))
    scenario_detail: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=DETAIL_COLUMNS))
    crop_demand_by_country_comm_year: Dict[COUNTRY_COMM_YEAR, float] = field(default_factory=dict)
    historical_crop_demand_by_country_comm_year: Dict[COUNTRY_COMM_YEAR, float] = field(default_factory=dict)
    historical_crop_base_by_country_comm: Dict[Tuple[str, str], float] = field(default_factory=dict)
    energy_balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    resource_balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_feasibility: pd.DataFrame = field(default_factory=pd.DataFrame)
    emissions_handoff: pd.DataFrame = field(default_factory=pd.DataFrame)
    land_handoff: pd.DataFrame = field(default_factory=pd.DataFrame)
    residue_management_handoff: pd.DataFrame = field(default_factory=pd.DataFrame)
    coproduct_feed_handoff: pd.DataFrame = field(default_factory=pd.DataFrame)
    residue_feed_competition_handoff: pd.DataFrame = field(default_factory=pd.DataFrame)
    energy_crop_land_requirement_by_country_year: Dict[Tuple[str, int], float] = field(default_factory=dict)
    crop_residue_management_multiplier: Dict[Tuple[str, str, str, int], float] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(
            self.crop_demand_by_country_comm_year
            or isinstance(self.scenario_detail, pd.DataFrame) and not self.scenario_detail.empty
        )


def normalize_m49(value: Any) -> str:
    """Normalize an M49 value to the model's apostrophe-prefixed key."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    if text.lower() in {"world", "global", "all"}:
        return "World"
    if text.startswith("'"):
        text = text[1:]
    if text.count(".") == 1:
        left, right = text.split(".", 1)
        if left.isdigit() and right.strip("0") == "":
            text = left
    if text.isdigit():
        return f"'{text.zfill(3)}"
    return f"'{text}"


def _empty_detail() -> pd.DataFrame:
    return pd.DataFrame(columns=DETAIL_COLUMNS)


def _empty_resource() -> pd.DataFrame:
    return pd.DataFrame(columns=P1_RESOURCE_COLUMNS)


def _read_table(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return read_excel_cached(str(p), sheet_name=0)
    return read_csv_cached(str(p), low_memory=False)


def _find_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        col = by_lower.get(str(alias).strip().lower())
        if col is not None:
            return col
    return None


def _copy_alias(df: pd.DataFrame, target: str, aliases: Iterable[str], default: Any = np.nan) -> None:
    col = _find_column(df, [target, *aliases])
    if col is None:
        df[target] = default
    elif col != target:
        df[target] = df[col]


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "market", "crop"}:
        return True
    if text in {"0", "false", "no", "n", "off", "resource", "nonmarket"}:
        return False
    return default


def _normalize_scenario_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    canonical = text.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "low": "low_bioenergy",
        "low_biomass": "low_bioenergy",
        "low_bioenergy": "low_bioenergy",
        "medium": "medium_bioenergy",
        "mid": "medium_bioenergy",
        "median": "medium_bioenergy",
        "medium_biomass": "medium_bioenergy",
        "medium_bioenergy": "medium_bioenergy",
        "high": "high_bioenergy",
        "high_biomass": "high_bioenergy",
        "high_bioenergy": "high_bioenergy",
    }
    return aliases.get(canonical, text)


def _is_default_scenario_label(value: Any) -> bool:
    return _normalize_scenario_label(value).lower() in {"", "all", "*", "default", "any"}


def _select_scenario_rows(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    """Select exact scenario rows plus default rows, with exact rows winning."""
    if df.empty or "scenario" not in df.columns:
        return df.copy()
    scenario_key = _normalize_scenario_label(scenario_name)
    work = df.copy()
    labels = work["scenario"].map(_normalize_scenario_label)
    exact = labels.eq(scenario_key)
    default = labels.map(_is_default_scenario_label)
    selected = work[exact | default].copy()
    if selected.empty:
        return selected
    selected["_scenario_priority"] = np.where(
        selected["scenario"].map(_normalize_scenario_label).eq(scenario_key),
        1,
        0,
    )
    key_cols = [
        c
        for c in [
            "M49_Country_Code",
            "year",
            "carrier",
            "feedstock",
            "model_commodity",
        ]
        if c in selected.columns
    ]
    if key_cols:
        selected = (
            selected.sort_values("_scenario_priority")
            .drop_duplicates(subset=key_cols, keep="last")
            .drop(columns=["_scenario_priority"], errors="ignore")
        )
    else:
        selected = selected.drop(columns=["_scenario_priority"], errors="ignore")
    selected["scenario"] = scenario_key
    return selected.reset_index(drop=True)


def _to_numeric_series(df: pd.DataFrame, col: str, default: Any = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _fill_text_column(df: pd.DataFrame, col: str, default: str = "") -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str).str.strip()


def _fill_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")


def load_feedstock_parameters(path: Optional[str]) -> pd.DataFrame:
    """Load and normalize feedstock conversion/resource parameters."""
    raw = _read_table(path)
    if raw.empty:
        return pd.DataFrame(columns=PARAMETER_COLUMNS)
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    _copy_alias(df, "feedstock", ["biomass_type", "feedstock_name", "item"])
    _copy_alias(df, "feedstock_category", ["category", "resource_class"])
    _copy_alias(df, "model_commodity", ["commodity", "Item_Emis"])
    _copy_alias(df, "market_link", ["link_to_market", "is_market_crop"], False)
    _copy_alias(df, "carrier", ["biofuel", "energy_carrier"], "")
    _copy_alias(df, "parent_commodity", ["source_commodity", "residue_parent_commodity", "crop_source"], "")
    _copy_alias(df, "sustainable_fraction", ["sustainable_removal_fraction", "removal_fraction", "collectable_fraction"], np.nan)
    _copy_alias(df, "yield_tdm_per_ha", ["yield_dm_t_per_ha", "biomass_yield_tdm_ha", "energy_crop_yield_tdm_per_ha"], np.nan)
    _copy_alias(df, "ghg_direct_kgco2e_per_tdm", ["direct_ef_kgco2e_per_tdm", "direct_emission_factor_kgco2e_per_tdm"], np.nan)
    _copy_alias(df, "ghg_soil_kgco2e_per_tdm", ["soil_carbon_ef_kgco2e_per_tdm", "soil_ef_kgco2e_per_tdm"], np.nan)
    _copy_alias(df, "ghg_avoided_kgco2e_per_tdm", ["avoided_ef_kgco2e_per_tdm", "methane_avoidance_kgco2e_per_tdm"], np.nan)
    _copy_alias(df, "fossil_displacement_kgco2e_per_tj", ["displacement_kgco2e_per_tj", "fossil_credit_kgco2e_per_tj"], np.nan)
    _copy_alias(df, "beccs_capture_kgco2_per_tdm", ["capture_kgco2_per_tdm", "beccs_capture_rate_kgco2_per_tdm"], np.nan)
    for q_col, aliases in RESIDUE_QUALITY_ALIASES.items():
        _copy_alias(df, q_col, aliases, np.nan)
    _copy_alias(df, "lhv_gj_per_tdm", ["lhv", "lower_heating_value_gj_per_tdm"])
    _copy_alias(df, "conversion_efficiency", ["efficiency", "conversion_eff"], 1.0)
    _copy_alias(df, "dry_matter_fraction", ["dm_fraction", "dry_matter"], 1.0)
    _copy_alias(df, "source", ["source_url", "reference"], "")
    _copy_alias(df, "notes", ["comment"], "")

    df["feedstock"] = df["feedstock"].fillna("").astype(str).str.strip()
    df = df[df["feedstock"] != ""].copy()
    df["feedstock_category"] = df["feedstock_category"].fillna("").astype(str).str.strip().str.lower()
    df["model_commodity"] = df["model_commodity"].fillna("").astype(str).str.strip()
    df["carrier"] = df["carrier"].fillna("").astype(str).str.strip()
    df["parent_commodity"] = df["parent_commodity"].fillna("").astype(str).str.strip()
    for col, default in [
        ("lhv_gj_per_tdm", np.nan),
        ("conversion_efficiency", 1.0),
        ("dry_matter_fraction", 1.0),
        ("sustainable_fraction", np.nan),
        ("yield_tdm_per_ha", np.nan),
        ("ghg_direct_kgco2e_per_tdm", np.nan),
        ("ghg_soil_kgco2e_per_tdm", np.nan),
        ("ghg_avoided_kgco2e_per_tdm", np.nan),
        ("fossil_displacement_kgco2e_per_tj", np.nan),
        ("beccs_capture_kgco2_per_tdm", np.nan),
        *[(q_col, np.nan) for q_col in RESIDUE_QUALITY_COLUMNS],
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df["conversion_efficiency"] = df["conversion_efficiency"].clip(lower=0.0, upper=1.0)
    df["dry_matter_fraction"] = df["dry_matter_fraction"].clip(lower=1e-9, upper=1.0)
    df["sustainable_fraction"] = df["sustainable_fraction"].clip(lower=0.0, upper=1.0)
    inferred_market = df["model_commodity"].ne("") & df["feedstock_category"].isin(
        {"crop", "food_crop", "feed_crop", "conventional_crop", "first_generation_crop"}
    )
    df["market_link"] = [
        _to_bool(raw_val, bool(inferred))
        for raw_val, inferred in zip(df["market_link"], inferred_market)
    ]
    return df[PARAMETER_COLUMNS].drop_duplicates(subset=["feedstock"], keep="last").reset_index(drop=True)


def _normalize_demand_rows(
    path: Optional[str],
    *,
    universe: Universe,
    default_scenario: str,
) -> pd.DataFrame:
    raw = _read_table(path)
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    _copy_alias(df, "scenario", ["scenario_id", "pathway"], default_scenario)
    _copy_alias(df, "M49_Country_Code", ["m49", "country_code", "region"], "")
    _copy_alias(df, "country_name", ["country", "area"], "")
    _copy_alias(df, "year", ["Year"])
    _copy_alias(df, "carrier", ["biofuel", "energy_carrier"], "")
    _copy_alias(df, "feedstock", ["biomass_type", "feedstock_name", "item"])
    _copy_alias(df, "feedstock_category", ["category", "resource_class"], "")
    _copy_alias(df, "model_commodity", ["commodity", "Item_Emis"], "")
    _copy_alias(df, "market_link", ["link_to_market", "is_market_crop"], np.nan)
    _copy_alias(df, "energy_basis", ["basis"], "final")
    _copy_alias(df, "parent_commodity", ["source_commodity", "residue_parent_commodity", "crop_source"], "")
    _copy_alias(df, "energy_target_tj", ["energy_tj", "target_tj", "bioenergy_tj"], np.nan)
    _copy_alias(df, "feedstock_demand_t", ["demand_t", "feedstock_t", "commodity_demand_t"], np.nan)
    _copy_alias(df, "feedstock_demand_tdm", ["demand_tdm", "feedstock_tdm"], np.nan)
    _copy_alias(df, "share", ["feedstock_share", "allocation_share"], 1.0)
    _copy_alias(df, "resource_available_t", ["available_t", "potential_t", "max_supply_t"], np.nan)
    _copy_alias(df, "resource_available_tdm", ["available_tdm", "potential_tdm", "max_supply_tdm", "sustainable_supply_tdm"], np.nan)
    _copy_alias(df, "competing_use_t", ["existing_use_t", "nonenergy_use_t"], np.nan)
    _copy_alias(df, "competing_use_tdm", ["existing_use_tdm", "nonenergy_use_tdm"], np.nan)
    _copy_alias(df, "sustainable_fraction", ["sustainable_removal_fraction", "removal_fraction", "collectable_fraction"], np.nan)
    _copy_alias(df, "yield_tdm_per_ha", ["yield_dm_t_per_ha", "biomass_yield_tdm_ha", "energy_crop_yield_tdm_per_ha"], np.nan)
    _copy_alias(df, "eligible_land_area_ha", ["eligible_area_ha", "available_land_ha", "land_available_ha"], np.nan)
    _copy_alias(df, "energy_crop_area_target_ha", ["cropland_reallocation_target_ha", "energy_crop_land_target_ha"], np.nan)
    _copy_alias(df, "ghg_direct_kgco2e_per_tdm", ["direct_ef_kgco2e_per_tdm", "direct_emission_factor_kgco2e_per_tdm"], np.nan)
    _copy_alias(df, "ghg_soil_kgco2e_per_tdm", ["soil_carbon_ef_kgco2e_per_tdm", "soil_ef_kgco2e_per_tdm"], np.nan)
    _copy_alias(df, "ghg_avoided_kgco2e_per_tdm", ["avoided_ef_kgco2e_per_tdm", "methane_avoidance_kgco2e_per_tdm"], np.nan)
    _copy_alias(df, "fossil_displacement_kgco2e_per_tj", ["displacement_kgco2e_per_tj", "fossil_credit_kgco2e_per_tj"], np.nan)
    _copy_alias(df, "beccs_capture_kgco2_per_tdm", ["capture_kgco2_per_tdm", "beccs_capture_rate_kgco2_per_tdm"], np.nan)
    for q_col, aliases in RESIDUE_QUALITY_ALIASES.items():
        _copy_alias(df, q_col, aliases, np.nan)
    _copy_alias(df, "lhv_gj_per_tdm", ["lhv", "lower_heating_value_gj_per_tdm"], np.nan)
    _copy_alias(df, "conversion_efficiency", ["efficiency", "conversion_eff"], np.nan)
    _copy_alias(df, "dry_matter_fraction", ["dm_fraction", "dry_matter"], np.nan)
    _copy_alias(df, "coproduct_feed_commodity", ["co_product_feed_commodity", "coproduct_commodity"], "")
    _copy_alias(df, "coproduct_feed_credit_tdm", ["co_product_feed_credit_tdm", "coproduct_credit_tdm"], np.nan)
    _copy_alias(df, "residue_feed_competition_tdm", ["residue_feed_competition_dm_t", "residue_feed_use_tdm"], np.nan)
    _copy_alias(df, "source", ["source_url", "reference"], "")
    _copy_alias(df, "notes", ["comment"], "")

    df["scenario"] = df["scenario"].fillna(default_scenario).astype(str).str.strip()
    df["feedstock"] = df["feedstock"].fillna("").astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["feedstock"] != "") & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    country_name = df["country_name"].fillna("").astype(str).str.strip()
    m49 = df["M49_Country_Code"].apply(normalize_m49)
    name_to_m49 = universe.m49_by_country or {}
    mapped_name = country_name.map(name_to_m49).fillna("")
    m49 = m49.where(m49.ne(""), mapped_name)
    m49 = m49.where(m49.ne(""), country_name.apply(normalize_m49))
    df["M49_Country_Code"] = m49
    df["country_name"] = df["M49_Country_Code"].map(universe.country_by_m49).fillna(country_name)
    df = df[df["M49_Country_Code"].ne("")].copy()

    for col in [
        "energy_target_tj",
        "feedstock_demand_t",
        "feedstock_demand_tdm",
        "share",
        "resource_available_t",
        "resource_available_tdm",
        "competing_use_t",
        "competing_use_tdm",
        "sustainable_fraction",
        "yield_tdm_per_ha",
        "eligible_land_area_ha",
        "energy_crop_area_target_ha",
        "ghg_direct_kgco2e_per_tdm",
        "ghg_soil_kgco2e_per_tdm",
        "ghg_avoided_kgco2e_per_tdm",
        "fossil_displacement_kgco2e_per_tj",
        "beccs_capture_kgco2_per_tdm",
        *RESIDUE_QUALITY_COLUMNS,
        "lhv_gj_per_tdm",
        "conversion_efficiency",
        "dry_matter_fraction",
        "coproduct_feed_credit_tdm",
        "residue_feed_competition_tdm",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["share"] = df["share"].fillna(1.0).clip(lower=0.0)
    df["parent_commodity"] = df["parent_commodity"].fillna("").astype(str).str.strip()
    df["coproduct_feed_commodity"] = df["coproduct_feed_commodity"].fillna("").astype(str).str.strip()
    return df


def _normalize_resource_rows(
    path: Optional[str],
    *,
    universe: Universe,
    default_scenario: str,
) -> pd.DataFrame:
    raw = _read_table(path)
    if raw.empty:
        return _empty_resource()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    _copy_alias(df, "scenario", ["scenario_id", "pathway"], default_scenario)
    _copy_alias(df, "M49_Country_Code", ["m49", "country_code", "region"], "")
    _copy_alias(df, "country_name", ["country", "area"], "")
    _copy_alias(df, "year", ["Year"])
    _copy_alias(df, "feedstock", ["biomass_type", "feedstock_name", "item"])
    _copy_alias(df, "feedstock_category", ["category", "resource_class"], "")
    _copy_alias(df, "parent_commodity", ["source_commodity", "residue_parent_commodity", "crop_source"], "")
    _copy_alias(df, "resource_available_t", ["available_t", "potential_t", "max_supply_t"], np.nan)
    _copy_alias(df, "resource_available_tdm", ["available_tdm", "potential_tdm", "max_supply_tdm", "sustainable_supply_tdm"], np.nan)
    _copy_alias(df, "competing_use_t", ["existing_use_t", "nonenergy_use_t"], np.nan)
    _copy_alias(df, "competing_use_tdm", ["existing_use_tdm", "nonenergy_use_tdm"], np.nan)
    _copy_alias(df, "sustainable_fraction", ["sustainable_removal_fraction", "removal_fraction", "collectable_fraction"], np.nan)
    _copy_alias(df, "yield_tdm_per_ha", ["yield_dm_t_per_ha", "biomass_yield_tdm_ha", "energy_crop_yield_tdm_per_ha"], np.nan)
    _copy_alias(df, "eligible_land_area_ha", ["eligible_area_ha", "available_land_ha", "land_available_ha"], np.nan)
    _copy_alias(df, "ghg_direct_kgco2e_per_tdm", ["direct_ef_kgco2e_per_tdm", "direct_emission_factor_kgco2e_per_tdm"], np.nan)
    _copy_alias(df, "ghg_soil_kgco2e_per_tdm", ["soil_carbon_ef_kgco2e_per_tdm", "soil_ef_kgco2e_per_tdm"], np.nan)
    _copy_alias(df, "ghg_avoided_kgco2e_per_tdm", ["avoided_ef_kgco2e_per_tdm", "methane_avoidance_kgco2e_per_tdm"], np.nan)
    _copy_alias(df, "fossil_displacement_kgco2e_per_tj", ["displacement_kgco2e_per_tj", "fossil_credit_kgco2e_per_tj"], np.nan)
    _copy_alias(df, "beccs_capture_kgco2_per_tdm", ["capture_kgco2_per_tdm", "beccs_capture_rate_kgco2_per_tdm"], np.nan)
    for q_col, aliases in RESIDUE_QUALITY_ALIASES.items():
        _copy_alias(df, q_col, aliases, np.nan)
    _copy_alias(df, "source", ["source_url", "reference"], "")
    _copy_alias(df, "notes", ["comment"], "")

    df["scenario"] = df["scenario"].fillna(default_scenario).astype(str).str.strip()
    df["feedstock"] = df["feedstock"].fillna("").astype(str).str.strip()
    df["feedstock_category"] = df["feedstock_category"].fillna("").astype(str).str.strip().str.lower()
    df["parent_commodity"] = df["parent_commodity"].fillna("").astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["feedstock"] != "") & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    country_name = df["country_name"].fillna("").astype(str).str.strip()
    m49 = df["M49_Country_Code"].apply(normalize_m49)
    name_to_m49 = universe.m49_by_country or {}
    mapped_name = country_name.map(name_to_m49).fillna("")
    m49 = m49.where(m49.ne(""), mapped_name)
    m49 = m49.where(m49.ne(""), country_name.apply(normalize_m49))
    df["M49_Country_Code"] = m49
    df["country_name"] = df["M49_Country_Code"].map(universe.country_by_m49).fillna(country_name)
    df = df[df["M49_Country_Code"].ne("")].copy()

    _fill_numeric_columns(df, [c for c in P1_RESOURCE_COLUMNS if c not in {
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "feedstock",
        "feedstock_category",
        "parent_commodity",
        "source",
        "notes",
    }])
    for col in ["source", "notes"]:
        _fill_text_column(df, col)
    return df[P1_RESOURCE_COLUMNS].reset_index(drop=True)


def _merge_parameters(rows: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    if parameters.empty:
        out = rows.copy()
    else:
        param = parameters.add_suffix("_param").rename(columns={"feedstock_param": "feedstock"})
        out = rows.merge(param, on="feedstock", how="left")
        for col in [
            "feedstock_category",
            "model_commodity",
            "market_link",
            "carrier",
            "parent_commodity",
            "sustainable_fraction",
            "yield_tdm_per_ha",
            "ghg_direct_kgco2e_per_tdm",
            "ghg_soil_kgco2e_per_tdm",
            "ghg_avoided_kgco2e_per_tdm",
            "fossil_displacement_kgco2e_per_tj",
            "beccs_capture_kgco2_per_tdm",
            *RESIDUE_QUALITY_COLUMNS,
            "lhv_gj_per_tdm",
            "conversion_efficiency",
            "dry_matter_fraction",
            "source",
            "notes",
        ]:
            pcol = f"{col}_param"
            if pcol not in out.columns:
                continue
            if col in {
                "lhv_gj_per_tdm",
                "conversion_efficiency",
                "dry_matter_fraction",
                "sustainable_fraction",
                "yield_tdm_per_ha",
                "ghg_direct_kgco2e_per_tdm",
                "ghg_soil_kgco2e_per_tdm",
                "ghg_avoided_kgco2e_per_tdm",
                "fossil_displacement_kgco2e_per_tj",
                "beccs_capture_kgco2_per_tdm",
                *RESIDUE_QUALITY_COLUMNS,
            }:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
                    pd.to_numeric(out[pcol], errors="coerce")
                )
            elif col == "market_link":
                out[col] = out[col].where(out[col].notna(), out[pcol])
            else:
                current = out[col].fillna("").astype(str).str.strip()
                replacement = out[pcol].fillna("").astype(str).str.strip()
                out[col] = current.where(current.ne(""), replacement)
        out = out.drop(columns=[c for c in out.columns if c.endswith("_param")], errors="ignore")
    out["feedstock_category"] = out["feedstock_category"].fillna("").astype(str).str.strip().str.lower()
    out["model_commodity"] = out["model_commodity"].fillna("").astype(str).str.strip()
    out["carrier"] = out["carrier"].fillna("").astype(str).str.strip()
    _fill_text_column(out, "parent_commodity")
    out["energy_basis"] = out["energy_basis"].fillna("final").astype(str).str.strip().str.lower()
    out["conversion_efficiency"] = pd.to_numeric(out["conversion_efficiency"], errors="coerce").fillna(1.0)
    out["dry_matter_fraction"] = pd.to_numeric(out["dry_matter_fraction"], errors="coerce").fillna(1.0)
    out["lhv_gj_per_tdm"] = pd.to_numeric(out["lhv_gj_per_tdm"], errors="coerce")
    _fill_numeric_columns(out, [
        "sustainable_fraction",
        "yield_tdm_per_ha",
        "ghg_direct_kgco2e_per_tdm",
        "ghg_soil_kgco2e_per_tdm",
        "ghg_avoided_kgco2e_per_tdm",
        "fossil_displacement_kgco2e_per_tj",
        "beccs_capture_kgco2_per_tdm",
    ])
    out["conversion_efficiency"] = out["conversion_efficiency"].clip(lower=0.0, upper=1.0)
    out["dry_matter_fraction"] = out["dry_matter_fraction"].clip(lower=1e-9, upper=1.0)
    out["sustainable_fraction"] = out["sustainable_fraction"].clip(lower=0.0, upper=1.0)
    inferred_market = out["model_commodity"].ne("") & out["feedstock_category"].isin(
        {"crop", "food_crop", "feed_crop", "conventional_crop", "first_generation_crop"}
    )
    out["market_link"] = [
        _to_bool(raw_val, bool(inferred))
        for raw_val, inferred in zip(out["market_link"], inferred_market)
    ]
    return out


def _calculate_physical_quantities(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return _empty_detail()
    out = rows.copy()
    _fill_text_column(out, "parent_commodity")
    _fill_numeric_columns(out, P1_NUMERIC_COLUMNS)
    allocated_energy = pd.to_numeric(out["energy_target_tj"], errors="coerce") * out["share"].fillna(1.0)
    wet_t = pd.to_numeric(out["feedstock_demand_t"], errors="coerce")
    dry_t = pd.to_numeric(out["feedstock_demand_tdm"], errors="coerce")
    dm_fraction = out["dry_matter_fraction"].clip(lower=1e-9)
    lhv = out["lhv_gj_per_tdm"]
    efficiency = out["conversion_efficiency"].clip(lower=0.0)

    dry_t = dry_t.where(dry_t.notna(), wet_t * dm_fraction)
    wet_t = wet_t.where(wet_t.notna(), dry_t / dm_fraction)
    usable_gj_per_tdm = lhv * efficiency
    derived_dry = allocated_energy * 1000.0 / usable_gj_per_tdm.replace(0.0, np.nan)
    dry_t = dry_t.where(dry_t.notna(), derived_dry)
    wet_t = wet_t.where(wet_t.notna(), dry_t / dm_fraction)
    supplied_tj = dry_t * usable_gj_per_tdm / 1000.0

    out["energy_target_tj"] = allocated_energy
    out["feedstock_demand_t"] = wet_t
    out["feedstock_demand_tdm"] = dry_t
    out["energy_supplied_tj"] = supplied_tj
    out["energy_gap_tj"] = allocated_energy - supplied_tj
    out["resource_available_tdm"] = out["resource_available_tdm"].where(
        out["resource_available_tdm"].notna(),
        out["resource_available_t"] * dm_fraction,
    )
    out["resource_available_t"] = out["resource_available_t"].where(
        out["resource_available_t"].notna(),
        out["resource_available_tdm"] / dm_fraction,
    )
    out["competing_use_tdm"] = out["competing_use_tdm"].where(
        out["competing_use_tdm"].notna(),
        out["competing_use_t"] * dm_fraction,
    )
    out["competing_use_t"] = out["competing_use_t"].where(
        out["competing_use_t"].notna(),
        out["competing_use_tdm"] / dm_fraction,
    )
    out["allocation_status"] = np.where(
        dry_t.notna(),
        "resolved",
        "missing_mass_or_conversion_parameter",
    )
    for col in DETAIL_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[DETAIL_COLUMNS]


def _latest_historical_rows(detail: pd.DataFrame, hist_end_year: int) -> pd.DataFrame:
    if detail.empty:
        return detail
    work = detail[pd.to_numeric(detail["year"], errors="coerce") <= int(hist_end_year)].copy()
    if work.empty:
        return work
    group_cols = [
        "M49_Country_Code",
        "carrier",
        "feedstock",
        "feedstock_category",
        "model_commodity",
        "market_link",
    ]
    max_year = work.groupby(group_cols, dropna=False)["year"].transform("max")
    return work[work["year"].eq(max_year)].copy()


def _interpolate_detail(detail: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    target_years = sorted({int(y) for y in years})
    if detail.empty or not target_years:
        return _empty_detail()
    group_cols = [
        "scenario",
        "M49_Country_Code",
        "country_name",
        "carrier",
        "feedstock",
        "feedstock_category",
        "model_commodity",
        "market_link",
        "energy_basis",
        "parent_commodity",
        "lhv_gj_per_tdm",
        "conversion_efficiency",
        "dry_matter_fraction",
        "coproduct_feed_commodity",
        "source",
        "notes",
    ]
    value_cols = [
        "energy_target_tj",
        "feedstock_demand_t",
        "feedstock_demand_tdm",
        "energy_supplied_tj",
        "energy_gap_tj",
        "resource_available_t",
        "resource_available_tdm",
        "competing_use_t",
        "competing_use_tdm",
        "sustainable_fraction",
        "yield_tdm_per_ha",
        "eligible_land_area_ha",
        "energy_crop_area_target_ha",
        "ghg_direct_kgco2e_per_tdm",
        "ghg_soil_kgco2e_per_tdm",
        "ghg_avoided_kgco2e_per_tdm",
        "fossil_displacement_kgco2e_per_tj",
        "beccs_capture_kgco2_per_tdm",
        "coproduct_feed_credit_tdm",
        "residue_feed_competition_tdm",
        *RESIDUE_QUALITY_COLUMNS,
    ]
    frames: List[pd.DataFrame] = []
    for keys, group in detail.groupby(group_cols, dropna=False, sort=False):
        g = group.copy()
        g["year"] = pd.to_numeric(g["year"], errors="coerce")
        g = g.dropna(subset=["year"]).sort_values("year")
        if g.empty:
            continue
        years_known = g["year"].to_numpy(dtype=float)
        frame = pd.DataFrame({"year": target_years})
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            valid = vals.notna()
            if not valid.any():
                frame[col] = np.nan
            elif valid.sum() == 1:
                frame[col] = float(vals[valid].iloc[0])
            else:
                frame[col] = np.interp(
                    np.asarray(target_years, dtype=float),
                    years_known[valid.to_numpy()],
                    vals[valid].to_numpy(dtype=float),
                )
        for col, value in zip(group_cols, keys if isinstance(keys, tuple) else (keys,)):
            frame[col] = value
        frame["allocation_status"] = "interpolated"
        frames.append(frame)
    if not frames:
        return _empty_detail()
    out = pd.concat(frames, ignore_index=True)
    for col in DETAIL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan if col in P1_NUMERIC_COLUMNS else ""
    return out[DETAIL_COLUMNS]


def _interpolate_resource_rows(resources: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    target_years = sorted({int(y) for y in years})
    if resources.empty or not target_years:
        return _empty_resource()
    group_cols = [
        "scenario",
        "M49_Country_Code",
        "country_name",
        "feedstock",
        "feedstock_category",
        "parent_commodity",
        "source",
        "notes",
    ]
    value_cols = [
        "resource_available_t",
        "resource_available_tdm",
        "competing_use_t",
        "competing_use_tdm",
        "sustainable_fraction",
        "yield_tdm_per_ha",
        "eligible_land_area_ha",
        "ghg_direct_kgco2e_per_tdm",
        "ghg_soil_kgco2e_per_tdm",
        "ghg_avoided_kgco2e_per_tdm",
        "fossil_displacement_kgco2e_per_tj",
        "beccs_capture_kgco2_per_tdm",
        *RESIDUE_QUALITY_COLUMNS,
    ]
    frames: List[pd.DataFrame] = []
    for keys, group in resources.groupby(group_cols, dropna=False, sort=False):
        g = group.copy()
        g["year"] = pd.to_numeric(g["year"], errors="coerce")
        g = g.dropna(subset=["year"]).sort_values("year")
        if g.empty:
            continue
        years_known = g["year"].to_numpy(dtype=float)
        frame = pd.DataFrame({"year": target_years})
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            valid = vals.notna()
            if not valid.any():
                frame[col] = np.nan
            elif valid.sum() == 1:
                frame[col] = float(vals[valid].iloc[0])
            else:
                frame[col] = np.interp(
                    np.asarray(target_years, dtype=float),
                    years_known[valid.to_numpy()],
                    vals[valid].to_numpy(dtype=float),
                )
        for col, value in zip(group_cols, keys if isinstance(keys, tuple) else (keys,)):
            frame[col] = value
        frames.append(frame)
    if not frames:
        return _empty_resource()
    return pd.concat(frames, ignore_index=True)[P1_RESOURCE_COLUMNS]


def _fill_from_resource_columns(out: pd.DataFrame, columns: Iterable[str], suffix: str = "_resource") -> pd.DataFrame:
    for col in columns:
        rcol = f"{col}{suffix}"
        if rcol not in out.columns:
            continue
        if col in {"feedstock_category", "parent_commodity", "source", "notes"}:
            current = out[col].fillna("").astype(str).str.strip()
            replacement = out[rcol].fillna("").astype(str).str.strip()
            out[col] = current.where(current.ne(""), replacement)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
                pd.to_numeric(out[rcol], errors="coerce")
            )
    return out.drop(columns=[c for c in out.columns if c.endswith(suffix)], errors="ignore")


def _apply_resource_rows(detail: pd.DataFrame, resources: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    out = detail.copy()
    _fill_text_column(out, "parent_commodity")
    _fill_numeric_columns(out, P1_NUMERIC_COLUMNS)
    if resources.empty:
        return _apply_resource_constraints(out)

    res = resources.copy()
    _fill_numeric_columns(res, [c for c in P1_RESOURCE_COLUMNS if c not in {
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "feedstock",
        "feedstock_category",
        "parent_commodity",
        "source",
        "notes",
    }])
    exact_cols = ["M49_Country_Code", "year", "feedstock"]
    exact = res[res["M49_Country_Code"].ne("World")].copy()
    if not exact.empty:
        exact = exact.sort_values(["M49_Country_Code", "feedstock", "year"]).drop_duplicates(
            subset=exact_cols,
            keep="last",
        )
        merge_cols = exact_cols + [
            c
            for c in [
                *P1_RESOURCE_ABSOLUTE_COLUMNS,
                *P1_RESOURCE_RATE_COLUMNS,
            ]
            if c in exact.columns and c not in exact_cols
        ]
        out = out.merge(
            exact[merge_cols],
            on=exact_cols,
            how="left",
            suffixes=("", "_resource"),
        )
        out = _fill_from_resource_columns(
            out,
            [*P1_RESOURCE_ABSOLUTE_COLUMNS, *P1_RESOURCE_RATE_COLUMNS],
        )

    world = res[res["M49_Country_Code"].eq("World")].copy()
    if not world.empty:
        world_cols = ["year", "feedstock"]
        world = world.sort_values(["feedstock", "year"]).drop_duplicates(
            subset=world_cols,
            keep="last",
        )
        merge_cols = world_cols + [
            c
            for c in P1_RESOURCE_RATE_COLUMNS
            if c in world.columns and c not in world_cols
        ]
        out = out.merge(
            world[merge_cols],
            on=world_cols,
            how="left",
            suffixes=("", "_resource"),
        )
        out = _fill_from_resource_columns(out, P1_RESOURCE_RATE_COLUMNS)

    return _apply_resource_constraints(out)


def _downscale_world_rows(
    scenario_detail: pd.DataFrame,
    historical_base: pd.DataFrame,
    *,
    universe: Universe,
) -> Tuple[pd.DataFrame, List[str]]:
    diagnostics: List[str] = []
    if scenario_detail.empty or not scenario_detail["M49_Country_Code"].eq("World").any():
        return scenario_detail, diagnostics
    national = scenario_detail[~scenario_detail["M49_Country_Code"].eq("World")].copy()
    world = scenario_detail[scenario_detail["M49_Country_Code"].eq("World")].copy()
    frames = [national] if not national.empty else []
    for row in world.itertuples(index=False):
        hist = historical_base[
            historical_base["feedstock"].astype(str).eq(str(row.feedstock))
        ].copy()
        if hist.empty and str(row.model_commodity).strip():
            hist = historical_base[
                historical_base["model_commodity"].astype(str).eq(str(row.model_commodity))
            ].copy()
        hist = hist[hist["M49_Country_Code"].isin(set(universe.countries))]
        weights = pd.to_numeric(hist["energy_supplied_tj"], errors="coerce").fillna(0.0)
        if weights.sum() <= 0:
            weights = pd.to_numeric(hist["feedstock_demand_tdm"], errors="coerce").fillna(0.0)
        if weights.sum() <= 0:
            diagnostics.append(
                f"World row not downscaled: feedstock={row.feedstock}, year={row.year}; "
                "no positive historical country weights."
            )
            unresolved = pd.DataFrame([row._asdict()])
            unresolved["allocation_status"] = "unresolved_world_no_historical_weights"
            frames.append(unresolved)
            continue
        hist = hist.assign(_weight=weights / weights.sum())
        for _, hist_row in hist.iterrows():
            rec = row._asdict()
            country_code = str(hist_row["M49_Country_Code"])
            rec["M49_Country_Code"] = country_code
            rec["country_name"] = universe.country_by_m49.get(country_code, "")
            for col in [
                "energy_target_tj",
                "feedstock_demand_t",
                "feedstock_demand_tdm",
                "energy_supplied_tj",
                "energy_gap_tj",
                "resource_available_t",
                "resource_available_tdm",
                "competing_use_t",
                "competing_use_tdm",
                "eligible_land_area_ha",
            ]:
                value = rec.get(col)
                rec[col] = float(value) * float(hist_row["_weight"]) if pd.notna(value) else np.nan
            rec["allocation_status"] = "world_downscaled_historical_share"
            frames.append(pd.DataFrame([rec]))
    if not frames:
        return _empty_detail(), diagnostics
    return pd.concat(frames, ignore_index=True)[DETAIL_COLUMNS], diagnostics


def _is_market_linked(series: pd.Series) -> pd.Series:
    return series.map(lambda value: _to_bool(value, False))


def _is_dedicated_energy_crop_category(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text in DEDICATED_ENERGY_CROP_CATEGORIES:
        return True
    return "energy_crop" in text or "dedicated" in text


def _apply_resource_constraints(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    out = detail.copy()
    _fill_text_column(out, "parent_commodity")
    _fill_numeric_columns(out, P1_NUMERIC_COLUMNS)

    demand_dry = pd.to_numeric(out["feedstock_demand_tdm"], errors="coerce")
    demand_wet = pd.to_numeric(out["feedstock_demand_t"], errors="coerce")
    dm_fraction = pd.to_numeric(out["dry_matter_fraction"], errors="coerce").fillna(1.0).clip(lower=1e-9)
    lhv = pd.to_numeric(out["lhv_gj_per_tdm"], errors="coerce")
    efficiency = pd.to_numeric(out["conversion_efficiency"], errors="coerce").fillna(1.0).clip(lower=0.0)
    usable_gj_per_tdm = lhv * efficiency
    energy_target = pd.to_numeric(out["energy_target_tj"], errors="coerce")
    energy_demand = energy_target.where(
        energy_target.notna(),
        demand_dry * usable_gj_per_tdm / 1000.0,
    )

    resource_available_dry = pd.to_numeric(out["resource_available_tdm"], errors="coerce")
    resource_available_wet = pd.to_numeric(out["resource_available_t"], errors="coerce")
    dedicated = out["feedstock_category"].map(_is_dedicated_energy_crop_category)
    energy_crop_yield = pd.to_numeric(out["yield_tdm_per_ha"], errors="coerce")
    eligible_land = pd.to_numeric(out["eligible_land_area_ha"], errors="coerce")
    implied_energy_crop_supply = (eligible_land * energy_crop_yield).where(dedicated)
    resource_available_dry = resource_available_dry.where(
        resource_available_dry.notna(),
        implied_energy_crop_supply,
    )
    resource_available_dry = resource_available_dry.where(
        resource_available_dry.notna(),
        resource_available_wet * dm_fraction,
    )
    resource_available_wet = resource_available_wet.where(
        resource_available_wet.notna(),
        resource_available_dry / dm_fraction,
    )
    out["resource_available_tdm"] = resource_available_dry
    out["resource_available_t"] = resource_available_wet

    competing_dry = pd.to_numeric(out["competing_use_tdm"], errors="coerce")
    competing_wet = pd.to_numeric(out["competing_use_t"], errors="coerce")
    competing_dry = competing_dry.where(competing_dry.notna(), competing_wet * dm_fraction)
    competing_wet = competing_wet.where(competing_wet.notna(), competing_dry / dm_fraction)
    out["competing_use_tdm"] = competing_dry
    out["competing_use_t"] = competing_wet

    sustainable_fraction = pd.to_numeric(out["sustainable_fraction"], errors="coerce").fillna(1.0)
    sustainable_fraction = sustainable_fraction.clip(lower=0.0, upper=1.0)
    sustainable_supply = (resource_available_dry * sustainable_fraction - competing_dry.fillna(0.0)).clip(lower=0.0)
    out["sustainable_fraction"] = sustainable_fraction
    out["sustainable_supply_tdm"] = sustainable_supply

    market_link = _is_market_linked(out["market_link"])
    has_resource = resource_available_dry.notna()
    constrained = has_resource & ~market_link
    feasible_dry = demand_dry.copy()
    feasible_dry = feasible_dry.where(~constrained, np.minimum(demand_dry, sustainable_supply))
    feasible_wet = feasible_dry / dm_fraction
    feasible_energy = feasible_dry * usable_gj_per_tdm / 1000.0
    resource_gap_dry = (demand_dry - sustainable_supply).clip(lower=0.0).where(constrained)
    resource_gap_wet = resource_gap_dry / dm_fraction
    unmet_energy = (energy_demand - feasible_energy).clip(lower=0.0).where(constrained, 0.0)
    use_ratio = demand_dry / sustainable_supply.replace(0.0, np.nan)

    out["feasible_feedstock_demand_tdm"] = feasible_dry
    out["feasible_feedstock_demand_t"] = feasible_wet
    out["feasible_energy_supplied_tj"] = feasible_energy
    out["resource_gap_tdm"] = resource_gap_dry
    out["resource_gap_t"] = resource_gap_wet
    out["unmet_energy_tj"] = unmet_energy
    out["resource_use_ratio"] = use_ratio.where(has_resource)

    status = np.full(len(out), "uncapped_missing_resource", dtype=object)
    status = np.where(market_link, "market_linked_solver_demand_not_capped", status)
    status = np.where(has_resource & ~market_link, "within_resource_limit", status)
    status = np.where(constrained & resource_gap_dry.fillna(0.0).gt(0.0), "resource_overdraw", status)
    status = np.where(demand_dry.fillna(0.0).le(0.0), "no_positive_demand", status)
    out["resource_status"] = status

    land_requirement = demand_dry / energy_crop_yield.replace(0.0, np.nan)
    cropland_target = pd.to_numeric(out["energy_crop_area_target_ha"], errors="coerce")
    cropland_reallocation = np.minimum(land_requirement, cropland_target).where(
        dedicated & cropland_target.notna()
    )
    additional_expansion = (land_requirement - cropland_target).clip(lower=0.0).where(
        dedicated & cropland_target.notna(),
        land_requirement.where(dedicated),
    )
    feasible_land_requirement = feasible_dry / energy_crop_yield.replace(0.0, np.nan)
    feasible_additional_expansion = (feasible_land_requirement - cropland_target).clip(lower=0.0).where(
        dedicated & cropland_target.notna(),
        feasible_land_requirement.where(dedicated),
    )
    out["land_requirement_ha"] = land_requirement.where(dedicated)
    out["cropland_reallocation_area_ha"] = cropland_reallocation
    out["additional_land_expansion_ha"] = additional_expansion
    out["feasible_additional_land_expansion_ha"] = feasible_additional_expansion
    out["land_gap_ha"] = (out["land_requirement_ha"] - eligible_land).clip(lower=0.0).where(
        dedicated & eligible_land.notna()
    )
    return out[DETAIL_COLUMNS]


def _crop_demand_map(detail: pd.DataFrame, universe: Universe) -> Dict[COUNTRY_COMM_YEAR, float]:
    if detail.empty:
        return {}
    valid_countries = set(universe.countries)
    valid_commodities = set(universe.commodities)
    work = detail[
        detail["market_link"].astype(bool)
        & detail["M49_Country_Code"].isin(valid_countries)
        & detail["model_commodity"].isin(valid_commodities)
    ].copy()
    work["feedstock_demand_t"] = pd.to_numeric(work["feedstock_demand_t"], errors="coerce")
    work = work[work["feedstock_demand_t"].gt(0.0)]
    if work.empty:
        return {}
    grouped = work.groupby(
        ["M49_Country_Code", "model_commodity", "year"],
        as_index=False,
    )["feedstock_demand_t"].sum()
    return {
        (str(r.M49_Country_Code), str(r.model_commodity), int(r.year)): float(r.feedstock_demand_t)
        for r in grouped.itertuples(index=False)
    }


def _energy_balance(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "M49_Country_Code",
                "year",
                "carrier",
                "energy_target_tj",
                "energy_supplied_tj",
                "energy_gap_tj",
                "relative_gap",
            ]
        )
    work = detail.copy()
    for col in ["energy_target_tj", "energy_supplied_tj", "energy_gap_tj"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    grouped = work.groupby(
        ["scenario", "M49_Country_Code", "year", "carrier"],
        as_index=False,
    )[["energy_target_tj", "energy_supplied_tj", "energy_gap_tj"]].sum()
    denom = grouped["energy_target_tj"].abs().replace(0.0, np.nan)
    grouped["relative_gap"] = grouped["energy_gap_tj"] / denom
    return grouped


def _resource_balance(detail: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "feedstock",
        "feedstock_category",
        "model_commodity",
        "market_link",
        "feedstock_demand_tdm",
        "resource_available_tdm",
        "competing_use_tdm",
        "sustainable_fraction",
        "residue_carbon_pct",
        "residue_nitrogen_pct",
        "residue_quality_n_obs",
        "sustainable_supply_tdm",
        "feasible_feedstock_demand_tdm",
        "resource_gap_tdm",
        "energy_target_tj",
        "feasible_energy_supplied_tj",
        "unmet_energy_tj",
        "resource_use_ratio",
        "resource_status",
    ]
    if detail.empty:
        return pd.DataFrame(columns=columns)
    work = detail.copy()
    for col in [
        "feedstock_demand_tdm",
        "resource_available_tdm",
        "competing_use_tdm",
        "sustainable_fraction",
        "residue_carbon_pct",
        "residue_nitrogen_pct",
        "residue_quality_n_obs",
        "sustainable_supply_tdm",
        "feasible_feedstock_demand_tdm",
        "resource_gap_tdm",
        "energy_target_tj",
        "feasible_energy_supplied_tj",
        "unmet_energy_tj",
        "resource_use_ratio",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work[columns].sort_values(
        ["scenario", "M49_Country_Code", "year", "feedstock"],
        na_position="last",
    ).reset_index(drop=True)


def _is_crop_residue_like(feedstock: Any, category: Any) -> bool:
    category_text = str(category or "").strip().lower()
    feedstock_text = str(feedstock or "").strip().lower()
    if _is_dedicated_energy_crop_category(category_text):
        return False
    residue_tokens = (
        "crop_residue",
        "agricultural_residue",
        "agricultural residue",
        "residue",
        "straw",
        "stover",
        "husk",
        "bagasse",
        "bran",
        "shell",
    )
    return any(token in category_text for token in residue_tokens) or any(
        token in feedstock_text for token in residue_tokens
    )


def _target_feasibility(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate formal target / feasible / unmet bioenergy diagnostics.

    This is a pre-solver physical/resource diagnostic.  Resource and eligible
    land caps are assigned directly from feedstock rows.  Market/land solver
    gaps are left at zero here and may be populated by post-solver audits in S4.
    """
    columns = [
        "diagnostic_stage",
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "carrier",
        "feedstock_category_group",
        "target_energy_tj",
        "target_energy_ej",
        "feasible_supplied_tj",
        "feasible_supplied_ej",
        "unmet_energy_tj",
        "unmet_energy_ej",
        "non_crop_cap_unmet_tj",
        "non_crop_cap_unmet_ej",
        "crop_residue_cap_unmet_tj",
        "crop_residue_cap_unmet_ej",
        "eligible_land_cap_unmet_tj",
        "eligible_land_cap_unmet_ej",
        "unattributed_pre_solver_unmet_tj",
        "unattributed_pre_solver_unmet_ej",
        "market_land_solver_unmet_tj",
        "market_land_solver_unmet_ej",
        "attributed_unmet_tj",
        "attributed_unmet_ej",
        "attribution_residual_tj",
        "attribution_residual_ej",
        "dominant_gap_reason",
        "resource_overdraw_rows",
        "eligible_land_overdraw_rows",
        "market_linked_rows",
        "uncapped_missing_resource_rows",
        "notes",
    ]
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame(columns=columns)
    work = detail.copy()
    for col in [
        "energy_target_tj",
        "feasible_energy_supplied_tj",
        "unmet_energy_tj",
        "land_gap_ha",
        "feedstock_demand_tdm",
        "lhv_gj_per_tdm",
        "conversion_efficiency",
    ]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce").fillna(0.0)
    implied_target_tj = (
        work["feedstock_demand_tdm"]
        * work["lhv_gj_per_tdm"]
        * work["conversion_efficiency"].clip(lower=0.0)
        / 1000.0
    )
    work["target_energy_for_feasibility_tj"] = work["energy_target_tj"].where(
        work["energy_target_tj"].gt(0.0),
        implied_target_tj,
    )
    work["market_link_bool"] = _is_market_linked(work.get("market_link", pd.Series(False, index=work.index)))
    work["resource_status_text"] = work.get("resource_status", "").astype(str)
    work["is_dedicated_energy_crop"] = work["feedstock_category"].map(_is_dedicated_energy_crop_category)
    work["is_crop_residue_like"] = [
        _is_crop_residue_like(feedstock, category)
        for feedstock, category in zip(work.get("feedstock", ""), work.get("feedstock_category", ""))
    ]
    work["eligible_land_overdraw"] = (
        work["is_dedicated_energy_crop"]
        & work["land_gap_ha"].gt(0.0)
        & work["unmet_energy_tj"].gt(0.0)
    )
    work["crop_residue_overdraw"] = (
        work["is_crop_residue_like"]
        & ~work["eligible_land_overdraw"]
        & work["resource_status_text"].eq("resource_overdraw")
        & work["unmet_energy_tj"].gt(0.0)
    )
    work["non_crop_overdraw"] = (
        ~work["market_link_bool"]
        & ~work["is_crop_residue_like"]
        & ~work["eligible_land_overdraw"]
        & work["resource_status_text"].eq("resource_overdraw")
        & work["unmet_energy_tj"].gt(0.0)
    )
    work["feedstock_category_group"] = np.select(
        [
            work["market_link_bool"],
            work["is_dedicated_energy_crop"],
            work["is_crop_residue_like"],
        ],
        [
            "market_linked_crop",
            "dedicated_energy_crop",
            "crop_residue",
        ],
        default="non_crop_feedstock",
    )
    work["non_crop_cap_unmet_tj"] = work["unmet_energy_tj"].where(work["non_crop_overdraw"], 0.0)
    work["crop_residue_cap_unmet_tj"] = work["unmet_energy_tj"].where(work["crop_residue_overdraw"], 0.0)
    work["eligible_land_cap_unmet_tj"] = work["unmet_energy_tj"].where(work["eligible_land_overdraw"], 0.0)
    work["market_land_solver_unmet_tj"] = 0.0
    work["resource_overdraw_row"] = work["resource_status_text"].eq("resource_overdraw").astype(int)
    work["eligible_land_overdraw_row"] = work["eligible_land_overdraw"].astype(int)
    work["market_linked_row"] = work["market_link_bool"].astype(int)
    work["uncapped_missing_resource_row"] = work["resource_status_text"].eq("uncapped_missing_resource").astype(int)

    grouped = work.groupby(
        ["scenario", "M49_Country_Code", "country_name", "year", "carrier", "feedstock_category_group"],
        dropna=False,
        as_index=False,
    ).agg(
        target_energy_tj=("target_energy_for_feasibility_tj", "sum"),
        feasible_supplied_tj=("feasible_energy_supplied_tj", "sum"),
        unmet_energy_tj=("unmet_energy_tj", "sum"),
        non_crop_cap_unmet_tj=("non_crop_cap_unmet_tj", "sum"),
        crop_residue_cap_unmet_tj=("crop_residue_cap_unmet_tj", "sum"),
        eligible_land_cap_unmet_tj=("eligible_land_cap_unmet_tj", "sum"),
        market_land_solver_unmet_tj=("market_land_solver_unmet_tj", "sum"),
        resource_overdraw_rows=("resource_overdraw_row", "sum"),
        eligible_land_overdraw_rows=("eligible_land_overdraw_row", "sum"),
        market_linked_rows=("market_linked_row", "sum"),
        uncapped_missing_resource_rows=("uncapped_missing_resource_row", "sum"),
    )
    attribution_cols = [
        "non_crop_cap_unmet_tj",
        "crop_residue_cap_unmet_tj",
        "eligible_land_cap_unmet_tj",
    ]
    grouped["unmet_energy_tj"] = (
        grouped["target_energy_tj"] - grouped["feasible_supplied_tj"]
    ).clip(lower=0.0)
    for col in attribution_cols:
        grouped[col] = grouped[col].clip(lower=0.0)
    known_attribution = grouped[attribution_cols].sum(axis=1)
    over_attributed = known_attribution.gt(grouped["unmet_energy_tj"])
    if over_attributed.any():
        scale = (
            grouped.loc[over_attributed, "unmet_energy_tj"]
            / known_attribution.loc[over_attributed].replace(0.0, np.nan)
        ).fillna(0.0)
        grouped.loc[over_attributed, attribution_cols] = (
            grouped.loc[over_attributed, attribution_cols].mul(scale, axis=0)
        )
        known_attribution = grouped[attribution_cols].sum(axis=1)
    grouped["unattributed_pre_solver_unmet_tj"] = (
        grouped["unmet_energy_tj"] - known_attribution
    ).clip(lower=0.0)
    grouped["market_land_solver_unmet_tj"] = 0.0
    grouped["attributed_unmet_tj"] = (
        grouped[attribution_cols].sum(axis=1)
        + grouped["unattributed_pre_solver_unmet_tj"]
        + grouped["market_land_solver_unmet_tj"]
    )
    grouped["attribution_residual_tj"] = (
        grouped["unmet_energy_tj"] - grouped["attributed_unmet_tj"]
    )
    grouped.loc[
        grouped["attribution_residual_tj"].abs().le(1e-9),
        "attribution_residual_tj",
    ] = 0.0
    grouped["diagnostic_stage"] = "pre_solver"
    for tj_col in [
        "target_energy_tj",
        "feasible_supplied_tj",
        "unmet_energy_tj",
        "non_crop_cap_unmet_tj",
        "crop_residue_cap_unmet_tj",
        "eligible_land_cap_unmet_tj",
        "unattributed_pre_solver_unmet_tj",
        "market_land_solver_unmet_tj",
        "attributed_unmet_tj",
        "attribution_residual_tj",
    ]:
        grouped[tj_col.replace("_tj", "_ej")] = grouped[tj_col] / 1_000_000.0

    reason_cols = [
        ("non_crop_cap_unmet_tj", "non_crop_cap"),
        ("crop_residue_cap_unmet_tj", "crop_residue_cap"),
        ("eligible_land_cap_unmet_tj", "eligible_land_cap"),
        ("unattributed_pre_solver_unmet_tj", "unattributed_pre_solver"),
    ]
    reasons: List[str] = []
    notes: List[str] = []
    for row in grouped.itertuples(index=False):
        values = [(label, float(getattr(row, col, 0.0) or 0.0)) for col, label in reason_cols]
        positive = [(label, val) for label, val in values if val > 0.0]
        if not positive:
            reasons.append("none")
        else:
            positive.sort(key=lambda item: item[1], reverse=True)
            reasons.append(positive[0][0])
        note_parts = ["Pre-solver physical/resource feasibility."]
        if float(getattr(row, "unattributed_pre_solver_unmet_tj", 0.0) or 0.0) > 0.0:
            note_parts.append(
                "Residual unmet energy is retained as unattributed_pre_solver_unmet; "
                "it is not labeled as a solver gap before a solution exists."
            )
        else:
            note_parts.append("market_land_solver_unmet is zero before post-solver audit.")
        if getattr(row, "uncapped_missing_resource_rows", 0) > 0:
            note_parts.append(
                "Some positive feedstock demands have missing resource caps; set strict resource data or choose observed-use/zero policy."
            )
        notes.append(" ".join(note_parts))
    grouped["dominant_gap_reason"] = reasons
    grouped["notes"] = notes
    return grouped[columns].sort_values(
        ["scenario", "year", "M49_Country_Code", "carrier", "feedstock_category_group"],
        na_position="last",
    ).reset_index(drop=True)


def build_bioenergy_postsolve_assessment(
    target_feasibility: Optional[pd.DataFrame],
    linear_result: Optional[Dict[str, Any]],
    land_balance_audit: Optional[pd.DataFrame] = None,
    *,
    energy_tolerance_tj: float = 1e-6,
    quantity_tolerance_t: float = 1e-6,
    land_tolerance_ha: float = 1.0,
    violation_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Build a global post-solver bioenergy feasibility assessment.

    Market slack is retained in tonnes because a reliable conversion to
    feedstock energy requires commodity-specific allocation information.
    """

    def _column_sum(frame: Optional[pd.DataFrame], column: str) -> float:
        if not isinstance(frame, pd.DataFrame) or frame.empty or column not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())

    def _mapping_sum(value: Any) -> float:
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, (list, tuple, set)):
            values = value
        elif value is None:
            return 0.0
        else:
            values = [value]
        total = 0.0
        for item in values:
            try:
                number = float(getattr(item, "X", item) or 0.0)
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(number):
                total += number
        return total

    feasibility = (
        target_feasibility.copy()
        if isinstance(target_feasibility, pd.DataFrame)
        else pd.DataFrame()
    )
    result = linear_result if isinstance(linear_result, dict) else {}
    raw_status = result.get("status")
    inferred_solution = bool(result.get("Qs") or result.get("Qd"))
    inferred_sol_count = result.get("sol_count")
    if inferred_sol_count is None and inferred_solution:
        inferred_sol_count = 1
    solver = normalize_solver_status(
        raw_status,
        sol_count=inferred_sol_count,
        has_solution=result.get("has_solution"),
        objective=result.get("objective"),
        runtime_seconds=result.get("runtime_seconds", result.get("runtime")),
        max_violation=result.get("max_violation"),
    )

    scenarios: List[str] = []
    if not feasibility.empty and "scenario" in feasibility.columns:
        scenarios = sorted(
            {
                str(value).strip()
                for value in feasibility["scenario"].dropna().tolist()
                if str(value).strip()
            }
        )
    years: List[int] = []
    if not feasibility.empty and "year" in feasibility.columns:
        years = sorted(
            {
                int(value)
                for value in pd.to_numeric(feasibility["year"], errors="coerce").dropna().tolist()
            }
        )

    target_tj = _column_sum(feasibility, "target_energy_tj")
    feasible_tj = _column_sum(feasibility, "feasible_supplied_tj")
    pre_unmet_tj = _column_sum(feasibility, "unmet_energy_tj")
    non_crop_unmet_tj = _column_sum(feasibility, "non_crop_cap_unmet_tj")
    crop_residue_unmet_tj = _column_sum(feasibility, "crop_residue_cap_unmet_tj")
    eligible_land_unmet_tj = _column_sum(feasibility, "eligible_land_cap_unmet_tj")
    unattributed_unmet_tj = _column_sum(
        feasibility,
        "unattributed_pre_solver_unmet_tj",
    )
    attribution_residual_tj = _column_sum(feasibility, "attribution_residual_tj")
    shortage_t = _mapping_sum(result.get("shortage"))
    excess_t = _mapping_sum(result.get("excess"))
    land_slack_ha = _mapping_sum(result.get("land_slack"))
    eligible_land_gap_ha = _column_sum(land_balance_audit, "bioenergy_land_gap_ha")
    max_violation = result.get("max_violation")
    try:
        max_violation_f = float(max_violation) if max_violation is not None else 0.0
    except (TypeError, ValueError, OverflowError):
        max_violation_f = np.nan

    reasons: List[str] = []
    usable_solver_statuses = {
        "optimal",
        "suboptimal",
        "time_limit",
        "solution_limit",
        "interrupted",
        "user_obj_limit",
        "work_limit",
        "memory_limit",
    }
    if target_tj <= float(energy_tolerance_tj):
        assessment_status = "not_applicable"
    elif (
        not bool(solver["has_solution"])
        or str(solver["status_name"]) not in usable_solver_statuses
    ):
        assessment_status = "failed"
        reasons.append(
            "solver_has_no_solution"
            if not bool(solver["has_solution"])
            else "solver_status_not_usable"
        )
    else:
        if pre_unmet_tj > float(energy_tolerance_tj):
            reasons.append("pre_solver_unmet_energy")
        if abs(attribution_residual_tj) > float(energy_tolerance_tj):
            reasons.append("pre_solver_attribution_not_conserved")
        if shortage_t > float(quantity_tolerance_t):
            reasons.append("market_shortage_slack")
        if excess_t > float(quantity_tolerance_t):
            reasons.append("market_excess_slack")
        if land_slack_ha > float(land_tolerance_ha):
            reasons.append("land_constraint_slack")
        if eligible_land_gap_ha > float(land_tolerance_ha):
            reasons.append("eligible_land_gap")
        if np.isfinite(max_violation_f) and max_violation_f > float(violation_tolerance):
            reasons.append("post_solve_constraint_violation")
        assessment_status = "stress_only" if reasons else "valid"

    attributed_tj = (
        non_crop_unmet_tj
        + crop_residue_unmet_tj
        + eligible_land_unmet_tj
        + unattributed_unmet_tj
    )
    return pd.DataFrame(
        [
            {
                "diagnostic_stage": "post_solver",
                "scenario": ";".join(scenarios),
                "year_min": min(years) if years else np.nan,
                "year_max": max(years) if years else np.nan,
                "solver_status_code": solver.get("status_code"),
                "solver_status_name": solver["status_name"],
                "solver_has_solution": bool(solver["has_solution"]),
                "solver_optimal": bool(solver["optimal"]),
                "target_energy_tj": target_tj,
                "target_energy_ej": target_tj / 1_000_000.0,
                "feasible_supplied_tj": feasible_tj,
                "feasible_supplied_ej": feasible_tj / 1_000_000.0,
                "pre_solver_unmet_tj": pre_unmet_tj,
                "pre_solver_unmet_ej": pre_unmet_tj / 1_000_000.0,
                "pre_solver_attributed_unmet_tj": attributed_tj,
                "pre_solver_attribution_residual_tj": pre_unmet_tj - attributed_tj,
                "market_shortage_t": shortage_t,
                "market_excess_t": excess_t,
                "land_constraint_slack_ha": land_slack_ha,
                "eligible_land_gap_ha": eligible_land_gap_ha,
                "max_constraint_violation_model_units": max_violation_f,
                "assessment_status": assessment_status,
                "target_feasible": assessment_status == "valid",
                "assessment_reasons": ";".join(reasons) if reasons else "none",
                "notes": (
                    "Post-solver assessment. Market shortage/excess remains in tonnes; "
                    "it is not converted to bioenergy because commodity-specific energy "
                    "allocation is not available in the aggregate solver slack."
                ),
            }
        ]
    )


def _emissions_handoff(detail: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "feedstock",
        "feedstock_category",
        "activity_tdm",
        "activity_t",
        "energy_supplied_tj",
        "ghg_direct_kgco2e_per_tdm",
        "ghg_soil_kgco2e_per_tdm",
        "ghg_avoided_kgco2e_per_tdm",
        "fossil_displacement_kgco2e_per_tj",
        "beccs_capture_kgco2_per_tdm",
        "direct_emissions_ktco2e",
        "soil_carbon_ktco2e",
        "avoided_emissions_ktco2e",
        "fossil_displacement_credit_ktco2e",
        "beccs_capture_credit_ktco2e",
        "net_biomass_emissions_ktco2e",
        "handoff_status",
        "source",
        "notes",
    ]
    if detail.empty:
        return pd.DataFrame(columns=columns)
    work = detail.copy()
    activity_dry = pd.to_numeric(work["feasible_feedstock_demand_tdm"], errors="coerce")
    activity_dry = activity_dry.where(
        activity_dry.notna(),
        pd.to_numeric(work["feedstock_demand_tdm"], errors="coerce"),
    )
    activity_wet = pd.to_numeric(work["feasible_feedstock_demand_t"], errors="coerce")
    activity_wet = activity_wet.where(
        activity_wet.notna(),
        pd.to_numeric(work["feedstock_demand_t"], errors="coerce"),
    )
    energy = pd.to_numeric(work["feasible_energy_supplied_tj"], errors="coerce")
    energy = energy.where(energy.notna(), pd.to_numeric(work["energy_supplied_tj"], errors="coerce"))
    direct_ef = pd.to_numeric(work["ghg_direct_kgco2e_per_tdm"], errors="coerce")
    soil_ef = pd.to_numeric(work["ghg_soil_kgco2e_per_tdm"], errors="coerce")
    avoided_ef = pd.to_numeric(work["ghg_avoided_kgco2e_per_tdm"], errors="coerce")
    displacement_ef = pd.to_numeric(work["fossil_displacement_kgco2e_per_tj"], errors="coerce")
    capture_ef = pd.to_numeric(work["beccs_capture_kgco2_per_tdm"], errors="coerce")

    direct = activity_dry * direct_ef / 1_000_000.0
    soil = activity_dry * soil_ef / 1_000_000.0
    avoided = activity_dry * avoided_ef / 1_000_000.0
    displacement = energy * displacement_ef / 1_000_000.0
    capture = activity_dry * capture_ef / 1_000_000.0
    has_any_factor = pd.concat(
        [direct_ef, soil_ef, avoided_ef, displacement_ef, capture_ef],
        axis=1,
    ).notna().any(axis=1)
    net = (
        direct.fillna(0.0)
        + soil.fillna(0.0)
        - avoided.fillna(0.0)
        - displacement.fillna(0.0)
        - capture.fillna(0.0)
    ).where(has_any_factor)

    out = pd.DataFrame({
        "scenario": work["scenario"],
        "M49_Country_Code": work["M49_Country_Code"],
        "country_name": work["country_name"],
        "year": pd.to_numeric(work["year"], errors="coerce").astype("Int64"),
        "feedstock": work["feedstock"],
        "feedstock_category": work["feedstock_category"],
        "activity_tdm": activity_dry,
        "activity_t": activity_wet,
        "energy_supplied_tj": energy,
        "ghg_direct_kgco2e_per_tdm": direct_ef,
        "ghg_soil_kgco2e_per_tdm": soil_ef,
        "ghg_avoided_kgco2e_per_tdm": avoided_ef,
        "fossil_displacement_kgco2e_per_tj": displacement_ef,
        "beccs_capture_kgco2_per_tdm": capture_ef,
        "direct_emissions_ktco2e": direct,
        "soil_carbon_ktco2e": soil,
        "avoided_emissions_ktco2e": avoided,
        "fossil_displacement_credit_ktco2e": displacement,
        "beccs_capture_credit_ktco2e": capture,
        "net_biomass_emissions_ktco2e": net,
        "handoff_status": np.where(
            has_any_factor,
            "ready_for_emissions_integration",
            "missing_emission_factors",
        ),
        "source": work["source"],
        "notes": work["notes"],
    })
    out = out[activity_dry.fillna(0.0).gt(0.0)].copy()
    return out[columns].sort_values(
        ["scenario", "M49_Country_Code", "year", "feedstock"],
        na_position="last",
    ).reset_index(drop=True)


def _land_handoff(detail: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "feedstock",
        "feedstock_category",
        "activity_tdm",
        "yield_tdm_per_ha",
        "energy_crop_area_target_ha",
        "eligible_land_area_ha",
        "land_requirement_ha",
        "cropland_reallocation_area_ha",
        "additional_land_expansion_ha",
        "feasible_additional_land_expansion_ha",
        "feasible_land_requirement_ha",
        "land_gap_ha",
        "handoff_status",
        "source",
        "notes",
    ]
    if detail.empty:
        return pd.DataFrame(columns=columns)
    work = detail.copy()
    dedicated = work["feedstock_category"].map(_is_dedicated_energy_crop_category)
    if not dedicated.any():
        return pd.DataFrame(columns=columns)
    activity = pd.to_numeric(work["feasible_feedstock_demand_tdm"], errors="coerce")
    activity = activity.where(activity.notna(), pd.to_numeric(work["feedstock_demand_tdm"], errors="coerce"))
    land_requirement = pd.to_numeric(work["land_requirement_ha"], errors="coerce")
    feasible_land_requirement = activity / pd.to_numeric(work["yield_tdm_per_ha"], errors="coerce").replace(0.0, np.nan)
    yield_tdm = pd.to_numeric(work["yield_tdm_per_ha"], errors="coerce")
    eligible_land = pd.to_numeric(work["eligible_land_area_ha"], errors="coerce")
    energy_crop_area_target = pd.to_numeric(work["energy_crop_area_target_ha"], errors="coerce")
    cropland_reallocation = pd.to_numeric(work["cropland_reallocation_area_ha"], errors="coerce")
    additional_expansion = pd.to_numeric(work["additional_land_expansion_ha"], errors="coerce")
    feasible_additional_expansion = pd.to_numeric(work["feasible_additional_land_expansion_ha"], errors="coerce")
    status = np.where(yield_tdm.notna() & yield_tdm.gt(0.0), "ready_for_land_integration", "missing_energy_crop_yield")
    status = np.where(
        yield_tdm.notna() & yield_tdm.gt(0.0) & eligible_land.notna() & pd.to_numeric(work["land_gap_ha"], errors="coerce").fillna(0.0).gt(0.0),
        "eligible_land_overdraw",
        status,
    )
    out = pd.DataFrame({
        "scenario": work["scenario"],
        "M49_Country_Code": work["M49_Country_Code"],
        "country_name": work["country_name"],
        "year": pd.to_numeric(work["year"], errors="coerce").astype("Int64"),
        "feedstock": work["feedstock"],
        "feedstock_category": work["feedstock_category"],
        "activity_tdm": activity,
        "yield_tdm_per_ha": yield_tdm,
        "energy_crop_area_target_ha": energy_crop_area_target,
        "eligible_land_area_ha": eligible_land,
        "land_requirement_ha": land_requirement,
        "cropland_reallocation_area_ha": cropland_reallocation,
        "additional_land_expansion_ha": additional_expansion,
        "feasible_additional_land_expansion_ha": feasible_additional_expansion,
        "feasible_land_requirement_ha": feasible_land_requirement,
        "land_gap_ha": pd.to_numeric(work["land_gap_ha"], errors="coerce"),
        "handoff_status": status,
        "source": work["source"],
        "notes": work["notes"],
    })
    out = out[dedicated & activity.fillna(0.0).gt(0.0)].copy()
    return out[columns].sort_values(
        ["scenario", "M49_Country_Code", "year", "feedstock"],
        na_position="last",
    ).reset_index(drop=True)


def _residue_management_handoff(detail: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "feedstock",
        "feedstock_category",
        "parent_commodity",
        "model_commodity",
        "total_residue_tdm",
        "bioenergy_residue_removed_tdm",
        "competing_use_tdm",
        "sustainable_supply_tdm",
        "resource_gap_tdm",
        "residue_removed_fraction_total",
        "residue_carbon_pct",
        "residue_nitrogen_pct",
        "bioenergy_residue_c_removed_t",
        "bioenergy_residue_n_removed_t",
        "crop_residue_n2o_multiplier",
        "burning_residue_multiplier",
        "feed_competition_status",
        "source",
        "notes",
    ]
    if detail.empty:
        return pd.DataFrame(columns=columns)
    work = detail.copy()
    category = work["feedstock_category"].fillna("").astype(str).str.lower()
    residue_mask = category.str.contains("residue", na=False)
    if not residue_mask.any():
        return pd.DataFrame(columns=columns)
    residue = work[residue_mask].copy()
    total = pd.to_numeric(residue["resource_available_tdm"], errors="coerce")
    removed = pd.to_numeric(residue["feasible_feedstock_demand_tdm"], errors="coerce")
    removed = removed.where(removed.notna(), pd.to_numeric(residue["feedstock_demand_tdm"], errors="coerce"))
    competing = pd.to_numeric(residue["competing_use_tdm"], errors="coerce").fillna(0.0)
    sustainable = pd.to_numeric(residue["sustainable_supply_tdm"], errors="coerce")
    gap = pd.to_numeric(residue["resource_gap_tdm"], errors="coerce").fillna(0.0)
    carbon_pct = pd.to_numeric(residue.get("residue_carbon_pct"), errors="coerce")
    nitrogen_pct = pd.to_numeric(residue.get("residue_nitrogen_pct"), errors="coerce")
    carbon_removed_t = removed * carbon_pct / 100.0
    nitrogen_removed_t = removed * nitrogen_pct / 100.0
    removed_fraction = (removed / total.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)
    multiplier = (1.0 - removed_fraction).clip(lower=0.0, upper=1.0)
    parent = residue["parent_commodity"].fillna("").astype(str).str.strip()
    model = residue["model_commodity"].fillna("").astype(str).str.strip()
    parent = parent.where(parent.ne(""), model)
    feed_status = np.where(
        gap.gt(0.0),
        "bioenergy_overdraw_after_feed_soil_screen",
        np.where(competing.gt(0.0), "competing_use_protected_by_resource_cap", "no_competing_use_parameter"),
    )
    out = pd.DataFrame({
        "scenario": residue["scenario"],
        "M49_Country_Code": residue["M49_Country_Code"],
        "country_name": residue["country_name"],
        "year": pd.to_numeric(residue["year"], errors="coerce").astype("Int64"),
        "feedstock": residue["feedstock"],
        "feedstock_category": residue["feedstock_category"],
        "parent_commodity": parent,
        "model_commodity": model,
        "total_residue_tdm": total,
        "bioenergy_residue_removed_tdm": removed,
        "competing_use_tdm": competing,
        "sustainable_supply_tdm": sustainable,
        "resource_gap_tdm": gap,
        "residue_removed_fraction_total": removed_fraction,
        "residue_carbon_pct": carbon_pct,
        "residue_nitrogen_pct": nitrogen_pct,
        "bioenergy_residue_c_removed_t": carbon_removed_t,
        "bioenergy_residue_n_removed_t": nitrogen_removed_t,
        "crop_residue_n2o_multiplier": multiplier,
        "burning_residue_multiplier": multiplier,
        "feed_competition_status": feed_status,
        "source": residue["source"],
        "notes": residue["notes"],
    })
    out = out[removed.fillna(0.0).gt(0.0)].copy()
    return out[columns].sort_values(
        ["scenario", "M49_Country_Code", "year", "feedstock"],
        na_position="last",
    ).reset_index(drop=True)


def _coproduct_feed_handoff(detail: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "feedstock",
        "model_commodity",
        "coproduct_feed_commodity",
        "coproduct_feed_credit_tdm",
        "handoff_status",
        "source",
        "notes",
    ]
    if detail.empty or "coproduct_feed_credit_tdm" not in detail.columns:
        return pd.DataFrame(columns=columns)
    work = detail.copy()
    credit = pd.to_numeric(work["coproduct_feed_credit_tdm"], errors="coerce").fillna(0.0)
    work = work[credit.gt(0.0)].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({
        "scenario": work["scenario"],
        "M49_Country_Code": work["M49_Country_Code"],
        "country_name": work["country_name"],
        "year": pd.to_numeric(work["year"], errors="coerce").astype("Int64"),
        "feedstock": work["feedstock"],
        "model_commodity": work["model_commodity"],
        "coproduct_feed_commodity": work["coproduct_feed_commodity"].fillna("").astype(str),
        "coproduct_feed_credit_tdm": pd.to_numeric(work["coproduct_feed_credit_tdm"], errors="coerce"),
        "handoff_status": "handoff_only_not_applied_to_feed_solver",
        "source": work["source"],
        "notes": work["notes"],
    })
    return out[columns].sort_values(
        ["scenario", "M49_Country_Code", "year", "feedstock"],
        na_position="last",
    ).reset_index(drop=True)


def _residue_feed_competition_handoff(detail: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "feedstock",
        "parent_commodity",
        "residue_feed_competition_tdm",
        "competing_use_tdm",
        "resource_gap_tdm",
        "handoff_status",
        "source",
        "notes",
    ]
    if detail.empty or "residue_feed_competition_tdm" not in detail.columns:
        return pd.DataFrame(columns=columns)
    work = detail.copy()
    demand = pd.to_numeric(work["residue_feed_competition_tdm"], errors="coerce").fillna(0.0)
    category = work["feedstock_category"].fillna("").astype(str).str.lower()
    work = work[demand.gt(0.0) & category.str.contains("residue", na=False)].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({
        "scenario": work["scenario"],
        "M49_Country_Code": work["M49_Country_Code"],
        "country_name": work["country_name"],
        "year": pd.to_numeric(work["year"], errors="coerce").astype("Int64"),
        "feedstock": work["feedstock"],
        "parent_commodity": work["parent_commodity"],
        "residue_feed_competition_tdm": pd.to_numeric(work["residue_feed_competition_tdm"], errors="coerce"),
        "competing_use_tdm": pd.to_numeric(work["competing_use_tdm"], errors="coerce"),
        "resource_gap_tdm": pd.to_numeric(work["resource_gap_tdm"], errors="coerce"),
        "handoff_status": "handoff_only_not_applied_to_feed_solver",
        "source": work["source"],
        "notes": work["notes"],
    })
    return out[columns].sort_values(
        ["scenario", "M49_Country_Code", "year", "feedstock"],
        na_position="last",
    ).reset_index(drop=True)


def _energy_crop_land_requirement_map(land_handoff: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    if not isinstance(land_handoff, pd.DataFrame) or land_handoff.empty:
        return {}
    work = land_handoff.copy()
    work["year"] = pd.to_numeric(work.get("year"), errors="coerce")
    # B02: reallocated existing cropland occupies the same finite budget as
    # expansion. The handoff already excludes conventional agricultural crops.
    value_col = "feasible_land_requirement_ha"
    if value_col not in work.columns:
        if {"activity_tdm", "yield_tdm_per_ha"}.issubset(work.columns):
            work[value_col] = pd.to_numeric(work["activity_tdm"], errors="coerce") / pd.to_numeric(
                work["yield_tdm_per_ha"], errors="coerce"
            ).replace(0.0, np.nan)
        else:
            raise ValueError("Dedicated energy crops require full resource-feasible area or biomass/yield; expansion alone is insufficient")
    work[value_col] = pd.to_numeric(work.get(value_col), errors="coerce")
    if work[value_col].isna().any() or (work[value_col] < 0).any():
        raise ValueError("Missing or negative full dedicated energy-crop area")
    work = work.dropna(subset=["year", value_col])
    work = work[work[value_col].gt(0.0)]
    if work.empty:
        return {}
    grouped = work.groupby(["M49_Country_Code", "year"], as_index=False)[value_col].sum()
    return {
        (str(r.M49_Country_Code), int(r.year)): float(getattr(r, value_col))
        for r in grouped.itertuples(index=False)
    }


def _crop_residue_multiplier_map(residue_handoff: pd.DataFrame) -> Dict[Tuple[str, str, str, int], float]:
    if not isinstance(residue_handoff, pd.DataFrame) or residue_handoff.empty:
        return {}
    rows: Dict[Tuple[str, str, str, int], List[float]] = {}
    for row in residue_handoff.itertuples(index=False):
        try:
            m49 = str(row.M49_Country_Code)
            item = str(row.parent_commodity or row.model_commodity).strip()
            year = int(row.year)
            residue_mult = float(row.crop_residue_n2o_multiplier)
            burn_mult = float(row.burning_residue_multiplier)
        except Exception:
            continue
        if not item or not np.isfinite(residue_mult):
            continue
        for process, mult in [
            ("Crop residues", residue_mult),
            ("Burning crop residues", burn_mult),
        ]:
            if not np.isfinite(mult):
                continue
            rows.setdefault((m49, item, process, year), []).append(float(np.clip(mult, 0.0, 1.0)))
    out: Dict[Tuple[str, str, str, int], float] = {}
    for key, values in rows.items():
        # Multiple residue streams for one crop should compound by retaining the
        # most conservative multiplier implied by total removal.
        out[key] = min(values) if values else 1.0
    return out


def _resource_diagnostics(resource_balance: pd.DataFrame, emissions_handoff: pd.DataFrame, land_handoff: pd.DataFrame) -> List[str]:
    diagnostics: List[str] = []
    if isinstance(resource_balance, pd.DataFrame) and not resource_balance.empty:
        over = resource_balance[resource_balance["resource_status"].astype(str).eq("resource_overdraw")]
        if not over.empty:
            diagnostics.append(
                "Bioenergy resource overdraw: rows=%d unmet_energy_tj=%.6g resource_gap_tdm=%.6g"
                % (
                    len(over),
                    pd.to_numeric(over["unmet_energy_tj"], errors="coerce").fillna(0.0).sum(),
                    pd.to_numeric(over["resource_gap_tdm"], errors="coerce").fillna(0.0).sum(),
                )
            )
        missing = resource_balance[
            resource_balance["resource_status"].astype(str).eq("uncapped_missing_resource")
            & pd.to_numeric(resource_balance["feedstock_demand_tdm"], errors="coerce").fillna(0.0).gt(0.0)
        ]
        if not missing.empty:
            feedstocks = ", ".join(sorted(missing["feedstock"].astype(str).unique())[:10])
            diagnostics.append(
                f"Non-market bioenergy feedstocks lack resource constraints: rows={len(missing)} feedstocks={feedstocks}"
            )
    if isinstance(emissions_handoff, pd.DataFrame) and not emissions_handoff.empty:
        missing_ef = emissions_handoff[
            emissions_handoff["handoff_status"].astype(str).eq("missing_emission_factors")
        ]
        if not missing_ef.empty:
            feedstocks = ", ".join(sorted(missing_ef["feedstock"].astype(str).unique())[:10])
            diagnostics.append(
                f"Bioenergy emissions handoff lacks factors: rows={len(missing_ef)} feedstocks={feedstocks}"
            )
    if isinstance(land_handoff, pd.DataFrame) and not land_handoff.empty:
        land_gap = land_handoff[
            land_handoff["handoff_status"].astype(str).eq("eligible_land_overdraw")
        ]
        if not land_gap.empty:
            diagnostics.append(
                "Dedicated energy-crop land overdraw: rows=%d land_gap_ha=%.6g"
                % (
                    len(land_gap),
                    pd.to_numeric(land_gap["land_gap_ha"], errors="coerce").fillna(0.0).sum(),
                )
            )
    return diagnostics


def build_bioenergy_bundle(
    *,
    universe: Universe,
    active_years: Iterable[int],
    hist_end_year: int,
    scenario: str,
    historical_feedstock_path: Optional[str],
    scenario_path: Optional[str],
    parameter_path: Optional[str],
    resource_path: Optional[str] = None,
    require_historical_bridge: bool = False,
    require_scenario_rows: bool = False,
) -> BioenergyBundle:
    """Build historical reconciliation and future scenario demand maps."""
    scenario_name = str(scenario or "").strip()
    bundle = BioenergyBundle(scenario=scenario_name)
    parameters = load_feedstock_parameters(parameter_path)

    hist_rows = _normalize_demand_rows(
        historical_feedstock_path,
        universe=universe,
        default_scenario="historical",
    )
    hist_rows = _merge_parameters(hist_rows, parameters)
    hist_detail = _calculate_physical_quantities(hist_rows)
    hist_base = _latest_historical_rows(hist_detail, hist_end_year)
    bundle.historical_detail = hist_detail
    bundle.historical_crop_demand_by_country_comm_year = _crop_demand_map(hist_detail, universe)
    hist_base_map_year = _crop_demand_map(hist_base, universe)
    bundle.historical_crop_base_by_country_comm = {
        (country, commodity): value
        for (country, commodity, _year), value in hist_base_map_year.items()
    }

    scenario_rows = _normalize_demand_rows(
        scenario_path,
        universe=universe,
        default_scenario=scenario_name,
    )
    scenario_rows = _select_scenario_rows(scenario_rows, scenario_name)
    if require_scenario_rows and scenario_rows.empty:
        path_text = str(scenario_path or "").strip() or "<not configured>"
        raise ValueError(
            "Bioenergy is enabled but no usable rows were found for scenario "
            f"'{scenario_name}' in {path_text}. Refusing to continue as a zero-bioenergy run."
        )
    scenario_rows = _merge_parameters(scenario_rows, parameters)
    scenario_known = _calculate_physical_quantities(scenario_rows)
    future_years = [int(y) for y in active_years if int(y) > int(hist_end_year)]
    if require_scenario_rows and not future_years:
        raise ValueError(
            "Bioenergy is enabled but active_years contains no future year after "
            f"hist_end_year={int(hist_end_year)}."
        )
    scenario_detail = _interpolate_detail(scenario_known, future_years)
    scenario_detail, world_diagnostics = _downscale_world_rows(
        scenario_detail,
        hist_base,
        universe=universe,
    )
    bundle.diagnostics.extend(world_diagnostics)
    if require_scenario_rows and scenario_detail.empty:
        path_text = str(scenario_path or "").strip() or "<not configured>"
        raise ValueError(
            "Bioenergy scenario rows were selected but produced no usable future "
            f"detail for scenario '{scenario_name}' from {path_text}."
        )
    scenario_crop_demand = _crop_demand_map(scenario_detail, universe)
    if require_historical_bridge and scenario_crop_demand and not bundle.historical_crop_base_by_country_comm:
        path_text = str(historical_feedstock_path or "").strip() or "<not configured>"
        raise ValueError(
            "Bioenergy scenario contains market-linked crop feedstock demand, but no usable "
            "historical feedstock-level crop bioenergy bridge was loaded. This would leave "
            "historical FBS residual demand unreconciled and can double-count biofuel crop "
            f"use. Provide bioenergy_historical_feedstock.csv via {path_text}, run "
            "S0_51_prepare_bioenergy_feedstock_bridge.py, or explicitly set "
            "CFG['bioenergy_require_historical_feedstock_bridge']=False for diagnostic-only runs."
        )
    resource_rows = _normalize_resource_rows(
        resource_path,
        universe=universe,
        default_scenario=scenario_name,
    )
    resource_rows = _select_scenario_rows(resource_rows, scenario_name)
    resource_detail = _interpolate_resource_rows(resource_rows, future_years)
    scenario_detail = _apply_resource_rows(scenario_detail, resource_detail)
    bundle.scenario_detail = scenario_detail
    bundle.crop_demand_by_country_comm_year = _crop_demand_map(scenario_detail, universe)
    bundle.energy_balance = _energy_balance(scenario_detail)
    bundle.resource_balance = _resource_balance(scenario_detail)
    bundle.target_feasibility = _target_feasibility(scenario_detail)
    bundle.emissions_handoff = _emissions_handoff(scenario_detail)
    bundle.land_handoff = _land_handoff(scenario_detail)
    bundle.residue_management_handoff = _residue_management_handoff(scenario_detail)
    bundle.coproduct_feed_handoff = _coproduct_feed_handoff(scenario_detail)
    bundle.residue_feed_competition_handoff = _residue_feed_competition_handoff(scenario_detail)
    bundle.energy_crop_land_requirement_by_country_year = _energy_crop_land_requirement_map(
        bundle.land_handoff
    )
    bundle.crop_residue_management_multiplier = _crop_residue_multiplier_map(
        bundle.residue_management_handoff
    )
    bundle.diagnostics.extend(
        _resource_diagnostics(
            bundle.resource_balance,
            bundle.emissions_handoff,
            bundle.land_handoff,
        )
    )

    if scenario_name and scenario_rows.empty:
        bundle.diagnostics.append(f"No rows found for bioenergy scenario '{scenario_name}'.")
    missing_market = scenario_detail[
        scenario_detail["market_link"].astype(bool)
        & ~scenario_detail["model_commodity"].isin(set(universe.commodities))
    ]
    if not missing_market.empty:
        missing = sorted(missing_market["model_commodity"].dropna().astype(str).unique().tolist())
        bundle.diagnostics.append(
            "Market-linked feedstocks reference commodities outside the model universe: "
            + ", ".join(missing[:20])
        )
    return bundle


def build_baseline_reconciliation(
    *,
    nodes: Iterable[Any],
    fbs_components: Optional[pd.DataFrame],
    historical_crop_base_by_country_comm: Dict[Tuple[str, str], float],
    hist_end_year: int,
) -> pd.DataFrame:
    """Create a baseline diagnostic proving that bioenergy is removed from residual."""
    node_rows = []
    for node in nodes:
        try:
            if int(getattr(node, "year")) != int(hist_end_year):
                continue
            node_rows.append(
                {
                    "country": str(getattr(node, "country")),
                    "commodity": str(getattr(node, "commodity")),
                    "D0_t": max(0.0, float(getattr(node, "D0", 0.0) or 0.0)),
                }
            )
        except Exception:
            continue
    base = pd.DataFrame(node_rows)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "commodity",
                "D0_t",
                "food_t",
                "feed_t",
                "bioenergy_base_t",
                "residual_before_bioenergy_t",
                "residual_after_bioenergy_t",
                "bioenergy_overdraw_t",
            ]
        )
    base = base.groupby(["country", "commodity"], as_index=False)["D0_t"].sum()
    comp = pd.DataFrame(columns=["country", "commodity", "food_t", "feed_t"])
    if isinstance(fbs_components, pd.DataFrame) and not fbs_components.empty:
        work = fbs_components.copy()
        work["year"] = pd.to_numeric(work.get("year"), errors="coerce")
        work = work[work["year"].eq(int(hist_end_year))]
        for col in ["food_t", "feed_t"]:
            work[col] = pd.to_numeric(work.get(col), errors="coerce").fillna(0.0)
        if not work.empty:
            comp = work.groupby(["country", "commodity"], as_index=False)[["food_t", "feed_t"]].sum()
    out = base.merge(comp, on=["country", "commodity"], how="left")
    out[["food_t", "feed_t"]] = out[["food_t", "feed_t"]].fillna(0.0)
    out["bioenergy_base_t"] = [
        float(historical_crop_base_by_country_comm.get((str(c), str(j)), 0.0) or 0.0)
        for c, j in zip(out["country"], out["commodity"])
    ]
    out["residual_before_bioenergy_t"] = (
        out["D0_t"] - out["food_t"] - out["feed_t"]
    ).clip(lower=0.0)
    out["residual_after_bioenergy_t"] = (
        out["residual_before_bioenergy_t"] - out["bioenergy_base_t"]
    ).clip(lower=0.0)
    out["bioenergy_overdraw_t"] = (
        out["bioenergy_base_t"] - out["residual_before_bioenergy_t"]
    ).clip(lower=0.0)
    return out


def _energy_physical_consistency_diagnostics(detail: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario", "M49_Country_Code", "country_name", "year", "carrier",
        "feedstock_category", "feedstock", "energy_basis", "energy_target_tj",
        "feedstock_demand_tdm", "lhv_gj_per_tdm", "conversion_efficiency",
        "implied_tdm_from_energy", "energy_supplied_from_physical_tj",
        "physical_minus_energy_implied_tdm", "physical_to_energy_implied_ratio",
        "energy_gap_tj", "relative_energy_gap", "consistency_status", "notes",
    ]
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame(columns=columns)
    work = detail.copy()
    for col in ["energy_target_tj", "feedstock_demand_tdm", "lhv_gj_per_tdm", "conversion_efficiency"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    energy = work["energy_target_tj"]
    physical = work["feedstock_demand_tdm"]
    lhv = work["lhv_gj_per_tdm"]
    eff = work["conversion_efficiency"].clip(lower=0.0)
    usable_gj_per_tdm = lhv * eff
    implied = energy * 1000.0 / usable_gj_per_tdm.replace(0.0, np.nan)
    supplied = physical * usable_gj_per_tdm / 1000.0
    gap = energy - supplied
    ratio = physical / implied.replace(0.0, np.nan)
    status = np.full(len(work), "consistent_with_energy_conversion", dtype=object)
    status[pd.isna(energy) | energy.fillna(0.0).le(0.0)] = "no_positive_energy_target"
    status[pd.isna(usable_gj_per_tdm) | usable_gj_per_tdm.fillna(0.0).le(0.0)] = "missing_or_zero_lhv_efficiency"
    status[(ratio < 0.9) & pd.notna(ratio)] = "physical_demand_below_energy_equivalent"
    status[(ratio > 1.1) & pd.notna(ratio)] = "physical_demand_above_energy_equivalent"
    out = pd.DataFrame({
        "scenario": work.get("scenario", ""),
        "M49_Country_Code": work.get("M49_Country_Code", ""),
        "country_name": work.get("country_name", ""),
        "year": pd.to_numeric(work.get("year"), errors="coerce").astype("Int64"),
        "carrier": work.get("carrier", ""),
        "feedstock_category": work.get("feedstock_category", ""),
        "feedstock": work.get("feedstock", ""),
        "energy_basis": work.get("energy_basis", ""),
        "energy_target_tj": energy,
        "feedstock_demand_tdm": physical,
        "lhv_gj_per_tdm": lhv,
        "conversion_efficiency": eff,
        "implied_tdm_from_energy": implied,
        "energy_supplied_from_physical_tj": supplied,
        "physical_minus_energy_implied_tdm": physical - implied,
        "physical_to_energy_implied_ratio": ratio,
        "energy_gap_tj": gap,
        "relative_energy_gap": gap / energy.abs().replace(0.0, np.nan),
        "consistency_status": status,
        "notes": (
            "Compares scenario energy target with physical dry-matter demand using "
            "feedstock LHV and conversion efficiency. Large gaps are expected when "
            "SCI Agricultural Demand physical quantities override energy-derived demand."
        ),
    })
    return out[columns].sort_values(
        ["scenario", "M49_Country_Code", "year", "carrier", "feedstock_category", "feedstock"],
        na_position="last",
    ).reset_index(drop=True)


def write_bioenergy_bundle(bundle: BioenergyBundle, output_dir: str) -> None:
    """Write stable diagnostic tables for a scenario run."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle.historical_detail.to_csv(
        out / "bioenergy_historical_feedstock_use.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.scenario_detail.to_csv(
        out / "bioenergy_feedstock_use.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.energy_balance.to_csv(
        out / "bioenergy_energy_balance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.resource_balance.to_csv(
        out / "bioenergy_resource_balance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.target_feasibility.to_csv(
        out / "bioenergy_target_feasibility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.emissions_handoff.to_csv(
        out / "bioenergy_emissions_handoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.land_handoff.to_csv(
        out / "bioenergy_land_handoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.residue_management_handoff.to_csv(
        out / "bioenergy_residue_management_handoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.coproduct_feed_handoff.to_csv(
        out / "bioenergy_coproduct_feed_handoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.residue_feed_competition_handoff.to_csv(
        out / "bioenergy_residue_feed_competition_handoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _energy_physical_consistency_diagnostics(bundle.scenario_detail).to_csv(
        out / "bioenergy_energy_physical_consistency_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    crop_rows = [
        {
            "M49_Country_Code": country,
            "year": year,
            "commodity": commodity,
            "bioenergy_demand_t": value,
        }
        for (country, commodity, year), value in sorted(bundle.crop_demand_by_country_comm_year.items())
    ]
    pd.DataFrame(
        crop_rows,
        columns=["M49_Country_Code", "year", "commodity", "bioenergy_demand_t"],
    ).to_csv(
        out / "bioenergy_crop_demand.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"diagnostic": bundle.diagnostics}).to_csv(
        out / "bioenergy_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
