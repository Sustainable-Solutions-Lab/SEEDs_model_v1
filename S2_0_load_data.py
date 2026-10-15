# -*- coding: utf-8 -*-
"""
S2.0_load_data: Read and construct data aligned with dict_v3, FAOSTAT files, and scenario workflows.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import logging
import builtins
from collections import defaultdict
from functools import lru_cache
import re
import numpy as np
import pandas as pd
import os
import sys
import xarray as xr
from config_paths import get_input_base, get_src_base
from runtime_data_cache import read_excel_cached as _runtime_read_excel_cached

from S1_0_schema import Universe, ScenarioConfig, Node

# File-check helpers
def _check_file_exists(file_path: str, file_description: str, critical: bool = True) -> bool:
    """
    Check file existence and report missing files.
    
    Args:
        file_path: File path.
        file_description: Description for error messages.
        critical: Whether a missing file terminates the program.
    
    Returns:
        Whether the file exists.
    """
    if not os.path.exists(file_path):
        error_msg = f"\n{'='*80}\n? 错误: 找不到{file_description}\n文件路径: {file_path}\n{'='*80}\n"
        print(error_msg, file=sys.stderr)
        if critical:
            sys.exit(1)
        return False
    return True


def _norm_m49(val) -> str:
    """
    Normalize M49 to a leading apostrophe plus three digits.
    Examples: apostrophe-wrapped 156, 0156, integer 156, and string 156 all become apostrophe-prefixed 156.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    s = str(val).strip()
    if s.startswith("'"):
        s = s[1:]
    s = s.strip()
    if not s:
        return ''
    if s.count('.') == 1:
        left, right = s.split('.', 1)
        if left.isdigit() and right.strip('0') == '':
            s = left
    if s.isdigit():
        return f"'{s.zfill(3)}"
    return f"'{s}"

# paths
@dataclass
class DataPaths:
    base: str = get_input_base()
    luh2_data_dir: str = os.path.join(get_input_base(), "Land", "LUH2_GCB2019", "data")
    # config/dictionaries under src
    dict_v3_path: str = os.path.join(get_src_base(), "dict_v3.xlsx")
    scenario_config_xlsx: str = os.path.join(get_src_base(), "Scenario_config_new.xlsx")
    elasticity_xlsx: str = os.path.join(get_input_base(), "Driver", "Elasticity", "Elasticity_v3_processed_filled_by_region.xlsx")
    feed_coeff_xlsx: str = os.path.join(get_input_base(), "Land", "Feed_pasture", "Feed_need_per_head_by_country_livestcok_refilled.xlsx")
    feed_need_xlsx: str = os.path.join(get_input_base(), "Land", "Feed_pasture", "Feed_need_per_head_by_country_livestcok_refilled.xlsx")
    grass_ratio_xlsx: str = os.path.join(get_input_base(), "Land", "Feed_pasture", "Grass_feed_ratio_by_country_livestock_refilled.xlsx")
    pasture_dm_yield_xlsx: str = os.path.join(get_input_base(), "Land", "Feed_pasture", "Pasture_DM_yield_by_country.xlsx")
    # inputs
    production_faostat_csv: str = os.path.join(get_input_base(), "Production_Trade", "Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled_baseYearFilled.csv")
    fbs_csv: str = os.path.join(get_input_base(), "Production_Trade", "FoodBalanceSheets_E_All_Data_NOFLAG_demand_refilled.xlsx")
    livestock_patterns_csv: str = os.path.join(get_input_base(), "Manure_Stock", "Environment_LivestockManure_with_ratio.csv")  # Point to the correct livestock stock file.
    inputs_landuse_csv: str = os.path.join(get_input_base(), "Constraint", "Inputs_LandUse_E_All_Data_NOFLAG.csv")
    fertilizer_efficiency_xlsx: str = os.path.join(get_input_base(), "Fertilizer", "Fertilizer_efficiency.xlsx")
    prices_csv: str = os.path.join(get_input_base(), "Price_Cost", "Price", "World_Production_Value_per_Unit.xlsx")
    price_wedge_xlsx: str = os.path.join(get_input_base(), "Price_Cost", "Price", "Price_wedge.xlsx")
    armington_xlsx: str = os.path.join(get_input_base(), "Production_Trade", "Armington_elasticity.xlsx")
    trade_crops_xlsx: str = os.path.join(get_input_base(), "Production_Trade", "Trade_CropsLivestock_E_All_Data_NOFLAG_filtered.xlsx")
    trade_forestry_csv: str = os.path.join(get_input_base(), "Production_Trade", "Forestry_E_All_Data_NOFLAG.csv")
    luh2_states_nc: str = os.path.join(get_input_base(), "Land", "LUH2_GCB2019", "data", "LUH2_GCB2019_states_2010_2020.nc4")
    luh2_transitions_nc: str = os.path.join(get_input_base(), "Land", "LUH2_GCB2019", "data", "LUH2_GCB2019_transitions_2010_2020.nc4")
    luh2_mask_nc: str = os.path.join(get_input_base(), "Land", "LUH2_GCB2019", "data", "mask_LUH2_025d.nc")
    luc_param_xlsx: str = os.path.join(get_src_base(), "LUCE_parameter.xlsx")
    # Simplified land-cover baseline file, preferred by default
    land_cover_base_xlsx: str = os.path.join(get_input_base(), "Land", "Land_cover_base_refill.xlsx")
    # Precomputed Forest EF file path
    luce_forest_ef_xlsx: str = os.path.join(get_src_base(), "LUCE_parameter.xlsx")
    # Simplified land-cover switch: True reads Excel; False extracts LUH2 NetCDF.
    use_simplified_land_cover: bool = True
    # Forest EF switch: True reads precomputed values; False derives them from historical data.
    use_precomputed_forest_ef: bool = True
    # optional price/cost sources under Price_Cost
    faostat_prices_csv: str = os.path.join(get_input_base(), "Price_Cost", "Price", "Prices_E_All_Data_NOFLAG.csv")
    macc_pkl: str = os.path.join(get_input_base(), "Price_Cost", "Cost", "MACC-Global-US.pkl")
    # Canonical, versioned nine-strategy cost database used by the solver.
    # The legacy XLSX path is retained for explicit backwards compatibility,
    # but new unit-cost runs load and validate this JSON file fail-closed.
    mitigation_cost_database_path: str = os.path.join(
        get_input_base(),
        "Price_Cost",
        "Cost",
        "food_mitigation_cost_database_v2.0.json",
    )
    unit_cost_xlsx: str = os.path.join(get_input_base(), "Price_Cost", "Cost", "MACC_2080_GapFilled_Final_overZero.xlsx")
    # constraints
    intake_constraint_xlsx: str = os.path.join(get_input_base(), "Constraint", "Intake_constraint.xlsx")
    # optional emissions csvs
    emis_fires_csv: str = os.path.join(get_input_base(), "Emissions_Land_Use_Fires_E_All_Data_NOFLAG.csv")
    # drivers and others
    population_wpp_csv: str = os.path.join(get_input_base(), "Driver", "Population", "WPP", "Population_E_All_Data_NOFLAG.csv")
    temperature_xlsx: str = os.path.join(get_input_base(), "Driver", "Temperature", "SSP_IAM_V2_201811_Temperature.xlsx")
    income_sspdb_xlsx: str = os.path.join(get_input_base(), "Driver", "Income", "SSPDB_future_GDP_with_change_ratio.xlsx")
    sspdb_scenario: str = "SSP2_v9_130325"
    production_trade_fbs_csv: str = os.path.join(get_input_base(), "Production_Trade", "FoodBalanceSheets_E_All_Data_NOFLAG_demand_refilled.xlsx")
    nonfood_balance_csv: str = os.path.join(get_input_base(), "Production_Trade", "CommodityBalances_(non-food)_(2010-)_E_All_Data_NOFLAG.csv")
    forestry_csv: str = os.path.join(get_input_base(), "Production_Trade", "Forestry_E_All_Data_NOFLAG.csv")
    bioenergy_faostat_csv: str = os.path.join(get_input_base(), "Production_Trade", "Environment_Bioenergy_E_All_Data_NOFLAG.csv")
    bioenergy_historical_feedstock_csv: str = os.path.join(get_input_base(), "Bioenergy", "bioenergy_historical_feedstock.csv")
    bioenergy_scenario_csv: str = os.path.join(get_input_base(), "Bioenergy", "bioenergy_scenario_targets.csv")
    bioenergy_feedstock_parameters_csv: str = os.path.join(get_input_base(), "Bioenergy", "bioenergy_feedstock_parameters.csv")
    bioenergy_carrier_feedstock_share_csv: str = os.path.join(get_input_base(), "Bioenergy", "bioenergy_carrier_feedstock_share.csv")
    bioenergy_resource_constraints_csv: str = os.path.join(get_input_base(), "Bioenergy", "bioenergy_resource_constraints.csv")
    manure_stock_csv: str = os.path.join(get_input_base(), "Manure_Stock", "Environment_LivestockManure_E_All_Data_NOFLAG.csv")
    manure_stock_with_ratio_csv: str = os.path.join(get_input_base(), "Manure_Stock", "Environment_LivestockManure_with_ratio.csv")

# helpers
def _lc(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z.columns = [str(c).strip() for c in z.columns]
    return z

_READ_CACHE_MAXSIZE = 64

def _file_mtime(path: Any) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

def _file_read_signature(path: Any) -> Tuple[Optional[int], Optional[int]]:
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_size
    except Exception:
        return None, None

@lru_cache(maxsize=_READ_CACHE_MAXSIZE)
def _excel_sheet_names_cached(path_str: str, mtime: Optional[float]) -> Tuple[str, ...]:
    return tuple(pd.ExcelFile(path_str).sheet_names)

def _excel_sheet_names(path: Any) -> Tuple[str, ...]:
    if not isinstance(path, (str, os.PathLike)):
        return tuple(pd.ExcelFile(path).sheet_names)
    return _excel_sheet_names_cached(str(path), _file_mtime(path))

def _freeze_value(val: Any) -> Any:
    if isinstance(val, (list, tuple)):
        return tuple(_freeze_value(v) for v in val)
    if isinstance(val, dict):
        return tuple(sorted((k, _freeze_value(v)) for k, v in val.items()))
    if isinstance(val, set):
        return tuple(sorted(_freeze_value(v) for v in val))
    try:
        hash(val)
    except Exception:
        return repr(val)
    return val

def _freeze_kwargs(kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((k, _freeze_value(v)) for k, v in kwargs.items()))

def _freeze_args(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    return tuple(_freeze_value(v) for v in args)

def _copy_df_result(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = v.copy() if isinstance(v, pd.DataFrame) else v
        return out
    return obj

@lru_cache(maxsize=_READ_CACHE_MAXSIZE)
def _read_excel_cached_impl(path_str: str,
                            file_signature: Tuple[Optional[int], Optional[int]],
                            args_key: Tuple[Any, ...],
                            kwargs_key: Tuple[Tuple[str, Any], ...]) -> Any:
    args = tuple(args_key)
    kwargs = dict(kwargs_key)
    return _runtime_read_excel_cached(path_str, *args, copy=False, **kwargs)

def _read_excel(path: Any, *args, **kwargs) -> Any:
    if isinstance(path, pd.ExcelFile):
        return pd.read_excel(path, *args, **kwargs)
    if not isinstance(path, (str, os.PathLike)):
        return pd.read_excel(path, *args, **kwargs)
    try:
        args_key = _freeze_args(args)
        kwargs_key = _freeze_kwargs(kwargs)
    except Exception:
        return pd.read_excel(path, *args, **kwargs)
    df = _read_excel_cached_impl(
        str(path), _file_read_signature(path), args_key, kwargs_key
    )
    return _copy_df_result(df)

@lru_cache(maxsize=_READ_CACHE_MAXSIZE)
def _read_csv_cached_impl(path_str: str,
                          file_signature: Tuple[Optional[int], Optional[int]],
                          args_key: Tuple[Any, ...],
                          kwargs_key: Tuple[Tuple[str, Any], ...]) -> Any:
    args = tuple(args_key)
    kwargs = dict(kwargs_key)
    return pd.read_csv(path_str, *args, **kwargs)

def _read_csv(path: Any, *args, **kwargs) -> Any:
    if not isinstance(path, (str, os.PathLike)):
        return pd.read_csv(path, *args, **kwargs)
    if kwargs.get('chunksize') is not None or kwargs.get('iterator') is not None:
        return pd.read_csv(path, *args, **kwargs)
    try:
        args_key = _freeze_args(args)
        kwargs_key = _freeze_kwargs(kwargs)
    except Exception:
        return pd.read_csv(path, *args, **kwargs)
    df = _read_csv_cached_impl(
        str(path), _file_read_signature(path), args_key, kwargs_key
    )
    return _copy_df_result(df)

def _read_fbs_table(path: str, **kwargs) -> pd.DataFrame:
    if str(path).lower().endswith(('.xlsx', '.xls')):
        return _read_excel(path, **kwargs)
    return _read_csv(path, **kwargs)

def _faostat_wide_to_long(df: pd.DataFrame, value_name: str = 'Value') -> pd.DataFrame:
    """Convert FAOSTAT-style wide year columns (Y1961, ...) into long format."""
    if df is None or len(df) == 0:
        return df
    year_cols = [c for c in df.columns if isinstance(c, str) and c.strip().startswith('Y') and c.strip()[1:].isdigit()]
    if not year_cols:
        return df
    # Exclude 'Year' from id_cols to avoid duplicate columns after melt
    id_cols = [c for c in df.columns if c not in year_cols and c != 'Year']
    long_df = df.melt(id_vars=id_cols, value_vars=year_cols,
                      var_name='Year', value_name=value_name)
    
    long_df['Year'] = pd.to_numeric(long_df['Year'].astype(str).str.strip().str.lstrip('Y'), errors='coerce')
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors='coerce')
    long_df = long_df.dropna(subset=['Year'])
    long_df['Year'] = long_df['Year'].astype(int)
    return long_df

def _filter_select_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where FAOSTAT Select flag==1 when column exists."""
    if df is None or 'Select' not in df.columns:
        return df
    mask = pd.to_numeric(df['Select'], errors='coerce')
    return df[mask == 1]

def _find_col(df: pd.DataFrame, names: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    # fuzzy
    for c in df.columns:
        for n in names:
            if n.lower() in str(c).lower():
                return c
    raise KeyError(f"columns {names} not found")

def _maybe_find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    try:
        return _find_col(df, names)
    except Exception:
        return None

def _tuple_field(name: str) -> str:
    """Convert a column name to the attribute name created by DataFrame.itertuples."""
    # Replace non-word characters with underscores and prefix underscores for leading digits
    return re.sub(r'\W|^(?=\d)', '_', str(name))


def _build_elasticity_map(df: pd.DataFrame, value_col: str = 'Elasticity_mean') -> Dict[Tuple[str, str], float]:
    """Return {(m49, commodity) -> elasticity} using M49 as primary key."""
    out: Dict[Tuple[str, str], float] = {}
    if df is None or df.empty:
        return out
    required = {'M49_Country_Code', 'Commodity', value_col}
    if not required.issubset(df.columns):
        return out
    for r in df[['M49_Country_Code', 'Commodity', value_col]].itertuples(index=False):
        m49_raw = getattr(r, _tuple_field('M49_Country_Code'))
        m49 = _norm_m49(m49_raw)
        if not m49:
            continue
        commodity_raw = str(getattr(r, _tuple_field('Commodity'))).strip()
        val = pd.to_numeric(getattr(r, _tuple_field(value_col)), errors='coerce')
        if pd.isna(val):
            continue
        out[(m49, commodity_raw)] = float(val)
    return out

def _build_country_elasticity(df: pd.DataFrame, value_col: str = 'Elasticity_mean') -> Dict[str, float]:
    """Return {m49 -> elasticity} using M49 as primary key."""
    out: Dict[str, float] = {}
    if df is None or df.empty:
        return out
    required = {'M49_Country_Code', value_col}
    if not required.issubset(df.columns):
        return out
    for r in df[['M49_Country_Code', value_col]].itertuples(index=False):
        m49_raw = getattr(r, _tuple_field('M49_Country_Code'))
        m49 = _norm_m49(m49_raw)
        if not m49:
            continue
        val = pd.to_numeric(getattr(r, _tuple_field(value_col)), errors='coerce')
        if pd.isna(val):
            continue
        out[m49] = float(val)
    return out

# Mapping helpers between Item_Emis and Item_Elasticity_Map
def _build_emis_to_elast_map() -> Dict[str, str]:
    """Item_Emis -> Item_Elasticity_Map"""
    mapping: Dict[str, str] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        df = _lc(_read_excel(dict_path, sheet_name='Emis_item'))
        if {'Item_Emis', 'Item_Elasticity_Map'}.issubset(df.columns):
            for r in df[['Item_Emis', 'Item_Elasticity_Map']].dropna().itertuples(index=False):
                emis = str(getattr(r, _tuple_field('Item_Emis'))).strip()
                elast = str(getattr(r, _tuple_field('Item_Elasticity_Map'))).strip()
                if emis and elast and elast.lower() not in {'nan', 'no'}:
                    mapping[emis] = elast
    except Exception:
        pass
    return mapping

def _build_elast_to_emis_map() -> Dict[str, str]:
    """Item_Elasticity_Map -> Item_Emis"""
    mapping: Dict[str, str] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        df = _lc(_read_excel(dict_path, sheet_name='Emis_item'))
        if {'Item_Emis', 'Item_Elasticity_Map'}.issubset(df.columns):
            for r in df[['Item_Emis', 'Item_Elasticity_Map']].dropna().itertuples(index=False):
                emis = str(getattr(r, _tuple_field('Item_Emis'))).strip()
                elast = str(getattr(r, _tuple_field('Item_Elasticity_Map'))).strip()
                if emis and elast and elast.lower() not in {'nan', 'no'}:
                    mapping[elast] = emis
    except Exception:
        pass
    return mapping

def _build_elast_to_emis_multi() -> Dict[str, List[str]]:
    """Item_Elasticity_Map -> [Item_Emis...], potentially one-to-many, e.g. Milk to several dairy nodes."""
    mapping: Dict[str, List[str]] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        df = _lc(_read_excel(dict_path, sheet_name='Emis_item'))
        if {'Item_Emis', 'Item_Elasticity_Map'}.issubset(df.columns):
            for r in df[['Item_Emis', 'Item_Elasticity_Map']].dropna().itertuples(index=False):
                emis = str(getattr(r, _tuple_field('Item_Emis'))).strip()
                elast = str(getattr(r, _tuple_field('Item_Elasticity_Map'))).strip()
                if not (emis and elast) or elast.lower() in {'nan', 'no'}:
                    continue
                mapping.setdefault(elast, []).append(emis)
    except Exception:
        pass
    return mapping

def _build_cross_elasticity_map(df: pd.DataFrame, commodity_filter: Optional[set] = None) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return {(m49, commodity) -> {other_commodity: elasticity}} for cross-price sheets."""
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    if df is None or df.empty:
        return out
    if 'M49_Country_Code' not in df.columns or 'Commodity' not in df.columns:
        return out
    # Build elasticity_name -> Item_Emis mapping from dict_v3
    elast_to_prod: Dict[str, str] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        mapping_df = _lc(_read_excel(dict_path, sheet_name='Emis_item'))
        if {'Item_Elasticity_Map', 'Item_Emis'}.issubset(mapping_df.columns):
            for r in mapping_df[['Item_Elasticity_Map', 'Item_Emis']].dropna().itertuples(index=False):
                elast_name = str(r.Item_Elasticity_Map).strip()
                emis_name = str(r.Item_Emis).strip()
                if elast_name and emis_name and elast_name.lower() not in {'nan', 'no'} and emis_name.lower() not in {'nan', 'no'}:
                    elast_to_prod[elast_name] = emis_name
    except Exception as e:
        print(f"Warning: Could not load commodity name mapping from dict_v3: {e}")
    attr_by_col = {col: _tuple_field(col) for col in df.columns}
    base_cols = {'Commodity', 'M49_Country_Code'}
    for row in df.itertuples(index=False):
        m49_raw = getattr(row, attr_by_col.get('M49_Country_Code', ''))
        m49 = _norm_m49(m49_raw)
        if not m49:
            continue
        commodity = str(getattr(row, attr_by_col['Commodity'])).strip()
        key = (m49, commodity)
        cross: Dict[str, float] = {}
        for col, attr in attr_by_col.items():
            if col in base_cols:
                continue
            mapped_col = elast_to_prod.get(col, col)
            if commodity_filter is not None and mapped_col not in commodity_filter:
                continue
            val = pd.to_numeric(getattr(row, attr), errors='coerce')
            if pd.isna(val):
                continue
            cross[mapped_col] = float(val)
        if cross:
            out[key] = cross
    return out

_AUTHORITATIVE_PROCESS_COST_OWNERS: Dict[str, str] = {
    'Enteric fermentation': 'EntericF',
    'Manure management': 'Manure',
    'Manure applied to soils': 'Manure',
    'Manure left on pasture': 'Manure',
    'Burning crop residues': 'Residue',
    'Crop residues': 'Residue',
    'Rice cultivation': 'Rice',
    'Synthetic fertilizers': 'Fertilizer',
    'Ag land abandonment_crop': 'Production value',
    'Ag land abandonment_pasture': 'Production value',
    'De/Reforestation_crop': 'Production value',
    'De/Reforestation_pasture': 'Production value',
}


def load_process_cost_mapping(dict_v3_path: str) -> Dict[str, str]:
    """
    Read Process-to-Process_Cost_Map mappings from dict_v3.xlsx Emis_item.

    Returns: {Process: Process_Cost_Map}.
    """
    mapping: Dict[str, str] = {}
    try:
        df = _lc(_read_excel(dict_v3_path, sheet_name='Emis_item'))
        if 'Process' not in df.columns or 'Process_Cost_Map' not in df.columns:
            print("Warning: dict_v3 Emis_item表缺少Process或Process_Cost_Map列")
            return mapping
        for r in df[['Process', 'Process_Cost_Map']].dropna().itertuples(index=False):
            process = str(getattr(r, _tuple_field('Process'))).strip()
            cost_map = str(getattr(r, _tuple_field('Process_Cost_Map'))).strip()
            if not process or not cost_map or cost_map.lower() in {'none', 'no', 'nan'}:
                continue
            expected_owner = _AUTHORITATIVE_PROCESS_COST_OWNERS.get(process)
            if expected_owner is None or cost_map != expected_owner:
                logging.getLogger(__name__).warning(
                    "Ignoring unsupported process-cost ownership %r -> %r",
                    process,
                    cost_map,
                )
                continue
            mapping[process] = cost_map
    except Exception as e:
        print(f"Warning: 无法读取Process_Cost_Map映射: {e}")
    return mapping


def load_unit_cost_data(cost_xlsx_path: str, dict_v3_path: str) -> Tuple[Dict[Tuple[str, str], float], Dict[str, str]]:
    """
    Read unit abatement costs from MACC_2080_GapFilled_Final_overZero.xlsx.

    Returns:
        - unit_costs: {(M49_Country_Code, process_cost_name): USD_per_tCO2e}
        - process_mapping: {process: process_cost_map}
    """
    unit_costs: Dict[Tuple[str, str], float] = {}
    process_mapping = load_process_cost_mapping(dict_v3_path)

    if not os.path.exists(cost_xlsx_path):
        print(f"Warning: 成本数据文件不存在: {cost_xlsx_path}")
        return unit_costs, process_mapping

    try:
        df = _read_excel(cost_xlsx_path, sheet_name='Gap_Filled_Data')
        required_cols = {'M49_Country_Code', 'Process', 'Final_Unit_Cost'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Error: 成本数据缺少必要列: {missing}")
            return unit_costs, process_mapping

        for _, row in df.iterrows():
            try:
                m49_raw = row['M49_Country_Code']
                process_cost = str(row['Process']).strip()
                unit_cost = float(row['Final_Unit_Cost'])
                if pd.isna(unit_cost) or unit_cost < 0:
                    continue
                m49 = _norm_m49(m49_raw)
                if not m49 or m49.lower() in {'nan', 'no'}:
                    continue
                unit_costs[(m49, process_cost)] = unit_cost
            except Exception:
                continue

        print(f"成功读取 {len(unit_costs)} 条单位减排成本数据")
        print(f"成本数据覆盖的Process: {sorted(set(p for _, p in unit_costs.keys()))}")
    except Exception as e:
        print(f"Error: 读取成本数据失败: {e}")
        import traceback
        traceback.print_exc()

    return unit_costs, process_mapping

def _country_by_m49(df: pd.DataFrame, universe: Universe, *, context: str = '') -> Optional[pd.Series]:
    """Return Series of standardized M49 codes using explicit M49 column."""
    if df is None:
        return None
    if 'M49_Country_Code' not in df.columns:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}缺少 M49_Country_Code 列")
    codes = df['M49_Country_Code'].apply(_norm_m49)
    return codes

def _attach_country_from_m49(df_source: pd.DataFrame,
                             df_target: pd.DataFrame,
                             universe: Universe,
                             *,
                             context: str) -> pd.DataFrame:
    """Assign country column to standardized M49 codes, and add country_name for display."""
    m49_codes = _country_by_m49(df_source, universe, context=context)
    if m49_codes is None:
        raise ValueError(f"{context}: 缺少 M49_Country_Code 列，无法进行国家映射")
    df_target = df_target.copy()
    df_target['M49_Country_Code'] = m49_codes.reindex(df_target.index)
    df_target['country'] = df_target['M49_Country_Code']
    df_target['country_name'] = df_target['country'].map(universe.country_by_m49 or {})
    df_target = df_target.dropna(subset=['country'])
    return df_target

def _estimate_area_ha_from_grid(ds: xr.Dataset) -> np.ndarray:
    """Return grid-cell areas in hectares aligned with ds (lat, lon)."""
    if 'areacella' in ds:
        return np.asarray(ds['areacella'].values, dtype=float) * 1e-4
    R = 6_371_000.0
    lat = np.asarray(ds['lat'].values, dtype=float)
    lon = np.asarray(ds['lon'].values, dtype=float)
    if lat.size < 2 or lon.size < 2:
        raise ValueError("LUH2 dataset lat/lon dimensions are insufficient to estimate cell area")
    dlat = np.deg2rad(abs(lat[1] - lat[0]))
    dlon = np.deg2rad(abs(lon[1] - lon[0]))
    lat_r = np.deg2rad(lat)
    strip = (np.sin(lat_r + dlat / 2.0) - np.sin(lat_r - dlat / 2.0)) * (R ** 2) * dlon
    area_lat = strip  # m2 per latitude band
    # broadcast to grid shape (lat, lon)
    area = np.repeat(area_lat[:, None], lon.size, axis=1)
    return area * 1e-4  # convert m2 to ha


def load_luh2_land_cover(states_nc_path: str,
                         mask_nc_path: str,
                         universe: Universe,
                         years: Optional[List[int]] = None) -> pd.DataFrame:
    """Aggregate LUH2 land-cover fractions (states file) to country-level cropland/pasture/forest areas."""
    columns = ['M49_Country_Code', 'country', 'country_name', 'iso3', 'year', 'land_use', 'area_ha']
    if not (states_nc_path and os.path.exists(states_nc_path)):
        return pd.DataFrame(columns=columns)
    if not (mask_nc_path and os.path.exists(mask_nc_path)):
        return pd.DataFrame(columns=columns)

    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        region_df = _lc(_read_excel(dict_path, 'region'))
    except Exception:
        region_df = pd.DataFrame(columns=['Region_label_new', 'Region_maskID', 'ISO3 Code', 'M49_Country_Code'])

    c_label = _find_col(region_df, ['Region_label_new'])
    c_mask = _find_col(region_df, ['Region_maskID'])
    c_iso3 = _find_col(region_df, ['ISO3', 'ISO3 Code'])
    c_m49 = _find_col(region_df, ['M49_Country_Code', 'M49 Code', 'M49'])
    region_df = region_df[[c_label, c_mask, c_iso3, c_m49]].dropna()
    region_df[c_mask] = pd.to_numeric(region_df[c_mask], errors='coerce').astype('Int64')
    region_df = region_df.dropna(subset=[c_mask])
    region_df['m49_code'] = region_df[c_m49].apply(_norm_m49)
    region_df = region_df.dropna(subset=['m49_code'])

    mask_id_to_m49 = {int(getattr(r, _tuple_field(c_mask))): str(getattr(r, _tuple_field('m49_code'))).strip()
                      for r in region_df.itertuples(index=False)}
    mask_id_to_country_name = {int(getattr(r, _tuple_field(c_mask))): str(getattr(r, _tuple_field(c_label))).strip()
                               for r in region_df.itertuples(index=False)}
    mask_id_to_iso3: Dict[int, str] = {}
    for mid, m49 in mask_id_to_m49.items():
        iso = universe.iso3_by_country.get(m49)
        if iso:
            mask_id_to_iso3[mid] = iso
    if not mask_id_to_iso3:
        return pd.DataFrame(columns=columns)

    target_years = sorted({int(y) for y in (years if years else universe.years or [])})

    with xr.open_dataset(states_nc_path) as ds:
        all_years = [int(getattr(t, 'year', getattr(t, 'year', t))) for t in ds['time'].values]
        ds = ds.assign_coords(year=('time', all_years)).swap_dims({'time': 'year'}).sortby('year')
        available_years = set(int(y) for y in ds['year'].values.tolist())
        if not target_years:
            target_years = sorted(available_years)
        max_available_year = max(available_years)
        source_years = sorted({y for y in target_years if y in available_years} | {max_available_year})
        area_grid = _estimate_area_ha_from_grid(ds.isel(year=0))

        with xr.open_dataset(mask_nc_path) as mask_ds:
            if 'id1' not in mask_ds:
                raise KeyError("Mask NetCDF must contain variable 'id1'")
            id_array = np.asarray(mask_ds['id1'].values, dtype=float)

        id_array = np.nan_to_num(id_array, nan=0.0)
        id_int = id_array.astype(np.int64)
        flat_ids = id_int.ravel()
        valid_idx = np.where(flat_ids > 0)[0]
        if not len(valid_idx):
            return pd.DataFrame(columns=columns)
        valid_ids = flat_ids[valid_idx]
        unique_ids, inverse_idx = np.unique(valid_ids, return_inverse=True)

        category_states = {
            'cropland_area_ha': ['c3ann', 'c3per', 'c4ann', 'c4per', 'c3nfx'],
            'pasture_area_ha': ['pastr', 'range'],
            'forest_area_ha': ['primf', 'secdf'],
        }

        cache: Dict[int, Dict[str, np.ndarray]] = {}
        for year in source_years:
            cat_sums: Dict[str, np.ndarray] = {}
            for cat_name, states in category_states.items():
                total = None
                for state in states:
                    if state not in ds:
                        continue
                    arr = ds[state].sel(year=year).values
                    arr = np.nan_to_num(arr, nan=0.0, copy=False)
                    total = arr if total is None else total + arr
                if total is None:
                    cat_sums[cat_name] = np.zeros(len(unique_ids), dtype=float)
                else:
                    weighted = (total * area_grid).reshape(-1)
                    cat_sums[cat_name] = np.bincount(
                        inverse_idx,
                        weights=weighted[valid_idx],
                        minlength=len(unique_ids)
                    )
            cache[year] = cat_sums

    records: List[Dict[str, Any]] = []
    for year in target_years:
        src_year = year if year in cache else max_available_year
        cat_sums = cache.get(src_year)
        if not cat_sums:
            continue
        for idx, mask_id in enumerate(unique_ids):
            mask_id_int = int(mask_id)
            m49_code = mask_id_to_m49.get(mask_id_int)
            iso3 = mask_id_to_iso3.get(mask_id_int)
            if not m49_code or not iso3:
                continue
            country_name = mask_id_to_country_name.get(mask_id_int, '')
            for cat_name, sums in cat_sums.items():
                if idx >= len(sums):
                    continue
                area_val = float(sums[idx])
                if area_val <= 0.0:
                    continue
                records.append({
                    'M49_Country_Code': m49_code,
                    'country': m49_code,
                    'country_name': country_name,
                    'iso3': iso3,
                    'year': year,
                    'land_use': cat_name,
                    'area_ha': area_val,
                })

    if not records:
        return pd.DataFrame(columns=columns)
    out_df = pd.DataFrame.from_records(records, columns=columns)
    group_cols = ['M49_Country_Code', 'country', 'country_name', 'iso3', 'year', 'land_use']
    # Distinct LUH2 mask regions mapped to one country contribute additive area.
    out_df = out_df.groupby(group_cols, as_index=False)['area_ha'].sum()
    return out_df


def load_land_cover_from_excel(excel_path: str,
                               universe: Universe,
                               years: Optional[List[int]] = None,
                               source: str = 'LUH2') -> pd.DataFrame:
    """
    Read simplified Excel land-cover data instead of extracting LUH2 NetCDF.

    source = 'LUH2':
      - sheet: LUH2
      - M49_Country_Code, Land cover, Y2010..Y2020, Unit=ha

    source = 'FAO':
      - sheet: FAO
      - M49_Country_Code, Item, Element, Yxxxx, Unit=1000 ha
      - Retain Element=Area and Item in {cropland, grassland, forest}.

    Return the same format as load_luh2_land_cover:
    - country, iso3, year, land_use, area_ha
    """
    columns = ['M49_Country_Code', 'country', 'country_name', 'iso3', 'year', 'land_use', 'area_ha']
    
    if not (excel_path and os.path.exists(excel_path)):
        print(f"[WARN] load_land_cover_from_excel: 文件不存在 {excel_path}")
        return pd.DataFrame(columns=columns)
    
    source_norm = str(source or 'LUH2').strip().upper()
    sheet_name = 'LUH2' if source_norm == 'LUH2' else 'FAO' if source_norm == 'FAO' else None
    if sheet_name is None:
        raise ValueError(f"load_land_cover_from_excel: 未知source={source}")

    try:
        df = _read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"[ERROR] load_land_cover_from_excel: 读取Excel失败 {e}")
        return pd.DataFrame(columns=columns)

    # Normalize column names.
    df.columns = [str(c).strip() for c in df.columns]

    if source_norm == 'LUH2':
        m49_col = None
        land_cover_col = None
        for c in df.columns:
            cl = c.lower()
            if 'm49' in cl:
                m49_col = c
            elif 'land cover' in cl or 'land_cover' in cl:
                land_cover_col = c
        if not m49_col or not land_cover_col:
            raise ValueError(f"load_land_cover_from_excel: {excel_path} 缺少必要列 M49={m49_col}, Land_cover={land_cover_col}")
    else:
        m49_col = _find_col(df, ['M49_Country_Code', 'M49'])
        item_col = _find_col(df, ['Item'])
        elem_col = _find_col(df, ['Element'])
        df = df[df[elem_col].astype(str).str.strip().str.lower() == 'area']
        if df.empty:
            print(f"[WARN] load_land_cover_from_excel: FAO表格中未找到Element=Area")
            return pd.DataFrame(columns=columns)

    year_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    if not year_cols:
        print(f"[ERROR] load_land_cover_from_excel: 未找到年份列 (Y2010, Y2011, ...)")
        return pd.DataFrame(columns=columns)

    records = []

    for _, row in df.iterrows():
        m49_raw = row.get(m49_col)
        if pd.isna(m49_raw):
            continue
        m49_str = _norm_m49(str(m49_raw))

        country_name = universe.country_by_m49.get(m49_str, '')
        iso3 = universe.iso3_by_country.get(m49_str, '')
        if not iso3:
            continue

        if source_norm == 'LUH2':
            land_type_raw = row.get(land_cover_col)
            if pd.isna(land_type_raw):
                continue
            land_type = str(land_type_raw).strip().lower()
            if 'forest' in land_type:
                land_use = 'forest'
            elif 'grass' in land_type or 'pasture' in land_type:
                land_use = 'grassland'
            elif 'crop' in land_type:
                land_use = 'cropland'
            else:
                continue
            unit_scale = 1.0
        else:
            item_raw = row.get(item_col)
            if pd.isna(item_raw):
                continue
            item_norm = str(item_raw).strip().lower()
            if item_norm not in {'cropland', 'grassland', 'forest'}:
                continue
            land_use = item_norm
            unit_scale = 1000.0

        for yc in year_cols:
            try:
                year = int(yc[1:])
            except ValueError:
                continue

            if years and year not in years:
                continue

            val = row.get(yc)
            if pd.isna(val):
                continue
            val_num = pd.to_numeric(val, errors='coerce')
            if pd.isna(val_num):
                continue
            area_ha = float(val_num) * unit_scale

            records.append({
                'M49_Country_Code': m49_str,
                'country': m49_str,
                'country_name': country_name,
                'iso3': iso3,
                'year': year,
                'land_use': land_use,
                'area_ha': area_ha,
            })
    
    if not records:
        print(f"[WARN] load_land_cover_from_excel: 未生成任何记录")
        return pd.DataFrame(columns=columns)
    
    out_df = pd.DataFrame.from_records(records, columns=columns)
    group_cols = ['M49_Country_Code', 'country', 'country_name', 'iso3', 'year', 'land_use']
    if source_norm == 'FAO':
        dup_count = int(out_df.duplicated(subset=group_cols, keep=False).sum())
        if dup_count > 0:
            print(
                f"[WARN] load_land_cover_from_excel: FAO sheet contains {dup_count} duplicate "
                f"(country, year, land_use) rows; use max(area_ha) instead of sum to avoid double counting"
            )
        out_df = out_df.groupby(group_cols, as_index=False)['area_ha'].max()
    else:
        out_df = out_df.groupby(group_cols, as_index=False)['area_ha'].sum()
    
    print(f"[INFO] load_land_cover_from_excel: 加载 {len(out_df)} 条土地覆盖记录 from {os.path.basename(excel_path)} ({source_norm})")
    return out_df


def load_forest_ef_from_excel(excel_path: str,
                              universe: Universe,
                              sheet_name: str = 'Forest_EF') -> Dict[str, Dict[int, float]]:
    """
    Read precomputed Forest emission factors from LUCE_parameter.xlsx.
    
    Excel format:
    - M49_Country_Code: Country M49 code.
    - Process: Fixed to Forest.
    - Unit: tCO2/ha/yr
    - Y2010, Y2011, ..., Y2020: Annual EF values.
    
    Returns:
    - Dict[M49, Dict[year, EF]].
    - EF units: tCO2/ha/year; negative indicates a sink.
    """
    result: Dict[str, Dict[int, float]] = {}
    
    if not (excel_path and os.path.exists(excel_path)):
        print(f"[WARN] load_forest_ef_from_excel: 文件不存在 {excel_path}")
        return result
    
    try:
        df = _read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"[ERROR] load_forest_ef_from_excel: 读取Excel失败 {e}")
        return result
    
    # Normalize column names.
    df.columns = [str(c).strip() for c in df.columns]
    
    # Find the M49 column.
    m49_col = None
    for c in df.columns:
        if 'm49' in c.lower():
            m49_col = c
            break
    
    if not m49_col:
        raise ValueError(f"load_forest_ef_from_excel: {excel_path} 未找到M49列")
    
    # Get year columns.
    year_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    if not year_cols:
        print(f"[ERROR] load_forest_ef_from_excel: 未找到年份列")
        return result
    
    for _, row in df.iterrows():
        m49_raw = row.get(m49_col)
        if pd.isna(m49_raw):
            continue
        
        m49_str = _norm_m49(str(m49_raw))
        
        # Skip World aggregate rows.
        if m49_str in {"'000", "'0"}:
            continue
        
        ef_by_year: Dict[int, float] = {}
        
        for yc in year_cols:
            try:
                year = int(yc[1:])
            except ValueError:
                continue
            
            val = row.get(yc)
            if pd.notna(val):
                ef_by_year[year] = float(val)
        
        if ef_by_year:
            result[m49_str] = ef_by_year
    
    print(f"[INFO] load_forest_ef_from_excel: 加载 {len(result)} 个国家的Forest EF from {os.path.basename(excel_path)}")
    return result


def load_roundwood_supply(forestry_csv_path: str,
                          universe: Universe,
                          years: Optional[List[int]] = None) -> pd.DataFrame:
    """Load FAOSTAT forestry production for Roundwood as m3 per country-year."""
    columns = ['M49_Country_Code', 'country', 'iso3', 'year', 'roundwood_m3']
    if not (forestry_csv_path and os.path.exists(forestry_csv_path)):
        return pd.DataFrame(columns=columns)
    df_raw = _read_csv(forestry_csv_path)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val = _find_col(df, ['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])
    if not all([c_area, c_year, c_item, c_elem, c_val]):
        return pd.DataFrame(columns=columns)
    keep_cols = [c_area, c_year, c_item, c_elem, c_val] + ([c_unit] if c_unit else [])
    if 'M49_Country_Code' in df.columns:
        keep_cols.append('M49_Country_Code')
    z = df[keep_cols].copy()
    rename_cols = {c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'}
    if c_unit:
        rename_cols[c_unit] = 'unit'
    z = z.rename(columns=rename_cols)
    z = _attach_country_from_m49(df, z, universe, context=f"Roundwood supply ({forestry_csv_path})")
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df['M49_Country_Code']
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.production_by_item).fillna(z['item_raw'])
    except Exception:
        z['commodity'] = z['item_raw']
    z = z[(z['commodity'].astype(str).str.strip().str.lower() == 'roundwood') &
          (z['country'].isin(universe.countries))]
    if z.empty:
        return pd.DataFrame(columns=columns)
    z = z[z['element'].astype(str).str.contains('production', case=False, na=False)]
    z['value'] = pd.to_numeric(z['value'], errors='coerce')
    if c_unit:
        z['unit'] = z['unit'].astype(str).str.lower()
        z['roundwood_m3'] = z.apply(
            lambda r: (r['value'] if np.isfinite(r['value']) else 0.0) *
            (1000.0 if '1000' in r['unit'] else 1.0),
            axis=1
        )
    else:
        z['roundwood_m3'] = z['value']
    z = z.dropna(subset=['roundwood_m3'])
    z['roundwood_m3'] = pd.to_numeric(z['roundwood_m3'], errors='coerce').fillna(0.0)
    if z.empty:
        return pd.DataFrame(columns=columns)
    if 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = z['country'].apply(_norm_m49)
    agg = z.groupby(['M49_Country_Code', 'year'], as_index=False)['roundwood_m3'].sum()
    agg['country'] = agg['M49_Country_Code']
    agg['iso3'] = agg['M49_Country_Code'].map(universe.iso3_by_country)
    agg = agg.dropna(subset=['iso3'])
    agg['year'] = pd.to_numeric(agg['year'], errors='coerce').astype('Int64')
    agg = agg.dropna(subset=['year'])
    agg['year'] = agg['year'].astype(int)
    if years:
        target_years = sorted({int(y) for y in years})
        if agg.empty:
            return pd.DataFrame(columns=columns)
        max_hist = int(agg['year'].max())
        frames = [agg]
        missing = [y for y in target_years if y not in agg['year'].values]
        if missing and max_hist in agg['year'].values:
            base_rows = agg[agg['year'] == max_hist]
            for y in missing:
                if y < max_hist and y in agg['year'].values:
                    continue
                tmp = base_rows.copy()
                tmp['year'] = y
                frames.append(tmp)
        agg = pd.concat(frames, ignore_index=True)
        agg = agg[agg['year'].isin(target_years)]
    return agg[['M49_Country_Code', 'country', 'iso3', 'year', 'roundwood_m3']]

# universe
def build_universe_from_dict_v3(path: str, config: ScenarioConfig) -> Universe:
    region = _lc(_read_excel(path, sheet_name='region'))
    # Extract the process list from Process in Emis_item.
    emis_item = _lc(_read_excel(path, sheet_name='Emis_item'))
    emis_proc = emis_item  # Use the same table.

    c_country = _find_col(region, ['Country'])
    c_iso3 = _find_col(region, ['ISO3'])
    c_label = _find_col(region, ['Region_label_new'])
    c_m49 = _find_col(region, ['M49_Country_Code', 'M49 Code', 'M49'])
    if not c_m49:
        raise ValueError("dict_v3.region 缺少 M49 列，无法构建M49主键Universe")
    region_clean = region[region[c_label].astype(str).str.lower() != 'no'].copy()
    region_clean['country_label'] = region_clean[c_label].astype(str).str.strip()
    region_clean['m49_code'] = region_clean[c_m49].apply(_norm_m49)
    region_clean = region_clean.dropna(subset=['m49_code'])
    region2 = region_clean[['country_label', 'm49_code', c_iso3]].drop_duplicates(subset=['m49_code'])
    countries = region2['m49_code'].tolist()
    iso3_map = dict(zip(region2['m49_code'], region2[c_iso3].astype(str)))

    # Region_aggMC map
    c_ragg = _find_col(region, ['Region_aggMC'])
    region_aggMC_by_country = dict(zip(region_clean['m49_code'], region_clean[c_ragg].astype(str)))
    # SSP region map
    c_ssp = 'Region_map_SSPDB' if 'Region_map_SSPDB' in region.columns else None
    ssp_region_by_country = dict(zip(region_clean['m49_code'], region_clean[c_ssp].astype(str))) if c_ssp else {}
    # M49 code map (country name -> m49)
    m49_by_country = dict(zip(region2['country_label'], region2['m49_code']))

    # Processes & meta
    process_meta = {}
    processes: List[str] = []
    if not emis_proc.empty and 'Process' in emis_proc.columns:
        c_proc = _find_col(emis_proc, ['Process'])
        processes = emis_proc[c_proc].astype(str).str.strip().dropna().unique().tolist()
        c_cat = next((nm for nm in ['category','Category','Sector'] if nm in emis_proc.columns), None)
        c_gas = next((nm for nm in ['gas','GHG','Gas'] if nm in emis_proc.columns), None)
        for r in emis_proc.itertuples(index=False):
            name = getattr(r, _tuple_field(c_proc))
            if pd.isna(name):
                continue
            proc_name = str(name).strip()
            if not proc_name:
                continue
            meta_entry = process_meta.setdefault(proc_name, {})
            if c_cat:
                meta_entry['category'] = str(getattr(r, c_cat))
            if c_gas:
                meta_entry['gas'] = str(getattr(r, c_gas))
    # Fallback to Emis_item if process list or meta missing
    c_proc_item = _find_col(emis_item, ['Process'])
    if not processes:
        processes = emis_item[c_proc_item].astype(str).str.strip().dropna().unique().tolist()
    c_gas_item = _find_col(emis_item, ['GHG']) if 'GHG' in emis_item.columns else None
    c_source_item = 'Emis_file_source' if 'Emis_file_source' in emis_item.columns else None
    proc_field = _tuple_field(c_proc_item)
    if c_gas_item or c_source_item:
        for r in emis_item.itertuples(index=False):
            proc_name = str(getattr(r, proc_field)).strip()
            if not proc_name:
                continue
            meta_entry = process_meta.setdefault(proc_name, {})
            if c_gas_item:
                gas_val = getattr(r, _tuple_field(c_gas_item))
                if not pd.isna(gas_val):
                    meta_entry.setdefault('gas', str(gas_val))
            if c_source_item:
                src_val = getattr(r, _tuple_field(c_source_item))
                if not pd.isna(src_val):
                    meta_entry.setdefault('file_source', str(src_val))

    # Commodities - Use Item_Emis for standardized model commodity names
    # Item_Production_Map is FAOSTAT CSV item name; Item_Emis is the model standard name
    c_item_emis = _find_col(emis_item, ['Item_Emis'])
    commodities = sorted(pd.unique(emis_item[c_item_emis].astype(str).str.strip()))

    # Cat2 mapping - use Item_Emis for consistency with commodities
    c_cat2 = _find_col(emis_item, ['Item_Cat2'])
    item_cat2_by_commodity = {}
    c_item_emis_attr = _tuple_field(c_item_emis)
    c_cat2_attr = _tuple_field(c_cat2)
    for r in emis_item.itertuples(index=False):
        item = str(getattr(r, c_item_emis_attr)).strip()
        cat2_val = getattr(r, c_cat2_attr)
        item_cat2_by_commodity[item] = '' if pd.isna(cat2_val) else str(cat2_val)

    # years
    years_hist = list(range(config.years_hist_start, config.years_hist_end + 1))
    years_future = config.years_future if config.years_future else []
    years = sorted(set(years_hist + years_future))

    return Universe(countries=countries, iso3_by_country=iso3_map, m49_by_country=m49_by_country,
                    commodities=commodities, years=years,
                    processes=processes, process_meta=process_meta,
                    region_aggMC_by_country=region_aggMC_by_country,
                    item_cat2_by_commodity=item_cat2_by_commodity,
                    ssp_region_by_country=ssp_region_by_country)

# nodes skeleton
def make_nodes_skeleton(universe: Universe) -> List[Node]:
    nodes: List[Node] = []
    # simple cartesian for skeleton (production/demand later fill)
    for i in universe.countries:
        iso3 = universe.iso3_by_country.get(i, '')
        country_name = universe.country_by_m49.get(i, '')
        for t in universe.years:
            for j in universe.commodities:
                nodes.append(Node(country=i, iso3=iso3, year=t, commodity=j,
                                  country_name=country_name, m49=i))
    return nodes

# elasticities
def apply_supply_ty_elasticity(nodes: List[Node], elasticity_xlsx: str) -> None:
    # optional; if not present, skip
    if not os.path.exists(elasticity_xlsx):
        return
    sheet_names = set(_excel_sheet_names(elasticity_xlsx))
    emis_to_elast = _build_emis_to_elast_map()
    def read(sheet: str) -> pd.DataFrame:
        return _lc(_read_excel(elasticity_xlsx, sheet_name=sheet)) if sheet in sheet_names else pd.DataFrame()

    temp_map = _build_elasticity_map(read('Supply-Temperature'))
    yield_map = _build_elasticity_map(read('Supply-Yield'))
    own_price_map = _build_elasticity_map(read('Supply-Own-Price'))
    commodity_filter = {str(n.commodity) for n in nodes}
    supply_cross_map = _build_cross_elasticity_map(read('Supply_Cross_mean'), commodity_filter=commodity_filter)

    for n in nodes:
        elast_key = emis_to_elast.get(str(n.commodity), str(n.commodity))
        key = (str(n.country), elast_key)
        temp = temp_map.get(key)
        if temp is not None:
            n.eps_supply_temp = float(temp)
        eta_y = yield_map.get(key)
        if eta_y is not None:
            n.eps_supply_yield = float(eta_y)
        own = own_price_map.get(key)
        if own is not None:
            n.eps_supply = float(own)
        cross = supply_cross_map.get(key)
        if cross:
            n.meta['supply_cross'] = dict(cross)
            setattr(n, 'epsS_row', dict(cross))

# FBS demand
FBS_TO_UNIVERSE: Dict[str, List[str]] = {
    # Cereals & grains
    'maize': ['Maize (corn)'],
    'maize and products': ['Maize (corn)'],
    'maize germ oil': ['Maize (corn)'],
    'wheat': ['Wheat'],
    'wheat and products': ['Wheat'],
    'rice': ['Rice'],
    'rice and products': ['Rice'],
    'ricebran oil': ['Rice'],
    'barley and products': ['Barley'],
    'oats': ['Oats'],
    'millet and products': ['Millet'],
    'sorghum and products': ['Sorghum'],
    'rye and products': ['Rye'],
    # Roots & tubers
    'cassava and products': ['Cassava, fresh'],
    'sweet potatoes': ['Sweet potatoes'],
    'potatoes and products': ['Potatoes'],
    'yams': ['Cassava, fresh'],
    'roots, other': ['Cassava, fresh'],
    'starchy roots': ['Cassava, fresh', 'Potatoes', 'Sweet potatoes'],
    # Oilcrops & oils
    'soyabeans': ['Soya beans'],
    'soyabean oil': ['Oilcrops, Oil Equivalent'],
    'sunflowerseed': ['Sunflower seed'],
    'sunflowerseed oil': ['Sunflower seed'],
    'rape and mustardseed': ['Rape or colza seed'],
    'rape and mustard oil': ['Rape or colza seed'],
    'groundnuts': ['Groundnuts, excluding shelled'],
    'groundnut oil': ['Groundnuts, excluding shelled'],
    'cottonseed': ['Seed cotton, unginned'],
    'cottonseed oil': ['Seed cotton, unginned'],
    'palm oil': ['Oilcrops, Oil Equivalent'],
    'palm kernels': ['Oilcrops, Oil Equivalent'],
    'palmkernel oil': ['Oilcrops, Oil Equivalent'],
    'coconut oil': ['Oilcrops, Oil Equivalent'],
    'coconuts - incl copra': ['Oilcrops, Oil Equivalent'],
    'olive oil': ['Oilcrops, Oil Equivalent'],
    'oilcrops oil, other': ['Oilcrops, Oil Equivalent'],
    'oilcrops': ['Oilcrops, Oil Equivalent'],
    'vegetable oils': ['Oilcrops, Oil Equivalent'],
    'sesame seed': ['Oilcrops, Oil Equivalent'],
    'sesameseed oil': ['Oilcrops, Oil Equivalent'],
    'nuts and products': ['Oilcrops, Oil Equivalent'],
    'treenuts': ['Oilcrops, Oil Equivalent'],
    # Fruits
    'bananas': ['Fruit Primary'],
    'apples and products': ['Fruit Primary'],
    'plantains': ['Fruit Primary'],
    'pineapples and products': ['Fruit Primary'],
    'dates': ['Fruit Primary'],
    'grapes and products (excl wine)': ['Fruit Primary'],
    'grapefruit and products': ['Fruit Primary'],
    'oranges, mandarines': ['Fruit Primary'],
    'lemons, limes and products': ['Fruit Primary'],
    'citrus, other': ['Fruit Primary'],
    'fruits - excluding wine': ['Fruit Primary'],
    'fruits, other': ['Fruit Primary'],
    'olives (including preserved)': ['Fruit Primary'],
    # Vegetables
    'tomatoes and products': ['Vegetables Primary'],
    'onions': ['Vegetables Primary'],
    'vegetables': ['Vegetables Primary'],
    'vegetables, other': ['Vegetables Primary'],
    # Pulses & beans
    'beans': ['Beans, dry'],
    'pulses': ['Beans, dry'],
    'pulses, other and products': ['Beans, dry'],
    'peas': ['Beans, dry'],
    # Sugar
    'sugar crops': ['Sugar cane', 'Sugar beet'],
    'sugar (raw equivalent)': ['Sugar cane', 'Sugar beet'],
    'sugar & sweeteners': ['Sugar cane', 'Sugar beet'],
    'sweeteners, other': ['Sugar cane', 'Sugar beet'],
    'sugar non-centrifugal': ['Sugar cane'],
    # Animal products
    'milk - excluding butter': ['Raw milk of cattle'],
    'butter, ghee': ['Raw milk of cattle'],
    'fats, animals, raw': ['Raw milk of cattle'],
    'cream': ['Raw milk of cattle'],
    'animal fats': ['Raw milk of cattle'],
    'eggs': ['Eggs Primary'],
    'bovine meat': ['Meat of cattle with the bone, fresh or chilled', 'Meat of buffalo, fresh or chilled'],
    'pigmeat': ['Meat of pig with the bone, fresh or chilled'],
    'poultry meat': ['Meat of chickens, fresh or chilled', 'Meat of turkeys, fresh or chilled', 'Meat of ducks, fresh or chilled'],
    'mutton & goat meat': ['Meat of sheep, fresh or chilled', 'Meat of goat, fresh or chilled'],
    'meat, other': [
        'Horse meat, fresh or chilled',
        'Meat of asses, fresh or chilled',
        'Meat of mules, fresh or chilled',
        'Meat of camels, fresh or chilled',
        'Meat of other domestic camelids, fresh or chilled'
    ],
    # Fish and aquatic products
    'demersal fish': ['Fish, Seafood'],
    'pelagic fish': ['Fish, Seafood'],
    'marine fish, other': ['Fish, Seafood'],
    'freshwater fish': ['Fish, Seafood'],
    'cephalopods': ['Fish, Seafood'],
    'crustaceans': ['Fish, Seafood'],
    'molluscs, other': ['Fish, Seafood'],
    'aquatic animals, others': ['Fish, Seafood'],
    'aquatic plants': ['Fish, Seafood'],
    'aquatic products, other': ['Fish, Seafood'],
    'fish, body oil': ['Fish, Seafood'],
    'fish, liver oil': ['Fish, Seafood'],
    'meat, aquatic mammals': ['Fish, Seafood'],
}

FBS_DEMAND_COMPONENT_COLUMNS = [
    'country', 'iso3', 'year', 'commodity',
    'food_t', 'feed_t', 'seed_t',
    'processing_t', 'losses_t', 'other_uses_t',
    'tourist_consumption_t', 'stock_variation_t',
    'demand_total_t', 'fbs_accounted_total_t',
]


def build_demand_components_from_fbs(fbs_csv: str,
                                     universe: Universe,
                                     *,
                                     production_lookup: Optional[Dict[Tuple[str, str, int], float]] = None,
                                     latest_hist_prod: Optional[Dict[Tuple[str, str], Tuple[int, float]]] = None,
                                     feed_override_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if not os.path.exists(fbs_csv):
        return pd.DataFrame(columns=FBS_DEMAND_COMPONENT_COLUMNS)
    df_raw = _read_fbs_table(fbs_csv)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val  = _find_col(df, ['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])
    keep_cols = [c_area, c_year, c_item, c_elem, c_val] + ([c_unit] if c_unit else [])
    z = df[keep_cols].copy()
    rename_map = {c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'}
    if c_unit:
        rename_map[c_unit] = 'unit'
    z = z.rename(columns=rename_map)
    if 'M49_Country_Code' in df.columns:
        z['M49_Country_Code'] = df['M49_Country_Code']
    z = _attach_country_from_m49(df, z, universe, context=f"FBS demand ({fbs_csv})")
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df['M49_Country_Code']
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.production_by_item).fillna(z['item_raw'])
    except Exception:
        z['commodity'] = z['item_raw']
    z = z[z['country'].isin(universe.countries)]
    if z.empty:
        return pd.DataFrame(columns=FBS_DEMAND_COMPONENT_COLUMNS)

    if 'unit' in z.columns:
        factors = z['unit'].astype(str).str.contains('1000', case=False, na=False).replace({True: 1000.0, False: 1.0})
        z['value'] = pd.to_numeric(z['value'], errors='coerce').fillna(0.0) * factors
    pivot = z.pivot_table(index=['country', 'year', 'commodity'],
                          columns='element', values='value', aggfunc='sum').reset_index()
    pivot.rename(columns={c: c.lower() for c in pivot.columns}, inplace=True)
    food = pivot.get('food', pd.Series(0, index=pivot.index))
    feed = pivot.get('feed', pd.Series(0, index=pivot.index))
    seed = pivot.get('seed', pd.Series(0, index=pivot.index))
    processing = pivot.get('processing', pd.Series(0, index=pivot.index))
    losses = pivot.get('losses', pd.Series(0, index=pivot.index))
    other_uses = pivot.get('other uses (non-food)', pd.Series(0, index=pivot.index))
    tourist = pivot.get('tourist consumption', pd.Series(0, index=pivot.index))
    stock_variation = pivot.get('stock variation', pd.Series(0, index=pivot.index))
    base = pd.DataFrame({
        'country': pivot['country'],
        'year': pivot['year'],
        'commodity': pivot['commodity'],
        'food_t': food.fillna(0.0).astype(float),
        'feed_t': feed.fillna(0.0).astype(float),
        'seed_t': seed.fillna(0.0).astype(float),
        'processing_t': processing.fillna(0.0).astype(float),
        'losses_t': losses.fillna(0.0).astype(float),
        'other_uses_t': other_uses.fillna(0.0).astype(float),
        'tourist_consumption_t': tourist.fillna(0.0).astype(float),
        'stock_variation_t': stock_variation.fillna(0.0).astype(float),
    })

    universe_set = set(universe.commodities or [])
    prod_lookup = production_lookup or {}
    latest_prod = latest_hist_prod or {}
    def _norm_key(name: str) -> str:
        return str(name).strip().lower()

    def _resolve_targets(name: str) -> List[str]:
        if name in universe_set:
            return [name]
        key = _norm_key(name)
        if key in FBS_TO_UNIVERSE:
            return FBS_TO_UNIVERSE[key]
        if key.endswith(' and products'):
            base_key = key[:-len(' and products')].strip()
            if base_key in FBS_TO_UNIVERSE:
                return FBS_TO_UNIVERSE[base_key]
        if key.endswith('s') and key[:-1] in FBS_TO_UNIVERSE:
            return FBS_TO_UNIVERSE[key[:-1]]
        return []
    def _production_value(country: str, commodity: str, year: int) -> float:
        val = prod_lookup.get((country, commodity, year))
        if val is None and latest_prod:
            prev = latest_prod.get((country, commodity))
            if prev is not None:
                val = prev[1]
        return max(float(val), 0.0) if val is not None else 0.0

    rows: List[Dict[str, float]] = []
    for row in base.itertuples(index=False):
        targets = _resolve_targets(row.commodity)
        if not targets:
            continue

        if len(targets) == 1:
            weights = {targets[0]: 1.0}
        else:
            shares = [_production_value(row.country, tgt, int(row.year)) for tgt in targets]
            total = sum(shares)
            if total <= 0:
                weights = {tgt: 1.0 / len(targets) for tgt in targets}
            else:
                weights = {tgt: share / total for tgt, share in zip(targets, shares)}

        for tgt, weight in weights.items():
            if weight <= 0:
                continue
            rows.append({
                'country': row.country,
                'year': int(row.year),
                'commodity': tgt,
                'food_t': float(row.food_t) * weight,
                'feed_t': float(row.feed_t) * weight,
                'seed_t': float(row.seed_t) * weight,
                'processing_t': float(row.processing_t) * weight,
                'losses_t': float(row.losses_t) * weight,
                'other_uses_t': float(row.other_uses_t) * weight,
                'tourist_consumption_t': float(row.tourist_consumption_t) * weight,
                'stock_variation_t': float(row.stock_variation_t) * weight,
            })

    if not rows:
        return pd.DataFrame(columns=FBS_DEMAND_COMPONENT_COLUMNS)

    result = pd.DataFrame(rows)
    component_cols = [
        'food_t', 'feed_t', 'seed_t', 'processing_t', 'losses_t',
        'other_uses_t', 'tourist_consumption_t', 'stock_variation_t',
    ]
    result = result.groupby(['country', 'year', 'commodity'], as_index=False)[component_cols].sum()

    if feed_override_df is not None and not feed_override_df.empty:
        cols_needed = {'country', 'year', 'commodity', 'feed_t'}
        if cols_needed.issubset(feed_override_df.columns):
            override = feed_override_df[['country', 'year', 'commodity', 'feed_t']].copy()
            override = override.groupby(['country','year','commodity'], as_index=False)['feed_t'].sum()
            result = result.merge(override,
                                  how='outer',
                                  on=['country','year','commodity'],
                                  suffixes=('', '_override'))
            for col in [c for c in component_cols if c != 'feed_t']:
                if col in result.columns:
                    result[col] = result[col].fillna(0.0)
            result['feed_t'] = result['feed_t_override'].fillna(result['feed_t']).fillna(0.0)
            result = result.drop(columns=['feed_t_override'])
        else:
            pass  # silently ignore malformed overrides to avoid crashing

    result['iso3'] = result['country'].map(universe.iso3_by_country)
    # Keep the historical meaning of demand_total_t for backward compatibility.
    result['demand_total_t'] = result['food_t'] + result['feed_t'] + result['seed_t']
    result['fbs_accounted_total_t'] = result[component_cols].sum(axis=1)
    return result[FBS_DEMAND_COMPONENT_COLUMNS]


def _load_demand_item_map(dict_v3_path: Optional[str]) -> Dict[str, str]:
    if dict_v3_path is None:
        try:
            dict_v3_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        except Exception:
            return {}
    if not os.path.exists(dict_v3_path):
        return {}
    try:
        df = _read_excel(dict_v3_path, sheet_name='Emis_item')
    except Exception:
        return {}
    if 'Item_Emis' not in df.columns or 'Item_Demand_Map' not in df.columns:
        return {}
    mapping: Dict[str, str] = {}
    conflicts: Dict[str, set] = {}
    for _, row in df[['Item_Emis', 'Item_Demand_Map']].dropna().iterrows():
        comm = str(row['Item_Emis']).strip()
        item = str(row['Item_Demand_Map']).strip()
        if not comm or not item or item.lower() in {'no', 'nan'}:
            continue
        if item in mapping and mapping[item] != comm:
            conflicts.setdefault(item, set()).update({mapping[item], comm})
            continue
        mapping[item] = comm
    if conflicts:
        logger = logging.getLogger(__name__)
        sample = sorted(conflicts.items())[:5]
        logger.warning(f"[FBS-DSQ] Item_Demand_Map 冲突 {len(conflicts)} 条，示例: {sample}")
    return mapping


def build_demand_total_from_fbs_domestic_supply(fbs_csv: str,
                                                dict_v3_path: str,
                                                universe: Universe) -> pd.DataFrame:
    if not os.path.exists(fbs_csv):
        return pd.DataFrame(columns=['country','iso3','year','commodity','demand_total_t'])
    item_map = _load_demand_item_map(dict_v3_path)
    if not item_map:
        return pd.DataFrame(columns=['country','iso3','year','commodity','demand_total_t'])
    df_raw = _read_fbs_table(fbs_csv)
    df_raw = _filter_select_rows(df_raw)
    raw_item_col = _find_col(df_raw, ['Item'])
    raw_element_col = _find_col(df_raw, ['Element'])
    item_mask = df_raw[raw_item_col].astype(str).str.strip().isin(item_map)
    element_mask = (
        df_raw[raw_element_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq('domestic supply quantity')
    )
    df_raw = df_raw.loc[item_mask & element_mask].copy()
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val = _find_col(df, ['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])
    keep_cols = [c_area, c_year, c_item, c_elem, c_val]
    if c_unit:
        keep_cols.append(c_unit)
    z = df[keep_cols].copy()
    rename_map = {c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'}
    if c_unit:
        rename_map[c_unit] = 'unit'
    z = z.rename(columns=rename_map)
    if 'M49_Country_Code' in df.columns:
        z['M49_Country_Code'] = df['M49_Country_Code']
    z = _attach_country_from_m49(df, z, universe, context=f"FBS demand DSQ ({fbs_csv})")

    z['item_raw'] = z['item_raw'].astype(str).str.strip()
    z['commodity'] = z['item_raw'].map(item_map)
    z = z.dropna(subset=['commodity'])
    z = z[z['element'].astype(str).str.strip().str.lower() == 'domestic supply quantity']
    if z.empty:
        return pd.DataFrame(columns=['country','iso3','year','commodity','demand_total_t'])
    z['value'] = pd.to_numeric(z['value'], errors='coerce').fillna(0.0)
    if 'unit' in z.columns:
        factors = z['unit'].astype(str).str.contains('1000', case=False, na=False).replace({True: 1000.0, False: 1.0})
        z['value'] = z['value'] * factors
    z['year'] = pd.to_numeric(z['year'], errors='coerce').astype(int)
    result = z.groupby(['country', 'year', 'commodity'], as_index=False)['value'].sum()
    result.rename(columns={'value': 'demand_total_t'}, inplace=True)
    result['iso3'] = result['country'].map(universe.iso3_by_country)
    return result[['country','iso3','year','commodity','demand_total_t']]


def apply_demand_total_to_nodes(nodes: List[Node], demand_df: pd.DataFrame) -> None:
    if demand_df is None or len(demand_df) == 0:
        return
    key = {(r.country, r.commodity, int(r.year)): float(r.demand_total_t) for r in demand_df.itertuples(index=False)}
    for n in nodes:
        val = key.get((n.country, n.commodity, int(n.year)))
        if val is not None:
            n.D0 = float(val)

def _parse_m49_code(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    m49 = _norm_m49(value)
    return m49 or None

def _attach_country_column(df: pd.DataFrame,
                           m49_to_country: Dict[str, str],
                           area_lower_to_country: Dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['country'])
    if 'M49_Country_Code' not in df.columns:
        raise ValueError("Expected 'M49_Country_Code' column for all FAOSTAT tables.")
    codes = df['M49_Country_Code'].apply(_norm_m49)
    m49_map = { _norm_m49(k): str(v) for k, v in (m49_to_country or {}).items() }
    df = df.assign(country=codes)
    df['country_name'] = df['country'].map(m49_map)
    df = df.dropna(subset=['country'])
    df['country'] = df['country'].astype(str)
    return df

def _melt_trade_quantities(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['country','Item','year','Element','value'])
    year_cols = [c for c in df.columns if isinstance(c, str) and c.strip().startswith('Y') and c.strip()[1:].isdigit()]
    if not year_cols:
        return pd.DataFrame(columns=['country','Item','year','Element','value'])
    df_year = df[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df = df.copy()
    df[year_cols] = df_year
    long_df = df.melt(id_vars=[c for c in df.columns if c not in year_cols],
                      value_vars=year_cols,
                      var_name='year',
                      value_name='value')
    long_df['year'] = pd.to_numeric(long_df['year'].astype(str).str.strip().str.lstrip('Y'), errors='coerce')
    long_df = long_df.dropna(subset=['year'])
    long_df['year'] = long_df['year'].astype(int)
    long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce').fillna(0.0)
    return long_df

def _load_trade_cropslivestock(path: str,
                               items: List[str],
                               m49_to_country: Dict[str, str],
                               area_lower_to_country: Dict[str, str]) -> pd.DataFrame:
    if not items or not os.path.exists(path):
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = _lc(_read_excel(path))
    df = df[df['Item'].astype(str).str.strip().isin(items)]
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = df[df['Element'].astype(str).str.lower().isin({'import quantity','export quantity'})]
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = _attach_country_column(df, m49_to_country, area_lower_to_country)
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df['Item'] = df['Item'].astype(str).str.strip()
    long_df = _melt_trade_quantities(df)
    if long_df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    long_df['Element'] = long_df['Element'].astype(str).str.lower()
    agg = long_df.groupby(['country','Item','year','Element'], as_index=False)['value'].sum()
    piv = agg.pivot_table(index=['country','Item','year'], columns='Element', values='value', aggfunc='sum', fill_value=0.0).reset_index()
    import_col = 'import quantity'
    export_col = 'export quantity'
    if import_col not in piv.columns:
        piv[import_col] = 0.0
    if export_col not in piv.columns:
        piv[export_col] = 0.0
    piv.rename(columns={import_col: 'import_t', export_col: 'export_t'}, inplace=True)
    return piv[['country','Item','year','import_t','export_t']]

def _load_trade_forestry(path: str,
                         items: List[str],
                         m49_to_country: Dict[str, str],
                         area_lower_to_country: Dict[str, str]) -> pd.DataFrame:
    if not items or not os.path.exists(path):
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = _lc(_read_csv(path))
    df = df[df['Item'].astype(str).str.strip().isin(items)]
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = df[df['Element'].astype(str).str.lower().isin({'import quantity','export quantity'})]
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = _attach_country_column(df, m49_to_country, area_lower_to_country)
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df['Item'] = df['Item'].astype(str).str.strip()
    long_df = _melt_trade_quantities(df)
    if long_df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    long_df['Element'] = long_df['Element'].astype(str).str.lower()
    agg = long_df.groupby(['country','Item','year','Element'], as_index=False)['value'].sum()
    piv = agg.pivot_table(index=['country','Item','year'], columns='Element', values='value', aggfunc='sum', fill_value=0.0).reset_index()
    import_col = 'import quantity'
    export_col = 'export quantity'
    if import_col not in piv.columns:
        piv[import_col] = 0.0
    if export_col not in piv.columns:
        piv[export_col] = 0.0
    piv.rename(columns={import_col: 'import_t', export_col: 'export_t'}, inplace=True)
    return piv[['country','Item','year','import_t','export_t']]

def _load_trade_fbs(path: str,
                    items: List[str],
                    m49_to_country: Dict[str, str],
                    area_lower_to_country: Dict[str, str]) -> pd.DataFrame:
    if not items or not os.path.exists(path):
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    base_cols = ['M49_Country_Code', 'Area', 'Item', 'Element', 'Unit']
    def usecols(col: str) -> bool:
        return col in base_cols or (isinstance(col, str) and col.startswith('Y') and col[1:].isdigit())
    frames: List[pd.DataFrame] = []
    if str(path).lower().endswith(('.xlsx', '.xls')):
        full_table = _read_fbs_table(path)
        selected_cols = [col for col in full_table.columns if usecols(col)]
        data_frames = [full_table[selected_cols].copy()]
    else:
        data_frames = _read_csv(path, usecols=usecols, chunksize=200000)
    for chunk in data_frames:
        chunk = _lc(chunk)
        chunk['Item'] = chunk['Item'].astype(str).str.strip()
        mask_item = chunk['Item'].isin(items)
        if not mask_item.any():
            continue
        chunk = chunk.loc[mask_item]
        chunk['Element'] = chunk['Element'].astype(str).str.lower()
        chunk = chunk[chunk['Element'].isin({'import quantity','export quantity'})]
        if chunk.empty:
            continue
        chunk = _attach_country_column(chunk, m49_to_country, area_lower_to_country)
        if chunk.empty:
            continue
        year_cols = [c for c in chunk.columns if c.startswith('Y') and c[1:].isdigit()]
        if not year_cols:
            continue
        values = chunk[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        factors = chunk['Unit'].astype(str).str.contains('1000', case=False, na=False).replace({True: 1000.0, False: 1.0}).to_numpy()
        values = values.mul(factors.reshape(-1, 1))
        chunk = chunk.drop(columns=year_cols)
        chunk[year_cols] = values
        long_df = chunk.melt(id_vars=[c for c in chunk.columns if c not in year_cols],
                             value_vars=year_cols,
                             var_name='year',
                             value_name='value')
        long_df['year'] = pd.to_numeric(long_df['year'].astype(str).str.strip().str.lstrip('Y'), errors='coerce')
        long_df = long_df.dropna(subset=['year'])
        long_df['year'] = long_df['year'].astype(int)
        long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce').fillna(0.0)
        frames.append(long_df[['country','Item','year','Element','value']])
    if not frames:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    long_all = pd.concat(frames, ignore_index=True)
    agg = long_all.groupby(['country','Item','year','Element'], as_index=False)['value'].sum()
    piv = agg.pivot_table(index=['country','Item','year'], columns='Element', values='value', aggfunc='sum', fill_value=0.0).reset_index()
    import_col = 'import quantity'
    export_col = 'export quantity'
    if import_col not in piv.columns:
        piv[import_col] = 0.0
    if export_col not in piv.columns:
        piv[export_col] = 0.0
    piv.rename(columns={import_col: 'import_t', export_col: 'export_t'}, inplace=True)
    return piv[['country','Item','year','import_t','export_t']]

def load_trade_import_export(paths: DataPaths, universe: Universe) -> Tuple[Dict[Tuple[str, str, int], float], Dict[Tuple[str, str, int], float]]:
    try:
        emis_df = _lc(_read_excel(paths.dict_v3_path, 'Emis_item'))
    except Exception:
        return {}, {}
    if 'Item_Trade_Map' not in emis_df.columns or 'Trade_file_source' not in emis_df.columns or 'Item_Emis' not in emis_df.columns:
        return {}, {}
    trade_attr = _tuple_field('Item_Trade_Map')
    trade_src_attr = _tuple_field('Trade_file_source')
    emis_attr = _tuple_field('Item_Emis')
    universe_commodities = set(universe.commodities or [])
    trade_mapping: Dict[str, set] = defaultdict(set)
    items_by_source: Dict[str, set] = defaultdict(set)
    for row in emis_df.itertuples(index=False):
        emis_val = getattr(row, emis_attr, None)
        trade_val = getattr(row, trade_attr, None)
        source_val = getattr(row, trade_src_attr, None)
        if pd.isna(emis_val) or pd.isna(trade_val) or pd.isna(source_val):
            continue
        item_name = str(emis_val).strip()
        if not item_name or item_name.lower() in {'nan', 'no'}:
            continue
        if item_name not in universe_commodities:
            continue
        trade_items = [item.strip() for item in str(trade_val).split(';') if item and str(item).strip().lower() not in {'nan','no'}]
        if not trade_items:
            continue
        sources = [src.strip() for src in str(source_val).split(';') if src and str(src).strip().lower() not in {'nan','no'}]
        if not sources:
            continue
        if len(sources) == 1 and len(trade_items) > 1:
            sources = sources * len(trade_items)
        elif len(sources) < len(trade_items):
            sources = sources + [sources[-1]] * (len(trade_items) - len(sources))
        for item, src in zip(trade_items, sources):
            trade_mapping[item_name].add((item, src))
            items_by_source[src].add(item)
    if not trade_mapping:
        return {}, {}

    m49_to_country: Dict[str, str] = { _norm_m49(k): str(v) for k, v in (universe.country_by_m49 or {}).items() }
    area_lower_to_country = {str(cty).strip().lower(): cty for cty in universe.countries}

    source_map = {
        'Trade_CropsLivestock_E_All_Data_NOFLAG_filtered.xlsx': paths.trade_crops_xlsx,
        'Forestry_E_All_Data_NOFLAG.csv': paths.trade_forestry_csv,
        'FoodBalanceSheets_E_All_Data_NOFLAG.csv': paths.fbs_csv,
    }

    data_by_source: Dict[str, pd.DataFrame] = {}
    for source_name, items in items_by_source.items():
        path = source_map.get(source_name)
        item_list = sorted({str(it).strip() for it in items if str(it).strip()})
        if not path or not item_list:
            continue
        if source_name.endswith('.xlsx') and 'Trade_CropsLivestock' in source_name:
            table = _load_trade_cropslivestock(path, item_list, m49_to_country, area_lower_to_country)
        elif source_name.endswith('.csv') and 'Forestry' in source_name:
            table = _load_trade_forestry(path, item_list, m49_to_country, area_lower_to_country)
        elif source_name.endswith('.csv') and 'FoodBalance' in source_name:
            table = _load_trade_fbs(path, item_list, m49_to_country, area_lower_to_country)
        else:
            table = pd.DataFrame(columns=['country','Item','year','import','export'])
        if table is not None and not table.empty:
            table['Item'] = table['Item'].astype(str).str.strip()
            data_by_source[source_name] = table

    imports_by = defaultdict(float)
    exports_by = defaultdict(float)
    for prod_name, pairs in trade_mapping.items():
        for trade_item, source_name in pairs:
            table = data_by_source.get(source_name)
            if table is None or table.empty:
                continue
            subset = table[table['Item'] == trade_item]
            if subset.empty:
                continue
            for record in subset.itertuples(index=False):
                key = (record.country, prod_name, int(record.year))
                imports_by[key] += float(getattr(record, 'import_t', 0.0) or 0.0)
                exports_by[key] += float(getattr(record, 'export_t', 0.0) or 0.0)

    return dict(imports_by), dict(exports_by)

def apply_fbs_components_to_nodes(nodes: List[Node], fbs_components: pd.DataFrame, feed_efficiency: float=1.0) -> None:
    if fbs_components is None or len(fbs_components)==0:
        return
    # index for quick lookup
    key = {(r.country, r.commodity, int(r.year)): (float(r.food_t), float(r.feed_t), float(r.seed_t)) for r in fbs_components.itertuples(index=False)}
    for n in nodes:
        tpl = key.get((n.country, n.commodity, n.year))
        if tpl:
            food, feed, seed = tpl
            feed_eff = float(feed_efficiency) if feed_efficiency>0 else 1.0
            n.D0 = float(food + feed/feed_eff + seed)

# production & activities
def build_production_from_faostat(csv_path: str,
                                  universe: Universe,
                                  fbs_csv: Optional[str] = None) -> pd.DataFrame:
    maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
    data = build_faostat_production_indicators(csv_path, universe, maps, fbs_csv=fbs_csv)
    return data['production']


def load_production_indicators(paths: DataPaths, universe: Universe) -> Dict[str, pd.DataFrame]:
    maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
    return build_faostat_production_indicators(paths.production_faostat_csv, universe, maps)


def load_production_statistics(paths: DataPaths,
                               universe: Universe,
                               feed_requirement_scheme: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Load production statistics, including crop and livestock parameters."""
    
    # Check critical inputs.
    dict_v3_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
    _check_file_exists(dict_v3_path, "字典文件 (dict_v3.xlsx)", critical=True)
    _check_file_exists(paths.production_faostat_csv, "生产数据文件 (Production_Crops_Livestock)", critical=True)
    _check_file_exists(paths.manure_stock_with_ratio_csv, "畜牧存栏数据文件 (Environment_LivestockManure_with_ratio)", critical=True)
    _check_file_exists(paths.feed_coeff_xlsx, "饲料需求系数文件 (Feed_need_per_head)", critical=True)
    
    maps = load_emis_item_mappings(dict_v3_path)
    stats = build_faostat_production_indicators(
        paths.production_faostat_csv,
        universe,
        maps,
        fbs_csv=paths.fbs_csv
    )
    # Attach fish aquaculture yield (Fish, Seafood) from panel if available
    fish_panel_path = os.path.join(get_input_base(), "Aquaculture", "fish_seafood_country_panel_2000_present.xlsx")
    if os.path.exists(fish_panel_path):
        try:
            panel = _read_excel(fish_panel_path, sheet_name="country_year_panel")
            panel.columns = [str(c).strip() for c in panel.columns]
            if "year" in panel.columns and "Year" not in panel.columns:
                panel = panel.rename(columns={"year": "Year"})
            req_cols = ["M49_Country_Code", "Year", "Aquaculture_yield_t_per_ha"]
            missing = [c for c in req_cols if c not in panel.columns]
            if not missing:
                fish_yield = panel[req_cols].copy()
                fish_yield["M49_Country_Code"] = fish_yield["M49_Country_Code"].apply(_norm_m49)
                fish_yield["year"] = pd.to_numeric(fish_yield["Year"], errors="coerce")
                fish_yield["yield_t_per_ha"] = pd.to_numeric(fish_yield["Aquaculture_yield_t_per_ha"], errors="coerce")
                fish_yield = fish_yield.dropna(subset=["M49_Country_Code", "year"])
                fish_yield["year"] = fish_yield["year"].astype(int)
                fish_yield.loc[fish_yield["yield_t_per_ha"] <= 0, "yield_t_per_ha"] = np.nan
                if fish_yield["yield_t_per_ha"].isna().any():
                    global_mean = fish_yield["yield_t_per_ha"].mean(skipna=True)
                    if pd.notna(global_mean):
                        fish_yield["yield_t_per_ha"] = fish_yield["yield_t_per_ha"].fillna(global_mean)
                fish_yield["country"] = fish_yield["M49_Country_Code"]
                fish_yield["country_name"] = fish_yield["country"].map(universe.country_by_m49)
                fish_yield["iso3"] = fish_yield["country"].map(universe.iso3_by_country)
                fish_yield["commodity"] = "Fish, Seafood"
                fish_yield = fish_yield.dropna(subset=["country", "iso3"])
                fish_yield = fish_yield[["M49_Country_Code", "country", "iso3", "year", "commodity", "yield_t_per_ha"]]
                if isinstance(stats.get("yield"), pd.DataFrame) and not stats["yield"].empty:
                    stats["yield"] = stats["yield"][stats["yield"]["commodity"].astype(str).str.lower() != "fish, seafood"].copy()
                stats["yield"] = pd.concat([stats.get("yield", pd.DataFrame()), fish_yield], ignore_index=True)
        except Exception as e:
            print(f"[WARNING] Fish aquaculture yield load failed: {e}")
    else:
        print(f"[WARNING] Fish panel not found: {fish_panel_path}")
    # WARNING: production must be extended to future years; otherwise 2080 livestock emissions cannot be computed
    stats['production'] = _extend_future_years(stats['production'], 'production_t', universe)
    stats['yield'] = _extend_future_years(stats['yield'], 'yield_t_per_ha', universe)
    stats['slaughter'] = _extend_future_years(stats['slaughter'], 'slaughter_head', universe)
    stats['livestock_yield'] = _extend_future_years(stats['livestock_yield'], 'yield_t_per_head', universe)
    
    # Load livestock stock from Environment_LivestockManure_with_ratio.csv
    print(f"[INFO] 从 Environment_LivestockManure_with_ratio.csv 加载livestock stock数据...")
    stock_from_env = build_livestock_stock_from_env(paths.manure_stock_with_ratio_csv, universe)
    if not stock_from_env.empty:
        print(f"[INFO] OK 从Environment文件加载了 {len(stock_from_env)} 行stock数据")
        stats['stock_env_raw'] = stock_from_env.copy()
        # Use stock from Environment file (more complete for livestock)
        stats['stock'] = stock_from_env
    else:
        print(f"[WARNING] WARN Environment文件未找到stock数据，使用Production CSV的stock")
        stats['stock_env_raw'] = pd.DataFrame(columns=['country','iso3','year','commodity','stock_head'])
        stats['stock'] = _extend_future_years(stats['stock'], 'stock_head', universe)
    
    # Extend stock to future years
    stats['stock'] = _extend_future_years(stats['stock'], 'stock_head', universe)
    
    fert = load_fertilizer_statistics(paths.fertilizer_efficiency_xlsx, universe, maps)
    feed = load_feed_requirement_per_head(
        paths.feed_coeff_xlsx,
        universe,
        maps,
        feed_requirement_scheme=feed_requirement_scheme,
    )
    # WARNING: use the correct manure file path
    manure = load_manure_management_ratio(paths.manure_stock_with_ratio_csv, universe, maps)
    stats.update({
        'fertilizer_efficiency': fert['efficiency'],
        'fertilizer_amount': fert['amount'],
        'feed_requirement': feed,
        'manure_management_ratio': manure,
    })
    return stats

def build_gce_activity_tables(production_csv: str,
                              fbs_csv: str,
                              fertilizer_eff_xlsx: str,
                              universe: Universe) -> Dict[str, pd.DataFrame]:
    # For now, return minimal frames with keys expected by orchestrator
    prod = build_production_from_faostat(production_csv, universe)
    residues_df = prod.rename(columns={'production_t':'residues_feedstock_t'})[['country','iso3','year','commodity','residues_feedstock_t']]
    burning_df = prod.rename(columns={'production_t':'burning_feedstock_t'})[['country','iso3','year','commodity','burning_feedstock_t']]
    rice_df = prod.rename(columns={'production_t':'rice_area_proxy'})[['country','iso3','year','commodity','rice_area_proxy']]
    fertilizers_df = pd.DataFrame(columns=['country','iso3','year','n_fert_t'])
    if os.path.exists(fertilizer_eff_xlsx):
        try:
            maps = load_emis_item_mappings(os.path.join(get_src_base(), "dict_v3.xlsx"))
            fert_stats = load_fertilizer_statistics(fertilizer_eff_xlsx, universe, maps)
            amt_df = fert_stats.get('amount', pd.DataFrame())
        except Exception:
            amt_df = pd.DataFrame()
    else:
        amt_df = pd.DataFrame()
    if isinstance(amt_df, pd.DataFrame) and not amt_df.empty:
        z = amt_df[['country','iso3','year','fertilizer_n_input_t']].copy()
        z['fertilizer_n_input_t'] = pd.to_numeric(z['fertilizer_n_input_t'], errors='coerce')
        z = z.dropna(subset=['fertilizer_n_input_t'])
        if not z.empty:
            z = z.groupby(['country','iso3','year'], as_index=False)['fertilizer_n_input_t'].sum()
            z = z.rename(columns={'fertilizer_n_input_t': 'n_fert_t'})
            z = z[z['year'].isin(universe.years)]
            fertilizers_df = z.sort_values(['country','year']).reset_index(drop=True)
    return {'residues_df': residues_df, 'burning_df': burning_df, 'rice_df': rice_df, 'fertilizers_df': fertilizers_df}

def build_livestock_stock_from_env(csv_path: str, universe: Universe) -> pd.DataFrame:
    """
    Load livestock stock from Environment_LivestockManure_with_ratio.csv
    WARNING: keep only commodities defined in dict_v3 (drop 'All Animals' and similar)
    Returns DataFrame with columns: country, iso3, year, commodity, stock_head
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','stock_head'])
    df_raw = _read_csv(csv_path)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    
    # Find column names
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year']) 
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val = _find_col(df, ['Value'])
    
    if not all([c_area, c_year, c_item, c_elem, c_val]):
        return pd.DataFrame(columns=['country','iso3','year','commodity','stock_head'])
    
    # Filter for "Stocks" element only
    df = df[df[c_elem].astype(str).str.strip() == 'Stocks'].copy()
    if df.empty:
        return pd.DataFrame(columns=['country','iso3','year','commodity','stock_head'])
    
    keep_cols = [c_area, c_year, c_item, c_val]
    if 'M49_Country_Code' in df.columns:
        keep_cols.append('M49_Country_Code')
    
    z = df[keep_cols].copy()
    z = z.rename(columns={c_area: 'area', c_year: 'year', c_item: 'item_raw', c_val: 'stock_head'})
    z['stock_head'] = pd.to_numeric(z['stock_head'], errors='coerce')
    z = z.dropna(subset=['stock_head'])
    
    # Attach country info
    z = _attach_country_from_m49(df, z, universe, context=f"Livestock stock ({csv_path})")
    # Preserve M49_Country_Code and ensure index alignment.
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df.loc[z.index, 'M49_Country_Code'].values
    
    # Map item to commodity using dict_v3
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.stock_item_to_comm)
        # B04: some stock rows already use the canonical Item_Emis name
        # (Swine, Llamas), while Item_Stock_Map contains a different alias.
        # Preserve those exact canonical matches; never map the parent herd
        # to both dairy and meat products.
        canonical_stock_items = set(maps.feed_item_to_comm.values())
        canonical = z['item_raw'].where(z['item_raw'].isin(canonical_stock_items))
        z['commodity'] = z['commodity'].fillna(canonical)
        
        # WARNING: keep only commodities that map to dict_v3; drop 'All Animals' and similar
        before_filter = len(z)
        z = z.dropna(subset=['commodity'])
        after_filter = len(z)
        if before_filter > after_filter:
            print(f"[INFO] 过滤掉 {before_filter - after_filter} 行非dict_v3定义的Item (如'All Animals')")
            
    except Exception as e:
        print(f"[WARNING] 无法加载dict_v3映射: {e}")
        z['commodity'] = z['item_raw']
    
    # Filter to the universe's 198 valid countries.
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    z = z.dropna(subset=['iso3'])
    z['year'] = pd.to_numeric(z['year'], errors='coerce').astype(int)
    z = z[z['year'].isin(universe.years)]
    
    # WARNING: note: stock_df uses Item_Stock_Map names (e.g., "Cattle, dairy")
    # universe.commodities uses Item_Production_Map names such as 'Raw milk of cattle'.
    # Do not filter with universe.commodities; maps.stock_item_to_comm already validated commodity names above.
    
    # Ensure M49_Country_Code exists in apostrophe-prefixed three-digit format.
    # Normalize M49 to an apostrophe plus three digits.
    def _format_m49_quote(val):
        """Format M49 as an apostrophe plus three digits."""
        if pd.isna(val):
            return val
        return _norm_m49(val)
    z['M49_Country_Code'] = z['M49_Country_Code'].apply(_format_m49_quote)
    
    print(f"[INFO] OK 最终保留 {len(z)} 行stock数据 ({z['commodity'].nunique()} 个物种, {z['country'].nunique()} 个国家)")
    
    # Include M49_Country_Code in returned data.
    return z[['M49_Country_Code','country','iso3','year','commodity','stock_head']].reset_index(drop=True)

def build_gv_areas_from_inputs(csv_path: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','land_use','area_ha'])
    df_raw = _read_csv(csv_path)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val  = _find_col(df, ['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])

    df = df.copy()
    mask_area = df[c_elem].astype(str).str.contains('area', case=False, na=False)
    mask_excl = df[c_elem].astype(str).str.contains('share|per capita|value', case=False, na=False) | \
                df[c_item].astype(str).str.contains('per capita', case=False, na=False)
    df = df[mask_area & ~mask_excl]
    if df.empty:
        return pd.DataFrame(columns=['country','iso3','year','land_use','area_ha'])

    if c_unit:
        unit_series = df[c_unit].astype(str).str.strip().str.lower()
        factor = unit_series.map(lambda u: 1000.0 if '1000' in u else 1.0)
    else:
        factor = 1.0
    df[c_val] = pd.to_numeric(df[c_val], errors='coerce')
    df[c_val] = df[c_val].multiply(factor, axis=0)

    item_series = df[c_item].astype(str).str.strip().str.lower()
    item_to_category = {
        'cropland': 'cropland_area_ha',
        'arable land': 'cropland_area_ha',
        'permanent crops': 'cropland_area_ha',
        'temporary crops': 'cropland_area_ha',
        'forest land': 'forest_area_ha',
        'naturally regenerating forest': 'forest_area_ha',
        'planted forest': 'forest_area_ha',
        'permanent meadows and pastures': 'pasture_area_ha',
        'permanent meadows & pastures - nat. growing': 'pasture_area_ha',
        'temporary meadows and pastures': 'pasture_area_ha',
    }
    df['land_use'] = item_series.map(item_to_category)
    df = df[df['land_use'].notna()]
    cols = [c_area, c_year, 'land_use', c_val]
    if 'M49_Country_Code' in df.columns:
        cols.append('M49_Country_Code')
    df = df[cols].copy()
    df = df.rename(columns={c_area: 'area', c_year: 'year', c_val: 'area_ha'})
    df = _attach_country_from_m49(df, df, universe, context=f"GV areas ({csv_path})")
    df = df[df['country'].isin(universe.countries)]
    df['iso3'] = df['country'].map(universe.iso3_by_country)

    hist_start = 2010
    hist_end = 2020
    df = df[(df['year'] >= hist_start) & (df['year'] <= hist_end)]

    needed_future = [y for y in universe.years if y > hist_end]
    if needed_future and not df.empty:
        latest = df.sort_values('year').drop_duplicates(['country','land_use'], keep='last')
        future_frames = []
        for y in needed_future:
            tmp = latest.copy()
            tmp['year'] = y
            future_frames.append(tmp)
        if future_frames:
            df = pd.concat([df] + future_frames, ignore_index=True)

    df['area_ha'] = pd.to_numeric(df['area_ha'], errors='coerce').fillna(0.0)
    df = df[df['area_ha'] > 0.0]
    return df[['country','iso3','year','land_use','area_ha']]

def build_land_use_fires_timeseries(csv_path: str, universe: Universe) -> pd.DataFrame:
    # Use historical values for 2010-2020 and hold the mean for future years.
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','co2e_kt'])
    df_raw = _read_csv(csv_path)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area=_find_col(df,['Area']); c_year=_find_col(df,['Year']); c_gas=_find_col(df,['Element','Gas','GHG']); c_val=_find_col(df,['Value'])
    z = df[[c_area,c_year,c_val]].copy(); z.columns=['area','year','co2e_kt']
    z = _attach_country_from_m49(df, z, universe, context=f"Land-use fires ({csv_path})")
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df['M49_Country_Code']
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    # fill future with mean of 2010-2020
    base = z[(z['year']>=2010)&(z['year']<=2020)].groupby(['country'], as_index=False)['co2e_kt'].mean().rename(columns={'co2e_kt':'mean_2010_2020'})
    fut = pd.DataFrame([(c,y) for c in universe.countries for y in universe.years if y>2020], columns=['country','year'])
    fut = fut.merge(base, on='country', how='left')
    fut['iso3'] = fut['country'].map(universe.iso3_by_country)
    fut['co2e_kt'] = fut['mean_2010_2020']
    fut['commodity'] = 'ALL'
    hist = z.copy(); hist['commodity']='ALL'
    out = pd.concat([hist[['country','iso3','year','commodity','co2e_kt']], fut[['country','iso3','year','commodity','co2e_kt']]], ignore_index=True)
    return out

# price

# constraints loaders
def load_intake_constraint(xlsx_path: str) -> Tuple[Dict[Tuple[str,int], float], Dict[str, float]]:
    """
    Read nutrition intake constraints and optional kcal-per-unit mapping.
    Returns:
      (rhs_map, kcal_map)
        rhs_map: {(country, year) -> required_kcal_total}
        kcal_map: {commodity -> kcal_per_unit}
    Heuristics are applied to match columns.
    """
    rhs_map: Dict[Tuple[str,int], float] = {}
    kcal_map: Dict[str, float] = {}
    if not os.path.exists(xlsx_path):
        return rhs_map, kcal_map
    xls = pd.ExcelFile(xlsx_path)
    # Primary sheet for RHS: try first sheet
    try:
        df = _lc(_read_excel(xls, xls.sheet_names[0]))
        c_m49 = _maybe_find_col(df, ['M49_Country_Code', 'M49 Code', 'M49'])
        c_year = _maybe_find_col(df, ['Year'])
        # RHS candidates
        cand = _maybe_find_col(df, ['kcal_min_pc','kcal_min','intake_kcal_pc','intake_kcal','rhs','value'])
        c_pop = _maybe_find_col(df, ['Population','Pop'])
        if not c_m49:
            raise ValueError(f"load_intake_constraint: {xlsx_path} 缺少 M49_Country_Code 列")
        if c_year and cand:
            z = df[[c_m49, c_year, cand] + ([c_pop] if c_pop else [])].copy()
            z.columns = ['M49_Country_Code','year','val'] + (['pop'] if c_pop else [])
            z['M49_Country_Code'] = z['M49_Country_Code'].apply(_norm_m49)
            if 'pop' in z.columns and z['pop'].notna().any():
                z['rhs'] = pd.to_numeric(z['val'], errors='coerce') * pd.to_numeric(z['pop'], errors='coerce')
            else:
                z['rhs'] = pd.to_numeric(z['val'], errors='coerce')
            for r in z.itertuples(index=False):
                try:
                    rhs_map[(str(r.M49_Country_Code), int(r.year))] = float(r.rhs)
                except Exception:
                    pass
    except ValueError:
        raise
    except Exception:
        pass
    # Optional kcals per commodity mapping: try a sheet named like 'kcal' or with columns
    try:
        sheet = None
        for nm in xls.sheet_names:
            if 'kcal' in nm.lower():
                sheet = nm; break
        if sheet is None:
            sheet = xls.sheet_names[0]
        df2 = _lc(_read_excel(xls, sheet))
        c_comm = _maybe_find_col(df2, ['Commodity','Item','Product'])
        c_kcal = _maybe_find_col(df2, ['kcal_per_unit','kcal per unit','kcal/unit','kcal_per_ton'])
        if c_comm and c_kcal:
            for r in df2[[c_comm, c_kcal]].itertuples(index=False):
                try:
                    kcal_map[str(r[0])] = float(r[1])
                except Exception:
                    pass
    except Exception:
        pass
    return rhs_map, kcal_map

def load_land_area_limits(csv_path: str) -> Dict[Tuple[str,int], float]:
    """Parse FAOSTAT Inputs csv to extract 'Land area' by country-year.
    Returns {(m49_code, year): land_area_ha} where m49_code is in 'xxx format.
    """
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(csv_path):
        return out
    df_raw = _read_csv(csv_path)
    # Process wide data directly without wide_to_long.
    # Required: M49_Country_Code, Item, Element, Y1961...Y2022.
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    # Find M49.
    m49_col = None
    for c in df.columns:
        if 'm49' in c.lower() and 'code' in c.lower():
            m49_col = c
            break
    if m49_col is None:
        raise ValueError(f"load_land_area_limits: {csv_path} 缺少 M49_Country_Code 列")
    
    # Select Item='Land area' and Element='Area'.
    item_col = _find_col(df, ['Item'])
    elem_col = _find_col(df, ['Element'])
    unit_col = _maybe_find_col(df, ['Unit'])
    
    mask = (df[item_col].astype(str).str.strip() == 'Land area') & \
           (df[elem_col].astype(str).str.strip() == 'Area')
    df_land = df[mask].copy()
    
    # Find year columns such as Y1961 and Y2020.
    year_cols = [c for c in df_land.columns if c.startswith('Y') and c[1:].isdigit()]
    
    for _, row in df_land.iterrows():
        # Standardize M49 as apostrophe-prefixed three-digit strings.
        raw_m49 = str(row[m49_col]).strip()
        # Remove existing quotes, then add one consistently.
        m49_code = _norm_m49(raw_m49)
        if not m49_code:
            continue
        
        unit_raw = str(row.get(unit_col, '')).strip().lower() if unit_col else ''
        unit_scale = 1000.0 if '1000' in unit_raw else 1.0
        for yc in year_cols:
            try:
                year = int(yc[1:])  # Y2020 -> 2020
                val = float(row[yc])
                if pd.notna(val):
                    out[(m49_code, year)] = val * unit_scale
            except (ValueError, TypeError):
                continue
    
    return out

def build_energy_supply_rhs(fbs_csv: str, universe: Universe) -> Dict[Tuple[str,int], float]:
    """Build country-year total energy supply RHS from FAOSTAT FBS.
    Uses elements containing 'kcal/capita/day' and multiplies by population if present.
    Returns {(country, year): kcal_total_per_year}.
    """
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(fbs_csv):
        return out
    df_raw = _read_fbs_table(fbs_csv)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_m49 = _maybe_find_col(df, ['M49_Country_Code'])
    if not c_m49:
        raise ValueError(f"build_energy_supply_rhs: {fbs_csv} 缺少 M49_Country_Code 列")
    c_year=_find_col(df,['Year']); c_elem=_find_col(df,['Element']); c_val=_find_col(df,['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])
    # Energy per capita (kcal/capita/day)
    e = df[(df[c_elem].astype(str).str.contains('kcal/capita/day', case=False, na=False))]
    if not len(e):
        return out
    e2 = e.groupby([c_m49, c_year], as_index=False)[c_val].sum().rename(columns={c_val:'kcal_pc_day'})
    # Population (if available)
    p = df[df[c_elem].astype(str).str.contains('population', case=False, na=False)].copy()
    if len(p):
        if c_unit and (p[c_unit].astype(str).str.contains('1000', case=False, na=False)).any():
            p[c_val] = pd.to_numeric(p[c_val], errors='coerce') * 1000.0
        p2 = p.groupby([c_m49, c_year], as_index=False)[c_val].sum().rename(columns={c_val:'population'})
        z = e2.merge(p2, on=[c_m49, c_year], how='left')
    else:
        z = e2.copy(); z['population'] = np.nan
    z['rhs'] = pd.to_numeric(z['kcal_pc_day'], errors='coerce') * 365.0 * pd.to_numeric(z['population'], errors='coerce')
    for r in z[[c_m49, c_year, 'rhs']].itertuples(index=False):
        try:
            out[(_norm_m49(r[0]), int(r[1]))] = float(r[2])
        except Exception:
            pass
    return out

# nutrition (future) helpers
def load_nutrient_factors_from_dict_v3(xls_path: str, indicator: str) -> Dict[str, float]:
    """Read Emis_item sheet and derive nutrient-per-ton factors by commodity.
    indicator: 'energy' | 'protein' | 'fat'
    Returns {commodity -> factor_per_ton}
    """
    if not os.path.exists(xls_path):
        return {}
    xls = pd.ExcelFile(xls_path)
    try:
        df = _lc(_read_excel(xls, 'Emis_item'))
    except Exception:
        return {}
    # Match nutrition primarily on Item_Production_Map; also map Item_Emis to the same coefficients to avoid misses.
    c_comm_prod = _find_col(df, ['Item_Production_Map'])
    c_comm_emis = _maybe_find_col(df, ['Item_Emis'])
    c_kcal = 'kcal_per_100g' if 'kcal_per_100g' in df.columns else None
    c_prot = 'g_protein_per_100g' if 'g_protein_per_100g' in df.columns else None
    c_fat = 'g_fat_per_100g' if 'g_fat_per_100g' in df.columns else None
    out: Dict[str, float] = {}
    for r in df.itertuples(index=False):
        keys: List[str] = []
        if c_comm_prod:
            v_prod = getattr(r, c_comm_prod)
            if pd.notna(v_prod):
                keys.append(str(v_prod).strip())
        if c_comm_emis:
            v_emis = getattr(r, c_comm_emis)
            if pd.notna(v_emis):
                keys.append(str(v_emis).strip())
        keys = [k for k in keys if k and k.lower() not in {'nan', 'no'}]
        if not keys:
            continue
        if indicator == 'energy' and c_kcal:
            v = getattr(r, c_kcal)
            try:
                val = float(v) * 10000.0  # 1 t = 10,000×100g
                if np.isfinite(val):
                    for k in keys:
                        out[k] = val
            except Exception:
                pass
        elif indicator == 'protein' and c_prot:
            v = getattr(r, c_prot)
            try:
                val = float(v) * 10000.0  # grams per ton
                if np.isfinite(val):
                    for k in keys:
                        out[k] = val
            except Exception:
                pass
        elif indicator == 'fat' and c_fat:
            v = getattr(r, c_fat)
            try:
                val = float(v) * 10000.0  # grams per ton
                if np.isfinite(val):
                    for k in keys:
                        out[k] = val
            except Exception:
                pass
    return out

def load_intake_targets(xlsx_path: str,
                        indicator: str,
                        value_col: Optional[str] = None) -> Dict[str, float]:
    """Load per-capita-per-day intake targets by country from Intake_constraint.xlsx.
    Uses 'Indicator' column to filter rows for one of ['Energy supply','Protein supply','Fat supply'].
    Returns {country -> mean_value_per_capita_per_day}. When value_col is provided, use that column name.
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(xlsx_path):
        return {}
    df = _lc(_read_excel(xlsx_path, sheet_name='extract'))
    c_cty = _maybe_find_col(df, ['M49_Country_Code', 'M49 Code', 'M49'])
    c_ind = _maybe_find_col(df, ['Indicator'])
    c_val = None
    if value_col:
        c_val = _maybe_find_col(df, [value_col])
        if not c_val:
            raise ValueError(f"load_intake_targets: {xlsx_path} missing column {value_col}")
    else:
        if indicator == 'energy':
            c_val = _maybe_find_col(df, ['MDER_2006_08_kcal_cap_day'])
            if not c_val:
                raise ValueError(f"load_intake_targets: {xlsx_path} missing column MDER_2006_08_kcal_cap_day")
        elif indicator in ('protein', 'fat'):
            c_val = _maybe_find_col(df, ['MIN', 'Min'])
    if not (c_cty and c_ind and c_val):
        if not c_cty:
            raise ValueError(f"load_intake_targets: {xlsx_path} missing M49_Country_Code column")
        return {}
    key = {'energy':'energy', 'protein':'protein', 'fat':'fat'}[indicator]
    z = df[df[c_ind].astype(str).str.lower().str.contains(key)].copy()
    out: Dict[str, float] = {}
    missing: List[str] = []
    value_col_label = value_col or c_val or ''
    for r in z[[c_cty, c_val]].itertuples(index=False):
        try:
            m49 = _norm_m49(r[0])
            val = float(r[1])
            out[m49] = val
        except Exception:
            if indicator == 'energy':
                missing.append(_norm_m49(r[0]))
            continue
    if indicator == 'energy':
        all_codes = [_norm_m49(x) for x in z[c_cty].tolist()]
        missing += [c for c in all_codes if c and c not in out]
        missing = [c for c in missing if c]
        if missing:
            uniq = sorted(set(missing))
            logger.error(
                "[NUTRI] %s missing for %s countries: %s",
                value_col_label,
                len(uniq), ", ".join(uniq)
            )
            raise ValueError(
                f"load_intake_targets: {value_col_label} missing for {len(uniq)} countries"
            )
    return out

def load_population_wpp(csv_path: str, universe: Optional[Universe] = None) -> Dict[Tuple[str,int], float]:
    """Load population by country-year from WPP.
    Returns {(country, year): population} (persons). When ``universe`` is provided,
    country names are normalised to match the model naming (prefer M49 codes).
    """
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(csv_path):
        return out
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'latin1']
    last_err = None
    for enc in encodings:
        try:
            raw = _read_csv(csv_path, encoding=enc)
            df = _lc(_faostat_wide_to_long(raw))
            break
        except UnicodeDecodeError as err:
            last_err = err
    else:
        raw = _read_csv(csv_path, encoding='utf-8', errors='replace')
        df = _lc(_faostat_wide_to_long(raw))
        if last_err:
            print(f"[load_population_wpp] WARNING: fallback to utf-8 with replacement due to encoding error: {last_err}")

    c_m49 = _maybe_find_col(df, ['M49_Country_Code'])
    c_year = _maybe_find_col(df, ['Year'])
    c_val  = _maybe_find_col(df, ['Value','Population'])
    if not c_m49:
        raise ValueError(f"load_population_wpp: {csv_path} 缺少 M49_Country_Code 列")
    if not (c_year and c_val):
        return out

    c_elem = _maybe_find_col(df, ['Element'])
    if c_elem:
        mask_total = df[c_elem].astype(str).str.lower().str.contains('total population') & df[c_elem].astype(str).str.lower().str.contains('both')
        df = df.loc[mask_total].copy()
    if df.empty:
        return out

    unit_col = _maybe_find_col(df, ['Unit'])
    if unit_col:
        has_thousand = df[unit_col].astype(str).str.contains('1000', case=False, na=False)
        if has_thousand.any():
            df[c_val] = pd.to_numeric(df[c_val], errors='coerce') * 1000.0
        else:
            df[c_val] = pd.to_numeric(df[c_val], errors='coerce')
    else:
        df[c_val] = pd.to_numeric(df[c_val], errors='coerce')

    country_series = df[c_m49].apply(_norm_m49)
    valid_countries = set(universe.countries) if universe is not None else None

    for country, year_val, pop_val in zip(country_series, df[c_year], df[c_val]):
        try:
            country_name = str(country).strip()
            year = int(year_val)
            pop = float(pop_val)
        except Exception:
            continue
        if not np.isfinite(pop) or pop <= 0:
            continue
        if valid_countries is not None and country_name not in valid_countries:
            continue
        out[(country_name, year)] = pop
    return out

def build_nutrition_rhs_for_future(universe: Universe,
                                   pop_map: Dict[Tuple[str,int], float],
                                   intake_target_pc_day: Dict[str, float]) -> Dict[Tuple[str,int], float]:
    """Construct RHS only for future years (t>2020): rhs[i,t] = mean_pc_day(country) * 365 * population(i,t)."""
    rhs: Dict[Tuple[str,int], float] = {}
    for i in universe.countries:
        for t in universe.years:
            if t <= 2020:
                continue
            mean_pc_day = intake_target_pc_day.get(i)
            pop = pop_map.get((i, t))
            if mean_pc_day is None or pop is None:
                continue
            rhs[(i, t)] = float(mean_pc_day) * 365.0 * float(pop)
    return rhs

# temperature driver
def apply_temperature_multiplier_to_nodes(temp_xlsx: str,
                                          nodes: List[Node],
                                          *,
                                          scenario: Optional[str] = None,
                                          region: Optional[str] = None,
                                          variable: Optional[str] = None,
                                          model: Optional[str] = None,
                                          model_aggregation: Optional[str] = None,
                                          strict: bool = True) -> None:
    """Apply temperature multiplier Tmult to nodes if file provides it.
    Supports two formats:
      1) Country/Area + Year + Tmult
      2) IAM wide table with SCENARIO/REGION/VARIABLE and YYYYY columns
    """
    if not os.path.exists(temp_xlsx):
        if strict:
            raise FileNotFoundError(f"Temperature input not found: {temp_xlsx}")
        return
    logger = logging.getLogger(__name__)
    try:
        df = _lc(_read_excel(temp_xlsx, sheet_name=0))
    except Exception as exc:
        if strict:
            raise
        logger.warning("Failed to read temperature file: %s", exc)
        return
    c_area = _maybe_find_col(df, ['Country','Area'])
    c_year = _maybe_find_col(df, ['Year'])
    c_tmul = _maybe_find_col(df, ['Tmult','temp_mult','temperature_multiplier'])
    if not (c_area and c_year and c_tmul):
        # IAM wide format: SCENARIO/REGION/VARIABLE + YYYYY columns
        if not all(c in df.columns for c in ['SCENARIO', 'REGION', 'VARIABLE']):
            return
        year_cols = [c for c in df.columns
                     if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit()]
        if not year_cols:
            return

        scenarios = [str(s) for s in df['SCENARIO'].dropna().unique()]
        variables = [str(v) for v in df['VARIABLE'].dropna().unique()]
        regions = [str(r) for r in df['REGION'].dropna().unique()]

        def _pick_default_scenario(cands: List[str]) -> Optional[str]:
            for pref in ('SSP2-Baseline', 'SSP2-45', 'SSP2-34', 'SSP2-26', 'SSP2-19', 'SSP2-60'):
                if pref in cands:
                    return pref
            for s in sorted(cands):
                if 'Baseline' in s:
                    return s
            return sorted(cands)[0] if cands else None

        def _select_scenario(req: Optional[str]) -> Optional[str]:
            if not scenarios:
                return None
            if req:
                req = str(req).strip()
                if req in scenarios:
                    return req
                prefix = req.split('_', 1)[0]
                matches = [s for s in scenarios if s.startswith(prefix)]
                if matches:
                    return _pick_default_scenario(matches)
            return _pick_default_scenario(scenarios)

        def _select_variable(req: Optional[str]) -> Optional[str]:
            if not variables:
                return None
            if req:
                req = str(req).strip()
                if req in variables:
                    return req
            if len(variables) == 1:
                return variables[0]
            for v in variables:
                if 'temperature' in v.lower() and 'global' in v.lower():
                    return v
            return sorted(variables)[0]

        def _select_region(req: Optional[str]) -> Optional[str]:
            if not regions:
                return None
            if req:
                req = str(req).strip()
                if req in regions:
                    return req
            if len(regions) == 1:
                return regions[0]
            if 'World' in regions:
                return 'World'
            return sorted(regions)[0]

        chosen_scenario = _select_scenario(scenario)
        chosen_variable = _select_variable(variable)
        chosen_region = _select_region(region)
        if strict:
            for label, requested, chosen in [('SCENARIO', scenario, chosen_scenario),
                                              ('VARIABLE', variable, chosen_variable),
                                              ('REGION', region, chosen_region)]:
                if requested is not None and str(requested).strip() != chosen:
                    raise ValueError(f"Temperature {label} {requested!r} not found; refusing implicit fallback to {chosen!r}")
        if not (chosen_scenario and chosen_variable):
            return

        df_use = df[(df['SCENARIO'] == chosen_scenario) & (df['VARIABLE'] == chosen_variable)]
        if chosen_region:
            df_use = df_use[df_use['REGION'] == chosen_region]
        if df_use.empty:
            if strict:
                raise ValueError("Temperature selection contains no records")
            logger.warning(
                "Temperature filter empty: scenario=%s variable=%s region=%s",
                chosen_scenario, chosen_variable, chosen_region
            )
            return

        df_use = df_use.copy()
        if model_aggregation not in (None, 'mean'):
            raise ValueError("Temperature model_aggregation must be None or 'mean'")
        if model is not None and model_aggregation is not None:
            raise ValueError("Choose a temperature MODEL or an aggregation policy, not both")
        models = sorted(df_use['MODEL'].dropna().astype(str).unique()) if 'MODEL' in df_use else []
        if model is not None:
            if str(model) not in models:
                raise ValueError(f"Temperature MODEL {model!r} not found; available={models}")
            df_use = df_use[df_use['MODEL'].astype(str) == str(model)].copy()
        elif len(models) > 1 and model_aggregation is None:
            raise ValueError(f"Temperature has multiple MODEL values {models}; select one or explicitly request mean")
        source_keys = ['SCENARIO', 'REGION', 'VARIABLE'] + (['MODEL'] if 'MODEL' in df_use else [])
        if df_use.duplicated(source_keys).any():
            raise ValueError("Temperature input has duplicate model/scenario/region/variable records")
        if 'UNIT' in df_use and df_use['UNIT'].dropna().nunique() > 1:
            raise ValueError("Temperature selection mixes units")
        provenance = {'scenario': chosen_scenario, 'region': chosen_region, 'variable': chosen_variable,
                      'model': str(model) if model is not None else (models[0] if len(models) == 1 else None),
                      'model_aggregation': model_aggregation, 'models': models if model is None else [str(model)],
                      'unit': str(df_use['UNIT'].dropna().iloc[0]) if 'UNIT' in df_use and df_use['UNIT'].notna().any() else None}
        long_df = df_use.melt(
            id_vars=['SCENARIO', 'REGION', 'VARIABLE'],
            value_vars=year_cols,
            var_name='Year',
            value_name='Tmult'
        )
        long_df = long_df.dropna(subset=['Tmult'])
        long_df['Year'] = long_df['Year'].astype(str).str.lstrip('Y')
        long_df = long_df[long_df['Year'].str.isdigit()]
        long_df['Year'] = long_df['Year'].astype(int)
        long_df['Tmult'] = pd.to_numeric(long_df['Tmult'], errors='raise')
        if not np.isfinite(long_df['Tmult']).all():
            raise ValueError("Temperature selection contains nonfinite values")
        if model_aggregation == 'mean':
            # Sort before aggregation so file row order cannot affect reduction order.
            long_df = long_df.sort_values(['REGION', 'Year', 'Tmult']).groupby(['REGION', 'Year'], as_index=False)['Tmult'].mean()
        if long_df.duplicated(['REGION', 'Year']).any():
            raise ValueError("Temperature selection has duplicate region/year keys")
        key = {(str(getattr(r, 'REGION')), int(getattr(r, 'Year'))): float(getattr(r, 'Tmult'))
               for r in long_df.itertuples(index=False) if pd.notna(getattr(r, 'Tmult'))}
        world_key = 'World' if 'World' in regions else None

        applied = 0
        for n in nodes:
            v = key.get((n.country, n.year))
            if v is None and world_key:
                v = key.get((world_key, n.year))
            if v is not None:
                n.Tmult = float(v)
                n.meta = dict(n.meta or {})
                n.meta['temperature_source'] = provenance.copy()
                applied += 1
            elif strict and int(n.year) > 2020:
                raise ValueError(f"Temperature input missing requested future year {n.year}")
        logger.info(
            "Applied temperature multiplier from IAM: scenario=%s variable=%s region=%s applied=%s",
            chosen_scenario, chosen_variable, chosen_region, applied
        )
        logger.info("Temperature source selection: %s", provenance)
        return

    if df.duplicated([c_area, c_year]).any():
        raise ValueError("Temperature input has duplicate country/year records")
    key = {(str(getattr(r, c_area)), int(getattr(r, c_year))): float(getattr(r, c_tmul))
           for r in df.itertuples(index=False) if pd.notna(getattr(r, c_tmul))}
    applied = 0
    for n in nodes:
        v = key.get((n.country, n.year))
        if v is not None:
            n.Tmult = float(v)
            applied += 1
    logger.info("Applied temperature multiplier: rows=%s applied=%s", len(key), applied)

# yield calculation & assignment
def compute_yield_from_prod_area(production_csv: str, inputs_csv: str, universe: Universe) -> pd.DataFrame:
    """Backward compatible wrapper that now derives yield directly from Production NOFLAG file."""
    maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
    data = build_faostat_production_indicators(production_csv, universe, maps)
    return data['yield']

def assign_yield0_to_nodes(nodes: List[Node], yield_df: pd.DataFrame, *, hist_start:int=2020, hist_end:int=2020) -> None:
    """Assign baseline yield0 (t/ha) to nodes; defaults to the base year (2020) when no range is specified."""
    if yield_df is None or len(yield_df)==0:
        return
    df = _lc(yield_df)
    df = df[(df['year']>=hist_start)&(df['year']<=hist_end)]
    base = df.groupby(['country','commodity'], as_index=False)['yield_t_per_ha'].mean().rename(columns={'yield_t_per_ha':'yield0'})
    key = {(r.country, r.commodity): float(r.yield0) for r in base.itertuples(index=False)}
    for n in nodes:
        v = key.get((n.country, n.commodity))
        if v is not None and v > 0:
            n.meta['yield0'] = float(v)


def assign_grassland_coef_to_nodes(nodes: List[Node],
                                    paths: DataPaths,
                                    universe: Universe,
                                    maps: EmisItemMappings,
                                    stock_df: Optional[pd.DataFrame] = None,
                                    livestock_yield_df: Optional[pd.DataFrame] = None,
                                    yield_multiplier: Optional[Dict[Tuple[str, str, int], float]] = None,
                                    dm_conversion_multiplier: Optional[Dict[Tuple[str, str, int], float]] = None,
                                    feed_requirement_scheme: Optional[str] = None,
                                    use_regional_aggregation: bool = True,
                                    verbose: bool = False) -> None:
    """
    Calculate and assign grassland coefficients to livestock node metadata.
    
    Workflow:
    1. Use S3_0_ds_linear_regional.load_grassland_coefficients for regional ha/head coefficients.
    2. Prefer yield times slaughter ratio for tonnes/head, consistent with GLE, then convert to ha/tonne.
    3. If yield is missing, estimate tonnes/head from production/stock and assign meta['grassland_coef'].
    
    Args:
        nodes: Node list.
        paths: Data path configuration.
        universe: Universe configuration with countries, years, etc.
        maps: Emissions-item mappings for commodity-to-species conversion.
        stock_df: Optional stocks for production/stock fallback.
        livestock_yield_df: Optional yields (t/head) for yield-times-slaughter-ratio conversion.
        yield_multiplier: Optional scenario yield multipliers {(country, commodity, year): mult}.
    """
    def print(*args, **kwargs):
        if verbose:
            builtins.print(*args, **kwargs)

    try:
        # Import the grassland coefficient loader.
        from S3_0_ds_linear_regional import load_grassland_coefficients
        
        # Calculate region-species grassland coefficients in ha/head.
        grassland_coef_ha_per_head = load_grassland_coefficients(
            feed_need_xlsx=paths.feed_need_xlsx,
            grass_ratio_xlsx=paths.grass_ratio_xlsx,
            pasture_yield_xlsx=paths.pasture_dm_yield_xlsx,
            dict_v3_path=paths.dict_v3_path,
            years=universe.years,
            use_regional_aggregation=use_regional_aggregation,
            feed_requirement_scheme=feed_requirement_scheme,
            verbose=verbose,
        )
        
        if not grassland_coef_ha_per_head:
            print("[GRASSLAND_COEF] ?? 未计算到任何grassland系数，跳过分配")
            return
        
        scope_label = "区域" if use_regional_aggregation else "国家"
        print(f"[GRASSLAND_COEF] 已加载 {len(grassland_coef_ha_per_head)} 个{scope_label}-species grassland系数")
        
        # Print unique species in the dictionary.
        unique_species = sorted(set(species for (region, species) in grassland_coef_ha_per_head.keys()))
        print(f"[GRASSLAND_COEF] 字典中的species列表: {unique_species}")
        
        # Print unique regions/countries in the dictionary.
        unique_regions = sorted(set(region for (region, species) in grassland_coef_ha_per_head.keys()))
        print(f"[GRASSLAND_COEF] 字典中的{scope_label}列表: {unique_regions}")
        
        # Print sample grassland_coef keys.
        sample_keys = list(grassland_coef_ha_per_head.keys())[:10]
        print(f"[GRASSLAND_COEF] 系数字典样例键: {sample_keys}")
        
        # Print Argentina's species using M49 keys.
        arg_m49 = next((m49 for m49, name in (universe.country_by_m49 or {}).items() if name == 'Argentina'), None)
        if arg_m49:
            argentina_species = sorted([species for (region, species) in grassland_coef_ha_per_head.keys() if region == arg_m49])
            print(f"\n[GRASSLAND_COEF] ?? Argentina({arg_m49})可用的species: {argentina_species}")
            if len(argentina_species) < 10:
                print(f"[GRASSLAND_COEF] ?? Argentina({arg_m49})只有 {len(argentina_species)} 个species，数据可能不完整！")
        else:
            print("\n[GRASSLAND_COEF] ?? Argentina未在country_by_m49中找到，跳过诊断")
        
        # Calculate production-to-head conversion factors if stock_df is available.
        def _build_ton_per_head_maps() -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[str, float]]:
            if stock_df is None or stock_df.empty:
                return {}, {}, {}
            try:
                prod_df = build_production_from_faostat(paths.production_faostat_csv, universe)
                hist_years = [y for y in universe.years if y <= 2020]
                prod_df = prod_df[prod_df['year'].isin(hist_years)]
                stocks = stock_df[stock_df['year'].isin(hist_years)]
                merged = prod_df.merge(
                    stocks[['country', 'commodity', 'year', 'stock_head']],
                    on=['country', 'commodity', 'year'],
                    how='inner'
                )
                merged = merged[(merged['production_t'] > 0) & (merged['stock_head'] > 0)]
                if merged.empty:
                    return {}, {}, {}
                merged['ton_per_head'] = merged['production_t'] / merged['stock_head']
                country_ratio = merged.groupby(['country', 'commodity'])['ton_per_head'].mean().to_dict()
                merged['region'] = merged['country'].map(country_to_region)
                region_ratio = merged.dropna(subset=['region']).groupby(['region', 'commodity'])['ton_per_head'].mean().to_dict()
                world_ratio = merged.groupby('commodity')['ton_per_head'].mean().to_dict()
                print(f"[GRASSLAND_COEF] ? ton/head映射：国家级 {len(country_ratio)} 对，区域级 {len(region_ratio)} 对，全球 {len(world_ratio)} 个品类")
                return country_ratio, region_ratio, world_ratio
            except Exception as e:
                print(f"[GRASSLAND_COEF] ?? 构建ton/head映射失败: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                return {}, {}, {}
        country_to_region = {}
        if hasattr(universe, 'region_aggMC_by_country'):
            country_to_region = universe.region_aggMC_by_country or {}
        
        if not country_to_region:
            # Load missing region mappings from dict_v3 using M49 keys.
            try:
                # pd is already imported at module level.
                dict_v3 = _read_excel(paths.dict_v3_path, sheet_name='region')
                for _, row in dict_v3.iterrows():
                    m49_raw = row.get('M49_Country_Code', row.get('M49 Code', row.get('M49')))
                    region = str(row.get('Region_market_agg', '')).strip()
                    if pd.notna(m49_raw) and region and region != 'no':
                        country_to_region[_norm_m49(m49_raw)] = region
            except Exception as e:
                print(f"[GRASSLAND_COEF] ?? 从dict_v3加载区域映射失败: {e}")

        hist_years = [y for y in universe.years if y <= 2020]
        base_year = max(hist_years) if hist_years else max(universe.years)
        yield_mult = yield_multiplier if isinstance(yield_multiplier, dict) else {}
        dm_mult = dm_conversion_multiplier if isinstance(dm_conversion_multiplier, dict) else {}

        def _get_dm_mult(country: str, commodity: str, year: int) -> float:
            m49 = _norm_m49(country)
            try:
                y = int(year)
            except Exception:
                y = base_year
            key = (m49, commodity, y)
            val = dm_mult.get(key)
            if val is None and y != base_year:
                val = dm_mult.get((m49, commodity, base_year))
            try:
                return float(val) if val is not None else 1.0
            except Exception:
                return 1.0

        def _build_livestock_yield_maps(yield_df: Optional[pd.DataFrame]) -> Tuple[Dict[Tuple[str, str, int], float], Dict[Tuple[str, str, int], float], Dict[Tuple[str, int], float]]:
            if yield_df is None or yield_df.empty:
                return {}, {}, {}
            df = _lc(yield_df)
            if 'yield_t_per_head' not in df.columns or 'commodity' not in df.columns:
                return {}, {}, {}
            df = df.copy()
            if 'M49_Country_Code' in df.columns:
                df['country_key'] = df['M49_Country_Code'].apply(_norm_m49)
            elif 'country' in df.columns:
                df['country_key'] = df['country'].apply(_norm_m49)
            else:
                return {}, {}, {}
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)
            df['yield_t_per_head'] = pd.to_numeric(df['yield_t_per_head'], errors='coerce')
            df = df.dropna(subset=['yield_t_per_head'])
            df = df[df['yield_t_per_head'] > 0]
            df = df[df['country_key'].notna() & (df['country_key'] != '')]
            if df.empty:
                return {}, {}, {}
            grouped = df.groupby(['country_key', 'commodity', 'year'], as_index=False)['yield_t_per_head'].mean()
            country_map = {(r.country_key, r.commodity, int(r.year)): float(r.yield_t_per_head) for r in grouped.itertuples(index=False)}
            grouped['region'] = grouped['country_key'].map(country_to_region)
            region_map = grouped.dropna(subset=['region']).groupby(['region', 'commodity', 'year'])['yield_t_per_head'].mean().to_dict()
            world_map = grouped.groupby(['commodity', 'year'])['yield_t_per_head'].mean().to_dict()
            print(f"[GRASSLAND_COEF]  yield_t_per_head映射：国家级 {len(country_map)} 对，区域级 {len(region_map)} 对，全球 {len(world_map)} 个品类-年份")
            return country_map, region_map, world_map

        def _build_slaughter_ratio_maps() -> Tuple[Dict[str, str], Dict[Tuple[str, str, int], float], Dict[Tuple[str, str, int], float], Dict[Tuple[str, int], float]]:
            try:
                emis_item_df = _read_excel(paths.dict_v3_path, sheet_name='Emis_item')
            except Exception as e:
                print(f"[GRASSLAND_COEF] ?? 读取Item_SlaughteredRatio失败: {e}")
                return {}, {}, {}, {}
            ratio_item_by_emis: Dict[str, str] = {}
            ratio_element_by_item: Dict[str, str] = {}
            for _, row in emis_item_df.iterrows():
                item_emis = row.get('Item_Emis')
                ratio_item = row.get('Item_SlaughteredRatio_Map')
                if pd.isna(item_emis) or pd.isna(ratio_item):
                    continue
                item_emis = str(item_emis).strip()
                ratio_item = str(ratio_item).strip()
                if not item_emis or not ratio_item or item_emis.lower() in {'nan', 'no'} or ratio_item.lower() in {'nan', 'no'}:
                    continue
                ratio_item_by_emis[item_emis] = ratio_item
                elem = row.get('Item_SlaughteredRatio_Element')
                elem_clean = str(elem).strip() if pd.notna(elem) else ''
                ratio_element_by_item[ratio_item] = elem_clean or 'Producing/Slaughtered ratio'
            if not ratio_element_by_item:
                return ratio_item_by_emis, {}, {}, {}
            try:
                df_raw = _read_csv(paths.production_faostat_csv)
                df_raw = _filter_select_rows(df_raw)
                df = _lc(_faostat_wide_to_long(df_raw))
                c_area = _find_col(df, ['Area'])
                c_year = _find_col(df, ['Year'])
                c_item = _find_col(df, ['Item'])
                c_elem = _find_col(df, ['Element'])
                c_val = _find_col(df, ['Value'])
            except Exception as e:
                print(f"[GRASSLAND_COEF] ?? 读取屠宰率数据失败: {e}")
                return ratio_item_by_emis, {}, {}, {}
            keep_cols = [c_area, c_year, c_item, c_elem, c_val]
            if 'M49_Country_Code' in df.columns:
                keep_cols.append('M49_Country_Code')
            z = df[keep_cols].copy()
            z = z.rename(columns={c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'})
            z['value'] = pd.to_numeric(z['value'], errors='coerce')
            z = z.dropna(subset=['value'])
            try:
                z = _attach_country_from_m49(df, z, universe, context="FAOSTAT slaughter ratio")
            except Exception as e:
                print(f"[GRASSLAND_COEF] ?? 绑定国家失败: {e}")
                return ratio_item_by_emis, {}, {}, {}
            if 'M49_Country_Code' in z.columns:
                z['M49_Country_Code'] = z['M49_Country_Code'].apply(_norm_m49)
            z = z[z['item_raw'].isin(ratio_element_by_item.keys())]
            z['element_norm'] = z['element'].astype(str).str.strip().str.lower()
            z['target_norm'] = z['item_raw'].map(ratio_element_by_item).fillna('').astype(str).str.strip().str.lower()
            z = z[(z['target_norm'] != '') & (z['element_norm'] == z['target_norm'])]
            z['year'] = pd.to_numeric(z['year'], errors='coerce')
            z = z.dropna(subset=['year'])
            z['year'] = z['year'].astype(int)
            if z.empty:
                return ratio_item_by_emis, {}, {}, {}
            grouped = z.groupby(['country', 'item_raw', 'year'], as_index=False)['value'].mean()
            country_map = {(r.country, r.item_raw, int(r.year)): float(r.value) for r in grouped.itertuples(index=False)}
            grouped['region'] = grouped['country'].map(country_to_region)
            region_map = grouped.dropna(subset=['region']).groupby(['region', 'item_raw', 'year'])['value'].mean().to_dict()
            world_map = grouped.groupby(['item_raw', 'year'])['value'].mean().to_dict()
            print(f"[GRASSLAND_COEF]  slaughter_ratio映射：国家级 {len(country_map)} 对，区域级 {len(region_map)} 对，全球 {len(world_map)} 个Item-年份")
            return ratio_item_by_emis, country_map, region_map, world_map

        yield_country_map, yield_region_map, yield_world_map = _build_livestock_yield_maps(livestock_yield_df)
        ratio_item_by_emis, ratio_country_map, ratio_region_map, ratio_world_map = _build_slaughter_ratio_maps()
        use_yield_ratio = bool(yield_country_map or yield_region_map or yield_world_map)

        country_ratio_map, region_ratio_map, world_ratio_map = _build_ton_per_head_maps()

        def _get_yield_t_per_head(country: str, commodity: str, year: int) -> float:
            if not use_yield_ratio:
                return 0.0
            y = int(year) if year is not None else base_year
            used_fallback = False
            val = yield_country_map.get((country, commodity, y), 0.0)
            if (not val or val <= 0) and y != base_year:
                used_fallback = True
                val = yield_country_map.get((country, commodity, base_year), 0.0)
            if (not val or val <= 0) and use_regional_aggregation:
                used_fallback = True
                region = country_to_region.get(country)
                if region:
                    val = yield_region_map.get((region, commodity, y), 0.0) or yield_region_map.get((region, commodity, base_year), 0.0)
            if not val or val <= 0:
                used_fallback = True
                val = yield_world_map.get((commodity, y), 0.0) or yield_world_map.get((commodity, base_year), 0.0)
            if val and yield_mult:
                country_candidates = [country]
                country_m49 = _norm_m49(country)
                if country_m49 and country_m49 not in country_candidates:
                    country_candidates.append(country_m49)
                mult = None
                for country_key in country_candidates:
                    mult = yield_mult.get((country_key, commodity, y))
                    if mult is not None:
                        break
                if mult is None and y != base_year:
                    for country_key in country_candidates:
                        mult = yield_mult.get((country_key, commodity, base_year))
                        if mult is not None:
                            break
                if mult is not None:
                    try:
                        val = float(val) * float(mult)
                    except Exception:
                        pass
            return float(val) if val and val > 0 else 0.0

        def _get_slaughter_ratio(country: str, commodity: str, year: int) -> float:
            ratio_item = ratio_item_by_emis.get(commodity)
            if not ratio_item:
                return 1.0
            y = int(year) if year is not None else base_year
            val = ratio_country_map.get((country, ratio_item, y), 0.0)
            if (not val or val <= 0) and y != base_year:
                val = ratio_country_map.get((country, ratio_item, base_year), 0.0)
            if (not val or val <= 0) and use_regional_aggregation:
                region = country_to_region.get(country)
                if region:
                    val = ratio_region_map.get((region, ratio_item, y), 0.0) or ratio_region_map.get((region, ratio_item, base_year), 0.0)
            if not val or val <= 0:
                val = ratio_world_map.get((ratio_item, y), 0.0) or ratio_world_map.get((ratio_item, base_year), 0.0)
            return float(val) if val and val > 0 else 1.0

        def get_ton_per_head(country: str, commodity: str, year: int) -> float:
            if use_yield_ratio:
                yield_val = _get_yield_t_per_head(country, commodity, year)
                if yield_val > 0:
                    ratio_val = _get_slaughter_ratio(country, commodity, year)
                    if ratio_val > 0:
                        return float(yield_val) * float(ratio_val)
            val = country_ratio_map.get((country, commodity), 0.0)
            if val and val > 0:
                return float(val)
            if use_regional_aggregation:
                region = country_to_region.get(country)
                if region:
                    val = region_ratio_map.get((region, commodity), 0.0)
                    if val and val > 0:
                        return float(val)
            val = world_ratio_map.get(commodity, 0.0)
            if val and val > 0:
                return float(val)
            return 0.0

        # Read standardized commodity-to-species mappings from dict_v3.
        # Node commodities use Item_Emis, e.g. 'Cattle, non-dairy' or 'Sheep, dairy'.
        # grassland_coef species use Item_Feed_Map, e.g. beef_cattle or dairy_sheep.
        commodity_to_species = {}
        try:
            emis_item_df = _read_excel(paths.dict_v3_path, sheet_name='Emis_item')
            for _, row in emis_item_df.iterrows():
                item_emis = row.get('Item_Emis')
                item_feed = row.get('Item_Feed_Map')
                if pd.notna(item_emis) and pd.notna(item_feed):
                    # Convert Item_Feed_Map formatting.
                    # dict_v3 may contain forms such as Beef_Cattle or Beef.
                    # Convert to grassland_coef species names: lowercase with underscores.
                    item_emis_clean = str(item_emis).strip()
                    item_feed_clean = str(item_feed).strip()
                    # Replace spaces with underscores and lowercase.
                    species_name = item_feed_clean.replace(' ', '_').lower()
                    commodity_to_species[item_emis_clean] = species_name
            
            print(f"[GRASSLAND_COEF] ? 从dict_v3加载了 {len(commodity_to_species)} 个 Item_Emis??Item_Feed_Map 映射")
            # Print the first 15 mapping examples.
            sample_mapping = list(commodity_to_species.items())[:15]
            print(f"[GRASSLAND_COEF] 映射样例:")
            for emis, feed in sample_mapping:
                print(f"  '{emis}' ?? '{feed}'")
        except Exception as e:
            print(f"[GRASSLAND_COEF] ?? 从dict_v3加载映射失败: {e}，使用默认映射")
            # Fall back to default mappings.
            commodity_to_species = {
                'Cattle, dairy': 'dairy_cattle',
                'Cattle, non-dairy': 'beef_cattle',
                'Buffalo, dairy': 'dairy_buffalo',
                'Buffalo, non-dairy': 'meat_buffalo',
                'Sheep, dairy': 'dairy_sheep',
                'Sheep, non-dairy': 'meat_sheep',
                'Goats, dairy': 'dairy_goat',
                'Goats, non-dairy': 'meat_goat',
                'Camel, dairy': 'dairy_camel',
                'Camel, non-dairy': 'meat_camel',
                'Swine': 'pigs',
                'Chickens, broilers': 'broilers',
                'Chickens, layers': 'layers',
                'Ducks': 'ducks',
                'Geese and guinea fowls': 'geese_guinea',
                'Turkeys': 'turkeys',
                'Horses': 'horse',
                'Asses': 'asses',
                'Mules and hinnies': 'mules_and_hinnies',
                'Llamas': 'llamas',
            }
        
        # Identify livestock directly from commodity_to_species keys.
        # These are actual commodity names; do not rely on the old LIVESTOCK_COMMODITIES constant.
        # commodity_to_species already includes all detailed livestock categories:
        # 'Cattle, dairy', 'Cattle, non-dairy', 'Buffalo, dairy', 'Buffalo, non-dairy', etc.
        livestock_commodities_actual = set(commodity_to_species.keys())
        
        def is_livestock(commodity: str) -> bool:
            """Identify livestock using commodity names present in the actual data."""
            return commodity in livestock_commodities_actual
        
        # Count matches.
        livestock_node_count = sum(1 for n in nodes if is_livestock(n.commodity))
        mapped_commodities = sum(1 for n in nodes if n.commodity in commodity_to_species)
        print(f"[GRASSLAND_COEF] 总节点数: {len(nodes)}, Livestock节点数: {livestock_node_count}, 可映射节点: {mapped_commodities}")
        
        # Check all Argentina livestock nodes across all years.
        argentina_all_livestock = {}
        for n in nodes:
            if n.country == 'Argentina' and is_livestock(n.commodity):
                if n.commodity not in argentina_all_livestock:
                    argentina_all_livestock[n.commodity] = []
                argentina_all_livestock[n.commodity].append(n.year)
        
        if argentina_all_livestock:
            print(f"\n[GRASSLAND_COEF] ?? Argentina所有livestock节点:")
            for comm, years in sorted(argentina_all_livestock.items()):
                print(f"  {comm:30s} | 年份数: {len(years)}, 年份范围: {min(years)}-{max(years)}")
        
        # Assign coefficients to nodes.
        assigned_count = 0
        skipped_count = 0
        # Collect diagnostics for Argentina 2020 only.
        argentina_2020_matched = []
        argentina_2020_unmatched = []
        # Collect nodes skipped due to missing ton_per_head.
        skipped_nodes = {}  # {(country, commodity): count}
        
        for n in nodes:
            if not is_livestock(n.commodity):
                continue
            
            # Get region/country.
            if use_regional_aggregation:
                region = country_to_region.get(n.country, n.country)
            else:
                region = n.country
            
            # Standardize species names.
            species = commodity_to_species.get(n.commodity, None)
            
            # Argentina 2020, using M49 keys
            if arg_m49 and n.country == arg_m49 and n.year == 2020:
                if species is None:
                    argentina_2020_unmatched.append((n.commodity, "commodity未在commodity_to_species字典中"))
                    continue
                
                # Print lookup keys on the first pass and record all commodity mappings.
                if len(argentina_2020_matched) == 0 and len(argentina_2020_unmatched) == 0:
                    print(f"\n[GRASSLAND_COEF] ?? Argentina({arg_m49}) commodity??species映射检查:")
                    print(f"  country_to_region['{arg_m49}'] = '{region}'")
                    # Print all Argentina livestock commodity-to-species mappings.
                    argentina_livestock = [nd for nd in nodes if nd.country == arg_m49 and nd.year == 2020 and is_livestock(nd.commodity)]
                    for nd in argentina_livestock[:10]:  # Print only the first ten.
                        sp = commodity_to_species.get(nd.commodity, 'NOT_FOUND')
                        print(f"  '{nd.commodity}' ?? '{sp}'")
                
                # Look up ha/head coefficients.
                coef_ha_per_head = grassland_coef_ha_per_head.get((region, species), 0.0)
                if coef_ha_per_head > 0:
                    coef_ha_per_head = coef_ha_per_head * _get_dm_mult(n.country, n.commodity, n.year)

                if coef_ha_per_head <= 0:
                    # Check whether failures arise from region names or species names.
                    region_exists = any(r == region for r, s in grassland_coef_ha_per_head.keys())
                    species_exists = any(s == species for r, s in grassland_coef_ha_per_head.keys())
                    detail = f"region存在={region_exists}, species存在={species_exists}"
                    argentina_2020_unmatched.append((n.commodity, f"grassland_coef中找不到('{region}', '{species}') | {detail}"))
                    continue

                # Convert to ha/tonne, preferring yield times slaughter ratio, then production/stock fallback.
                ton_per_head = get_ton_per_head(n.country, n.commodity, n.year)
                if ton_per_head > 0:
                    coef_ha_per_ton = coef_ha_per_head / ton_per_head
                    argentina_2020_matched.append((n.commodity, coef_ha_per_ton, f"species={species}, coef={coef_ha_per_head:.3f} ha/head, ton_per_head={ton_per_head:.3f} t/head"))
                    n.meta['grassland_coef'] = float(coef_ha_per_ton)
                    assigned_count += 1
                else:
                    argentina_2020_unmatched.append((n.commodity, "ton_per_head缺失：yield/屠宰率与产量-存栏比均无有效数据"))
                continue

            # Process other nodes normally.
            if species is None:
                continue
            
            # Look up ha/head coefficients.
            coef_ha_per_head = grassland_coef_ha_per_head.get((region, species), 0.0)
            if coef_ha_per_head > 0:
                coef_ha_per_head = coef_ha_per_head * _get_dm_mult(n.country, n.commodity, n.year)
            
            if coef_ha_per_head <= 0:
                continue

            # Convert to ha/tonne, preferring yield times slaughter ratio, then production/stock fallback.
            ton_per_head = get_ton_per_head(n.country, n.commodity, n.year)
            if ton_per_head > 0:
                coef_ha_per_ton = coef_ha_per_head / ton_per_head
                n.meta['grassland_coef'] = float(coef_ha_per_ton)
                assigned_count += 1
            else:
                # Record skipped nodes.
                key = (n.country, n.commodity)
                skipped_nodes[key] = skipped_nodes.get(key, 0) + 1
                skipped_count += 1
        
        print(f"[GRASSLAND_COEF] ? 已为 {assigned_count} 个livestock节点分配grassland系数")
        print(f"[GRASSLAND_COEF] ??  跳过 {skipped_count} 个节点（缺少ton_per_head数据）")
        
        # Summarize skipped nodes by country-commodity.
        if skipped_nodes:
            print(f"\n[GRASSLAND_COEF] ?? 跳过节点汇总（所有 {len(skipped_nodes)} 个country-commodity组合）:")
            sorted_skipped = sorted(skipped_nodes.items(), key=lambda x: (x[0][0], x[0][1]))  # Sort by country and commodity.
            for (country, commodity), count in sorted_skipped:
                print(f"  {country:25s} | {commodity:25s} | {count} 个年份")
        
        # Print detailed Argentina 2020 diagnostics.
        if arg_m49:
            print(f"\n[GRASSLAND_COEF] ?? Argentina({arg_m49}) 2020年匹配诊断:")
        else:
            print(f"\n[GRASSLAND_COEF] ?? Argentina 2020年匹配诊断:")
        print(f"  ? 匹配成功 ({len(argentina_2020_matched)} 个):")
        for comm, coef, detail in sorted(argentina_2020_matched):
            print(f"    {comm:25s} | {coef:10.3f} ha/ton | {detail}")
        
        print(f"\n  ? 匹配失败 ({len(argentina_2020_unmatched)} 个):")
        for comm, reason in sorted(argentina_2020_unmatched):
            print(f"    {comm:25s} | {reason}")
    
    except Exception as e:
        print(f"[GRASSLAND_COEF] ? 分配grassland系数失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


# demand elasticities (cross-price & income)
def load_demand_elasticities(elasticity_xlsx: str, universe: Universe) -> Tuple[Dict[str, float], Dict[str, float], Dict[Tuple[str, str], float], Dict[Tuple[str,str], Dict[str, float]]]:
    """Load demand-side elasticities from the processed elasticity workbook.
    Returns (income_by_country, pop_by_country, own_price_by_node, cross_price_by_node)."""
    eps_income: Dict[str, float] = {}
    eps_pop: Dict[str, float] = {}
    eps_own: Dict[Tuple[str, str], float] = {}
    cross: Dict[Tuple[str,str], Dict[str, float]] = {}
    if not os.path.exists(elasticity_xlsx):
        return eps_income, eps_pop, eps_own, cross
    sheet_names = set(_excel_sheet_names(elasticity_xlsx))
    emis_to_elast = _build_emis_to_elast_map()
    elast_to_emis_single = _build_elast_to_emis_map()
    elast_to_emis_multi = _build_elast_to_emis_multi()
    # demand income
    if 'Demand-Income' in sheet_names:
        df = _lc(_read_excel(elasticity_xlsx, sheet_name='Demand-Income'))
        eps_income = _build_country_elasticity(df, value_col='Elasticity_mean')
    # demand population
    if 'Demand-Population' in sheet_names:
        df = _lc(_read_excel(elasticity_xlsx, sheet_name='Demand-Population'))
        eps_pop = _build_country_elasticity(df, value_col='Elasticity_mean')
    # demand own-price
    if 'Demand-Own-Price' in sheet_names:
        df = _lc(_read_excel(elasticity_xlsx, sheet_name='Demand-Own-Price'))
        raw_own = _build_elasticity_map(df, value_col='Elasticity_mean')
        eps_own = {}
        for (country, elast_key), val in raw_own.items():
            emis_list = elast_to_emis_multi.get(elast_key, [elast_to_emis_single.get(elast_key, elast_key)])
            for emis_comm in emis_list:
                eps_own[(country, emis_comm)] = val
    # demand cross-price matrix
    if 'Demand_Cross_mean' in sheet_names:
        df = _lc(_read_excel(elasticity_xlsx, sheet_name='Demand_Cross_mean'))
        cross_raw = _build_cross_elasticity_map(df, commodity_filter=set(universe.commodities or []))
        # Remap elasticity row keys to emissions keys, allowing one-to-many mappings.
        cross = {}
        for (country, elast_key), row in cross_raw.items():
            emis_list = elast_to_emis_multi.get(elast_key, [elast_to_emis_single.get(elast_key, elast_key)])
            for emis_comm in emis_list:
                row_mapped = {}
                for k, v in row.items():
                    emis_k_list = elast_to_emis_multi.get(k, [elast_to_emis_single.get(k, k)])
                    for emis_k in emis_k_list:
                        row_mapped[emis_k] = v
                cross[(country, emis_comm)] = row_mapped
    return eps_income, eps_pop, eps_own, cross

def apply_demand_elasticities_to_nodes(nodes: List[Node], universe: Universe, elasticity_xlsx: str) -> None:
    eps_income, eps_pop, eps_own, cross = load_demand_elasticities(elasticity_xlsx, universe)
    emis_to_elast = _build_emis_to_elast_map()
    for n in nodes:
        # income/population elasticity
        setattr(n, 'eps_income_demand', float(eps_income.get(n.country, 0.0)))
        setattr(n, 'eps_pop_demand', float(eps_pop.get(n.country, 0.0)))
        elast_key = emis_to_elast.get(n.commodity, n.commodity)
        setattr(n, 'eps_demand', float(eps_own.get((n.country, n.commodity), eps_own.get((n.country, elast_key), 0.0))))
        # cross-price row dict
        eps_row = cross.get((n.country, n.commodity), cross.get((n.country, elast_key), {}))
        setattr(n, 'epsD_row', dict(eps_row))
def load_prices(csv_path: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    try:
        df_raw = _read_excel(csv_path, sheet_name=0)
    except Exception:
        df_raw = _read_excel(csv_path)
    df = _lc(df_raw)

    required = ['Item', 'Unit']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"load_prices: missing required columns {missing}")

    unit_norm = df['Unit'].astype(str).str.replace('–', '-').str.strip().str.lower()
    valid_units = {
        'int$ (2014-2016 const) per tonne',
        'int$ (2014-2016 const) per m3'
    }
    df = df[unit_norm.isin(valid_units)].copy()
    if 'Area' in df.columns:
        df = df[df['Area'].astype(str).str.strip().str.lower() == 'world']
    if df.empty:
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    year_cols = [c for c in df.columns if isinstance(c, str) and c.strip().startswith('Y') and c.strip()[1:].isdigit()]
    if not year_cols:
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
    mapping_df = _lc(_read_excel(dict_path, sheet_name='Emis_item'))
    price_col = 'Item_Price_Map'
    emis_col = 'Item_Emis'
    price_map: Dict[str, str] = {}
    for r in mapping_df[[price_col, emis_col]].dropna().itertuples(index=False):
        price_name = str(getattr(r, price_col)).strip()
        emis_name = str(getattr(r, emis_col)).strip()
        if not price_name or not emis_name:
            continue
        if price_name.lower() in {'nan', 'no'} or emis_name.lower() in {'nan', 'no'}:
            continue
        price_map[price_name.lower()] = emis_name

    def _map_item(name: str) -> Optional[str]:
        if name is None:
            return None
        key = str(name).strip().lower()
        return price_map.get(key)

    df['commodity'] = df['Item'].map(_map_item)
    df = df[df['commodity'].isin(universe.commodities)]
    if df.empty:
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    long_df = df.melt(id_vars=['commodity'], value_vars=year_cols, var_name='year', value_name='price')
    long_df['year'] = pd.to_numeric(long_df['year'].astype(str).str.strip().str.lstrip('Y'), errors='coerce')
    long_df['price'] = pd.to_numeric(long_df['price'], errors='coerce')
    long_df = long_df.dropna(subset=['year', 'price'])
    long_df['year'] = long_df['year'].astype(int)

    rows: List[Dict[str, Any]] = []
    for r in long_df.itertuples(index=False):
        year = int(r.year)
        price_val = float(r.price)
        commodity = str(r.commodity)
        for country in universe.countries:
            rows.append({
                'country': country,
                'iso3': universe.iso3_by_country.get(country, ''),
                'year': year,
                'commodity': commodity,
                'price': price_val
            })

    if not rows:
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    out = pd.DataFrame(rows)
    return out[['country','iso3','year','commodity','price']]


def load_price_wedge(
    xlsx_path: str,
    universe: Universe,
    use_regional_aggregation: bool,
    *,
    load_region_comm_year: bool = False,
    load_region_comm: bool = False,
    load_region: bool = False,
) -> Dict[str, Dict]:
    logger = logging.getLogger(__name__)
    out = {
        'by_region_comm_year': {},
        'by_region_comm': {},
        'by_region': {},
    }
    if not (load_region_comm_year or load_region_comm or load_region):
        return out
    if not os.path.exists(xlsx_path):
        logger.warning("[PRICE_WEDGE] file missing: %s", xlsx_path)
        return out

    try:
        xls = pd.ExcelFile(xlsx_path)
    except Exception as exc:
        logger.warning("[PRICE_WEDGE] failed to open: %s (%s)", xlsx_path, exc)
        return out

    comm_set = set(universe.commodities or [])
    country_set = set(universe.countries or [])
    region_col = 'Region_market_agg' if use_regional_aggregation else 'M49_Country_Code'

    def _map_region(val: Any) -> Optional[str]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if use_regional_aggregation:
            r = str(val).strip()
            return r or None
        return _norm_m49(val) or None

    def _read_sheet(sheet: str) -> Optional[pd.DataFrame]:
        if sheet not in xls.sheet_names:
            logger.warning("[PRICE_WEDGE] sheet missing: %s", sheet)
            return None
        try:
            return _lc(_read_excel(xls, sheet_name=sheet))
        except Exception as exc:
            logger.warning("[PRICE_WEDGE] read failed: %s (%s)", sheet, exc)
            return None

    if load_region_comm_year:
        sheet = 'price_wedge_by_region_com_year' if use_regional_aggregation else 'price_wedge_by_country_com_year'
        df = _read_sheet(sheet)
        if df is not None:
            needed = {region_col, 'Item_Emis'}
            if not needed.issubset(df.columns):
                logger.warning("[PRICE_WEDGE] missing columns in %s: %s", sheet, sorted(needed - set(df.columns)))
            else:
                year_cols: List[Tuple[str, int]] = []
                for col in df.columns:
                    m = re.match(r'^Price_wedge_Y(\\d{4})$', str(col).strip())
                    if m:
                        year_cols.append((col, int(m.group(1))))
                if not year_cols:
                    logger.warning("[PRICE_WEDGE] no year columns in %s", sheet)
                else:
                    for col, _ in year_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    loaded = 0
                    skipped = 0
                    for row in df.itertuples(index=False):
                        region = _map_region(getattr(row, region_col, None))
                        comm = str(getattr(row, 'Item_Emis', '')).strip()
                        if not region or (not use_regional_aggregation and country_set and region not in country_set):
                            skipped += 1
                            continue
                        if not comm or (comm_set and comm not in comm_set):
                            skipped += 1
                            continue
                        for col, year in year_cols:
                            val = getattr(row, col, None)
                            if val is None or (isinstance(val, float) and pd.isna(val)):
                                continue
                            try:
                                v = float(val)
                            except Exception:
                                continue
                            if not np.isfinite(v):
                                continue
                            out['by_region_comm_year'][(region, comm, year)] = v
                            loaded += 1
                    logger.info("[PRICE_WEDGE] %s loaded=%d skipped=%d", sheet, loaded, skipped)

    if load_region_comm:
        sheet = 'price_wedge_by_region_com' if use_regional_aggregation else 'price_wedge_by_country_com'
        df = _read_sheet(sheet)
        if df is not None:
            needed = {region_col, 'Item_Emis', 'Price_wedge'}
            if not needed.issubset(df.columns):
                logger.warning("[PRICE_WEDGE] missing columns in %s: %s", sheet, sorted(needed - set(df.columns)))
            else:
                df['Price_wedge'] = pd.to_numeric(df['Price_wedge'], errors='coerce')
                loaded = 0
                skipped = 0
                for row in df.itertuples(index=False):
                    region = _map_region(getattr(row, region_col, None))
                    comm = str(getattr(row, 'Item_Emis', '')).strip()
                    if not region or (not use_regional_aggregation and country_set and region not in country_set):
                        skipped += 1
                        continue
                    if not comm or (comm_set and comm not in comm_set):
                        skipped += 1
                        continue
                    val = getattr(row, 'Price_wedge', None)
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    if not np.isfinite(v):
                        continue
                    out['by_region_comm'][(region, comm)] = v
                    loaded += 1
                logger.info("[PRICE_WEDGE] %s loaded=%d skipped=%d", sheet, loaded, skipped)

    if load_region:
        sheet = 'price_wedge_by_region' if use_regional_aggregation else 'price_wedge_by_country'
        df = _read_sheet(sheet)
        if df is not None:
            needed = {region_col, 'Price_wedge'}
            if not needed.issubset(df.columns):
                logger.warning("[PRICE_WEDGE] missing columns in %s: %s", sheet, sorted(needed - set(df.columns)))
            else:
                df['Price_wedge'] = pd.to_numeric(df['Price_wedge'], errors='coerce')
                loaded = 0
                skipped = 0
                for row in df.itertuples(index=False):
                    region = _map_region(getattr(row, region_col, None))
                    if not region or (not use_regional_aggregation and country_set and region not in country_set):
                        skipped += 1
                        continue
                    val = getattr(row, 'Price_wedge', None)
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    if not np.isfinite(v):
                        continue
                    out['by_region'][region] = v
                    loaded += 1
                logger.info("[PRICE_WEDGE] %s loaded=%d skipped=%d", sheet, loaded, skipped)

    return out


def load_armington_elasticity(
    xlsx_path: Optional[str],
    universe: Universe,
    *,
    sheet: str = 'armington_sigma',
    use_regional_aggregation: bool = False,
) -> Dict[Any, float]:
    logger = logging.getLogger(__name__)
    if not xlsx_path or not os.path.exists(xlsx_path):
        logger.warning("[ARMINGTON] file missing: %s", xlsx_path)
        return {}
    try:
        df = _lc(_read_excel(xlsx_path, sheet_name=sheet))
    except Exception as exc:
        logger.warning("[ARMINGTON] read failed: %s (%s)", xlsx_path, exc)
        return {}

    item_col = None
    for cand in ('Item_Emis', 'Item', 'commodity', 'Commodity'):
        if cand in df.columns:
            item_col = cand
            break
    if item_col is None:
        logger.warning("[ARMINGTON] missing item column in %s", xlsx_path)
        return {}

    sigma_col = None
    for cand in ('Armington_sigma', 'sigma', 'Sigma', 'armington_sigma'):
        if cand in df.columns:
            sigma_col = cand
            break
    if sigma_col is None:
        logger.warning("[ARMINGTON] missing sigma column in %s", xlsx_path)
        return {}

    comm_set = set(universe.commodities or [])
    region_col = 'Region_market_agg' if use_regional_aggregation else 'M49_Country_Code'
    has_region = region_col in df.columns

    out: Dict[Any, float] = {}
    loaded = 0
    skipped = 0
    for row in df.itertuples(index=False):
        comm = str(getattr(row, item_col, '')).strip()
        if not comm or (comm_set and comm not in comm_set):
            skipped += 1
            continue
        sigma_val = getattr(row, sigma_col, None)
        try:
            sigma = float(sigma_val)
        except Exception:
            skipped += 1
            continue
        if not np.isfinite(sigma):
            skipped += 1
            continue
        if has_region:
            region_raw = getattr(row, region_col, None)
            if use_regional_aggregation:
                region = str(region_raw).strip() if region_raw is not None else ''
            else:
                region = _norm_m49(region_raw) or ''
            if region:
                out[(region, comm)] = sigma
                loaded += 1
                continue
        out[comm] = sigma
        loaded += 1

    logger.info(
        "[ARMINGTON] loaded=%d skipped=%d (use_regional_aggregation=%s, region_col=%s)",
        loaded,
        skipped,
        use_regional_aggregation,
        region_col if has_region else 'none',
    )
    return out

# attach emission factors via FAO modules
def attach_emission_factors_from_fao_modules(nodes: List[Node], params_wide: Optional[pd.DataFrame],
                                             production_df: pd.DataFrame,
                                             crop_activity: Dict[str, pd.DataFrame],
                                             livestock_activity: Dict[str, pd.DataFrame],
                                             soils_activity: Dict[str, pd.DataFrame],
                                             forest_activity: Optional[Dict[str, pd.DataFrame]],
                                             module_paths: Dict[str, str]) -> None:
    """
    Call uploaded *_fao.py modules by node (i,j,t), using a placeholder interface to demonstrate n.e0_by_proc assignment.
    Actual interfaces depend on the supplied modules; call available functions, otherwise leave e0_by_proc empty.
    """
    import importlib.util
    def _safe_import(path):
        spec = importlib.util.spec_from_file_location("mod_"+os.path.basename(path).replace('.py',''), path)
        if spec and spec.loader:
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
        return None
    mods = {k:_safe_import(v) for k,v in module_paths.items() if os.path.exists(v)}

    # Demonstration: call get_default_intensity(process, commodity) when a module exposes it.
    # Never overwrite existing process intensities; empty placeholder returns would erase valid e0_by_proc values.
    for n in nodes:
        e = dict(getattr(n, 'e0_by_proc', {}) or {})
        for p in []:  # To initialize placeholders from dict_v3 processes: for p in universe.processes.
            pass
        # Try module interfaces, e.g. default commodity intensities from gfe/gce/gle.
        for key, m in mods.items():
            if not m: continue
            for cand in ['get_default_intensity','get_emission_intensity','intensity_for']:
                f = getattr(m, cand, None)
                if callable(f):
                    try:
                        val = float(f(process='ALL', commodity=n.commodity))  # Assumed interface
                        if val>0:
                            e[key.replace('_module_fao.py','')] = val
                    except Exception:
                        pass
        n.e0_by_proc = e
def load_income_multipliers_from_sspdb(xlsx_path: str, scenario: str, universe: Universe) -> Dict[Tuple[str,int], float]:
    """Load per-country GDP change ratios (relative to 2020) from SSPDB_future_GDP_with_change_ratio.xlsx."""
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(xlsx_path):
        return out
    try:
        df = _lc(_read_excel(xlsx_path, sheet_name='change_ratio_country'))
    except Exception:
        return out
    if 'M49_Country_Code' not in df.columns:
        return out
    df = df[df['SCENARIO'].astype(str) == str(scenario)]
    value_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit()]
    if not value_cols:
        return out
    for r in df.itertuples(index=False):
        code = _parse_m49_code(getattr(r, 'M49_Country_Code'))
        if not code:
            continue
        m49 = code
        country = m49
        for col in value_cols:
            year = int(col[1:])
            if year not in universe.years:
                continue
            val = pd.to_numeric(getattr(r, col), errors='coerce')
            if not np.isfinite(val):
                continue
            out[(country, year)] = float(val)
    return out

# FAO modules runner (lme integration)
def run_fao_modules_and_cache(nodes: List[Node], *, livestock_stock_df: pd.DataFrame, module_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Run selected *_emissions_module_fao modules (notably lme_manure_module_fao) and cache results.
    Returns {module_key: df}. Keeps a copy under LAST_FAO_RUNS.
    """
    import importlib.util
    from config_paths import get_input_base, get_src_base

    def _safe_import(path: str):
        spec = importlib.util.spec_from_file_location("mod_"+os.path.basename(path).replace('.py',''), path)
        if spec and spec.loader:
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
        return None

    out: Dict[str, pd.DataFrame] = {}
    try:
        mods = {os.path.basename(k): _safe_import(v) for k, v in (module_paths or {}).items() if os.path.exists(v)}
    except Exception:
        mods = {}

    # lme manure
    for fname, mod in mods.items():
        if not mod: continue
        if 'lme_manure_module_fao' in fname:
            try:
                load_wide = getattr(mod, 'load_parameters_wide', None)
                run_wide = getattr(mod, 'run_lme_from_wide', None)
                if callable(load_wide) and callable(run_wide):
                    path_params = os.path.join(get_src_base(), 'Livestock_Manure_parameters.xlsx')
                    P = load_wide(path_params) if os.path.exists(path_params) else None
                    pop = livestock_stock_df.copy() if isinstance(livestock_stock_df, pd.DataFrame) else pd.DataFrame()
                    if P is not None and len(pop):
                        z = _lc(pop)
                        c_cty = _find_col(z, ['country'])
                        c_year= _find_col(z, ['year'])
                        c_comm= _find_col(z, ['commodity'])
                        c_head= _find_col(z, ['headcount'])
                        def _m49_to_int(val: Any) -> str:
                            m49 = _norm_m49(val)
                            return m49 if m49 else ''
                        z['AreaCode'] = z[c_cty].apply(_m49_to_int)
                        z['ItemName'] = z[c_comm]
                        z = z.rename(columns={c_year:'year', c_head:'head'})[['AreaCode','year','ItemName','head']]
                        lme_df = run_wide(P, z, years=sorted(z['year'].unique().tolist()), itemname_col='ItemName', head_col='head')
                        out['lme'] = lme_df
                        try:
                            global LAST_FAO_RUNS
                            LAST_FAO_RUNS['lme'] = lme_df
                        except Exception:
                            pass
            except Exception:
                pass
    return out
@dataclass
class EmisItemMappings:
    production_by_item: Dict[str, str]
    fertilizer_by_item: Dict[str, str]
    yield_item_to_comm: Dict[str, str]
    yield_element_by_item: Dict[str, str]
    yield_unit_by_item: Dict[str, str]
    area_item_to_comm: Dict[str, str]
    area_element_by_item: Dict[str, str]
    area_unit_by_item: Dict[str, str]
    slaughter_item_to_comm: Dict[str, str]
    slaughter_element_by_item: Dict[str, str]
    slaughter_unit_by_item: Dict[str, str]
    stock_item_to_comm: Dict[str, str]
    stock_element_by_item: Dict[str, str]
    stock_unit_by_item: Dict[str, str]
    elasticity_by_item: Dict[str, str]
    feed_item_to_comm: Dict[str, str]

def load_emis_item_mappings(xls_path: str) -> EmisItemMappings:
    """Parse dict_v3.xlsx Emis_item sheet for multi-domain item mappings and units.
    Returns a structured mapping object for consistent FAOSTAT alignment.
    """
    if not os.path.exists(xls_path):
        return EmisItemMappings({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    xls = pd.ExcelFile(xls_path)
    df = _lc(_read_excel(xls, 'Emis_item'))

    # columns
    def col(name: str) -> Optional[str]:
        return name if name in df.columns else None

    c_emis = col('Item_Emis')  # The model commodity name
    c_prod = col('Item_Production_Map')
    c_fert = col('Item_Fertilizer_Map')
    c_yield = col('Item_Yield_Map')
    c_yield_elem = col('Item_Yield_Element')
    c_yield_unit = col('Item_Yield_Unit')
    c_area = col('Item_Area_Map')
    c_area_elem = col('Item_Area_Element')
    c_area_unit = col('Item_Area_Unit')
    c_sl_map = col('Item_Slaughtered_Map')
    c_sl_elem = col('Item_Slaughtered_Element')
    c_sl_unit = col('Item_Slaughtered_Unit')
    c_stock_map = col('Item_Stock_Map')
    c_stock_elem = col('Item_Stock_Element')
    c_stock_unit = col('Item_Stock_Unit')
    c_elast = col('Item_Elasticity_Map')
    c_feed = col('Item_Feed_Map')

    def _clean(val: Any) -> Optional[str]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        if not s or s.lower() in {'nan', 'no'}:
            return None
        return s

    production_by_item: Dict[str, str] = {}
    fertilizer_by_item: Dict[str, str] = {}
    yield_item_to_comm: Dict[str, str] = {}
    yield_element_by_item: Dict[str, str] = {}
    yield_unit_by_item: Dict[str, str] = {}
    area_item_to_comm: Dict[str, str] = {}
    area_element_by_item: Dict[str, str] = {}
    area_unit_by_item: Dict[str, str] = {}
    slaughter_item_to_comm: Dict[str, str] = {}
    slaughter_element_by_item: Dict[str, str] = {}
    slaughter_unit_by_item: Dict[str, str] = {}
    stock_item_to_comm: Dict[str, str] = {}
    stock_element_by_item: Dict[str, str] = {}
    stock_unit_by_item: Dict[str, str] = {}
    elasticity_by_item: Dict[str, str] = {}
    feed_item_to_comm: Dict[str, str] = {}
    feed_item_priority = {
        'beef_cattle': ['Cattle, non-dairy', 'Mules and hinnies'],
        'horse': ['Horses', 'Llamas'],
    }

    for r in df.itertuples(index=False):
        # Get the model commodity name (Item_Emis) - this is the target for all mappings
        commodity = _clean(getattr(r, c_emis)) if c_emis else None
        if not commodity:
            continue  # Skip rows without Item_Emis

        # Map FAOSTAT Item names to model Item_Emis
        prod_item = _clean(getattr(r, c_prod)) if c_prod else None
        if prod_item and commodity:
            production_by_item[prod_item] = commodity

        fert_item = _clean(getattr(r, c_fert)) if c_fert else None
        if fert_item and commodity:
            fertilizer_by_item[fert_item] = commodity

        yield_item = _clean(getattr(r, c_yield)) if c_yield else None
        if yield_item and commodity:
            yield_item_to_comm[yield_item] = commodity
            elem = _clean(getattr(r, c_yield_elem)) if c_yield_elem else None
            if elem:
                yield_element_by_item[yield_item] = elem
            unit = _clean(getattr(r, c_yield_unit)) if c_yield_unit else None
            if unit:
                yield_unit_by_item[yield_item] = unit

        area_item = _clean(getattr(r, c_area)) if c_area else None
        if area_item and commodity:
            area_item_to_comm[area_item] = commodity
            elem = _clean(getattr(r, c_area_elem)) if c_area_elem else None
            if elem:
                area_element_by_item[area_item] = elem
            unit = _clean(getattr(r, c_area_unit)) if c_area_unit else None
            if unit:
                area_unit_by_item[area_item] = unit

        slaughter_item = _clean(getattr(r, c_sl_map)) if c_sl_map else None
        if slaughter_item and commodity:
            slaughter_item_to_comm[slaughter_item] = commodity
            elem = _clean(getattr(r, c_sl_elem)) if c_sl_elem else None
            if elem:
                slaughter_element_by_item[slaughter_item] = elem
            unit = _clean(getattr(r, c_sl_unit)) if c_sl_unit else None
            if unit:
                slaughter_unit_by_item[slaughter_item] = unit

        stock_item = _clean(getattr(r, c_stock_map)) if c_stock_map else None
        if stock_item and commodity:
            stock_item_to_comm[stock_item] = commodity
            elem = _clean(getattr(r, c_stock_elem)) if c_stock_elem else None
            if elem:
                stock_element_by_item[stock_item] = elem
            unit = _clean(getattr(r, c_stock_unit)) if c_stock_unit else None
            if unit:
                stock_unit_by_item[stock_item] = unit

        elast_item = _clean(getattr(r, c_elast)) if c_elast else None
        if elast_item and commodity:
            elasticity_by_item[elast_item] = commodity

        feed_item = _clean(getattr(r, c_feed)) if c_feed else None
        if feed_item and commodity:
            existing = feed_item_to_comm.get(feed_item)
            if existing:
                priority = feed_item_priority.get(feed_item)
                if priority:
                    def _rank(name: str) -> int:
                        return priority.index(name) if name in priority else len(priority)
                    if _rank(commodity) < _rank(existing):
                        feed_item_to_comm[feed_item] = commodity
            else:
                feed_item_to_comm[feed_item] = commodity

    return EmisItemMappings(
        production_by_item=production_by_item,
        fertilizer_by_item=fertilizer_by_item,
        yield_item_to_comm=yield_item_to_comm,
        yield_element_by_item=yield_element_by_item,
        yield_unit_by_item=yield_unit_by_item,
        area_item_to_comm=area_item_to_comm,
        area_element_by_item=area_element_by_item,
        area_unit_by_item=area_unit_by_item,
        slaughter_item_to_comm=slaughter_item_to_comm,
        slaughter_element_by_item=slaughter_element_by_item,
        slaughter_unit_by_item=slaughter_unit_by_item,
        stock_item_to_comm=stock_item_to_comm,
        stock_element_by_item=stock_element_by_item,
        stock_unit_by_item=stock_unit_by_item,
        elasticity_by_item=elasticity_by_item,
        feed_item_to_comm=feed_item_to_comm,
    )
def _match_element(row_element: Any, target: Optional[str]) -> bool:
    if target is None:
        return False
    if row_element is None or (isinstance(row_element, float) and pd.isna(row_element)):
        return False
    return str(row_element).strip().lower() == str(target).strip().lower()


def _convert_yield_unit(value: float, unit: Optional[str]) -> float:
    if not np.isfinite(value):
        return np.nan
    u = (unit or '').strip().lower()
    if u in {'kg/ha', 'kg ha-1', 'kilogram per hectare'}:
        return value / 1000.0
    if u in {'hg/ha', 'hectogram per hectare'}:
        return value / 100.0
    if u in {'t/ha', 'tonne per hectare', 'tonnes per hectare', 'ton/ha'}:
        return value
    return value


def _convert_carcass_unit(value: float, unit: Optional[str]) -> float:
    if not np.isfinite(value):
        return np.nan
    u = (unit or '').strip().lower()
    if u in {'kg/an', 'kg/animal', 'kilogram per animal'}:
        return value / 1000.0
    if u in {'t/an', 'tonne per animal'}:
        return value
    return value


def _convert_livestock_yield_unit(value: float, unit: Optional[str], item_raw: str) -> float:
    """
    Convert livestock yield units to t/head.
    Milk/egg source units may be kg/head, t/head, hg/head, etc.
    """
    if not np.isfinite(value):
        return np.nan
    u = (unit or '').strip().lower()
    item_lower = (item_raw or '').lower()
    
    # Milk: kg -> t.
    if 'milk' in item_lower:
        if u in {'kg/an', 'kg/animal', 'kilogram per animal', 'kg'}:
            return value / 1000.0
        if u in {'hg/an', 'hg/animal', 'hectogram per animal', 'hg'}:
            return value / 10000.0
        if u in {'t/an', 'tonne per animal', 't', 'tonne'}:
            return value
    
    # Eggs: convert pieces/head with an assumed average weight, or use values directly.
    # Eggs are usually calculated from production/stock; yields here may already be standardized.
    if 'egg' in item_lower:
        if u in {'kg/an', 'kg/animal', 'kilogram per animal', 'kg'}:
            return value / 1000.0
        if u in {'t/an', 'tonne per animal', 't', 'tonne'}:
            return value
        # Pieces: assume approximately 0.06 kg per egg.
        if u in {'head', 'pieces', 'number'}:
            return value * 0.06 / 1000.0
    
    # Default: kg -> t.
    if 'kg' in u:
        return value / 1000.0
    
    return value


def _convert_area_unit(value: float, unit: Optional[str]) -> float:
    if not np.isfinite(value):
        return np.nan
    u = (unit or '').strip().lower()
    if u in {'1000 ha', 'thousand ha', '1000ha'}:
        return value * 1000.0
    return value


def _convert_production_unit(value: float, unit: Optional[str]) -> float:
    if not np.isfinite(value):
        return np.nan
    u = (unit or '').strip().lower()
    if ('1000' in u or 'thousand' in u) and ('t' in u or 'ton' in u):
        return value * 1000.0
    if u in {'kg', 'kilogram', 'kilograms'} or u.endswith(' kg'):
        return value / 1000.0
    return value


def build_faostat_production_indicators(production_csv: str,
                                        universe: Universe,
                                        maps: Optional[EmisItemMappings] = None,
                                        fbs_csv: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    # All DataFrames include M49_Country_Code as a unique identifier.
    empty = {
        'production': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','production_t']),
        'yield': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','yield_t_per_ha']),
        'area': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','area_ha']),
        'slaughter': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','slaughter_head']),
        'livestock_yield': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','yield_t_per_head']),
        'stock': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','stock_head']),
    }
    if not os.path.exists(production_csv):
        print(f"[ERROR] 关键文件不存在: {production_csv}")
        raise FileNotFoundError(f"Production CSV file not found: {production_csv}")
    if maps is None:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
    # Map Item_Production_Map -> Production_file_source (if provided in dict_v3)
    prod_source_by_item: Dict[str, str] = {}
    try:
        emis_df = _lc(_read_excel(os.path.join(get_src_base(), 'dict_v3.xlsx'), 'Emis_item'))
        if 'Item_Production_Map' in emis_df.columns and 'Production_file_source' in emis_df.columns:
            for r in emis_df[['Item_Production_Map', 'Production_file_source']].itertuples(index=False):
                item = str(getattr(r, _tuple_field('Item_Production_Map'))).strip()
                src = str(getattr(r, _tuple_field('Production_file_source'))).strip()
                if item and item.lower() not in {'nan', 'no'} and src and src.lower() not in {'nan', 'no'}:
                    prod_source_by_item[item] = src
    except Exception:
        prod_source_by_item = {}
    fbs_source_items = {
        item for item, src in prod_source_by_item.items()
        if src == 'FoodBalanceSheets_E_All_Data_NOFLAG.csv'
    }

    df_raw = _read_csv(production_csv)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val = _find_col(df, ['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])
    if not all([c_area, c_year, c_item, c_elem, c_val]):
        return empty

    keep_cols = [c_area, c_year, c_item, c_elem, c_val] + ([c_unit] if c_unit else [])
    if 'M49_Country_Code' in df.columns:
        keep_cols.append('M49_Country_Code')
    z = df[keep_cols].copy()
    rename_map = {c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'}
    if c_unit:
        rename_map[c_unit] = 'unit'
    z = z.rename(columns=rename_map)
    z['value'] = pd.to_numeric(z['value'], errors='coerce')
    z = z.dropna(subset=['value'])

    z = _attach_country_from_m49(df, z, universe, context=f"FAOSTAT production ({production_csv})")
    # Preserve M49_Country_Code and ensure index alignment.
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        # Use .loc[] to ensure index alignment.
        z['M49_Country_Code'] = df.loc[z.index, 'M49_Country_Code'].values
    
    # Normalize M49 to an apostrophe plus three digits.
    if 'M49_Country_Code' in z.columns:
        def _format_m49_quote(val):
            """Format M49 as an apostrophe plus three digits."""
            if pd.isna(val):
                return val
            return _norm_m49(val)
        z['M49_Country_Code'] = z['M49_Country_Code'].apply(_format_m49_quote)
    
    z['country'] = z['country'].astype(str).str.strip()
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    z = z.dropna(subset=['iso3'])
    z['iso3'] = z['iso3'].astype(str)
    z['year'] = pd.to_numeric(z['year'], errors='coerce')
    z = z.dropna(subset=['year'])
    z['year'] = z['year'].astype(int)
    if 'unit' not in z.columns:
        z['unit'] = ''

    def _group(df_subset: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if df_subset.empty:
            base_cols = ['M49_Country_Code','country','iso3','year','commodity', value_col]
            return pd.DataFrame(columns=base_cols)
        
        # Retain M49_Country_Code if present.
        group_cols = ['country','iso3','year','commodity']
        if 'M49_Country_Code' in df_subset.columns:
            group_cols = ['M49_Country_Code'] + group_cols
        
        g = df_subset.groupby(group_cols, as_index=False)['value'].sum()
        g = g.rename(columns={'value': value_col})
        # WARNING: do not filter using universe.commodities!
        # Slaughter, stock, etc. use different Item_XXX_Map names.
        # Commodity was already validated by maps.xxx_item_to_comm.
        return g

    # Production
    prod = z[z['element'].str.contains('Production', case=False, na=False)].copy()
    if fbs_source_items and fbs_csv and os.path.exists(fbs_csv):
        prod = prod[~prod['item_raw'].isin(fbs_source_items)]
    prod['commodity'] = prod['item_raw'].map(maps.production_by_item).fillna(prod['item_raw'])
    production_df = _group(prod, 'production_t')
    # Production from FBS for selected items
    if fbs_csv and fbs_source_items and os.path.exists(fbs_csv):
        fbs_raw = _read_fbs_table(fbs_csv)
        fbs_raw = _filter_select_rows(fbs_raw)
        raw_item_col = _find_col(fbs_raw, ['Item'])
        raw_element_col = _find_col(fbs_raw, ['Element'])
        item_mask = fbs_raw[raw_item_col].astype(str).str.strip().isin(fbs_source_items)
        element_mask = (
            fbs_raw[raw_element_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq('production')
        )
        fbs_raw = fbs_raw.loc[item_mask & element_mask].copy()
        fbs = _lc(_faostat_wide_to_long(fbs_raw))
        c_area_f = _find_col(fbs, ['Area'])
        c_year_f = _find_col(fbs, ['Year'])
        c_item_f = _find_col(fbs, ['Item'])
        c_elem_f = _find_col(fbs, ['Element'])
        c_val_f = _find_col(fbs, ['Value'])
        c_unit_f = _maybe_find_col(fbs, ['Unit'])
        if all([c_area_f, c_year_f, c_item_f, c_elem_f, c_val_f]):
            keep_cols = [c_area_f, c_year_f, c_item_f, c_elem_f, c_val_f]
            if c_unit_f:
                keep_cols.append(c_unit_f)
            if 'M49_Country_Code' in fbs.columns:
                keep_cols.append('M49_Country_Code')
            fbs_z = fbs[keep_cols].copy()
            rename_map = {
                c_area_f: 'area',
                c_year_f: 'year',
                c_item_f: 'item_raw',
                c_elem_f: 'element',
                c_val_f: 'value',
            }
            if c_unit_f:
                rename_map[c_unit_f] = 'unit'
            fbs_z = fbs_z.rename(columns=rename_map)
            fbs_z['value'] = pd.to_numeric(fbs_z['value'], errors='coerce')
            fbs_z = fbs_z.dropna(subset=['value'])
            if 'unit' not in fbs_z.columns:
                fbs_z['unit'] = ''
            fbs_z['value'] = fbs_z.apply(
                lambda r: _convert_production_unit(float(r['value']), r['unit']),
                axis=1
            )
            fbs_z = _attach_country_from_m49(fbs, fbs_z, universe, context=f"FBS production ({fbs_csv})")
            if 'M49_Country_Code' in fbs.columns and 'M49_Country_Code' not in fbs_z.columns:
                fbs_z['M49_Country_Code'] = fbs['M49_Country_Code']
            if 'M49_Country_Code' in fbs_z.columns:
                def _format_m49_quote_fbs(val):
                    if pd.isna(val):
                        return val
                    return _norm_m49(val)
                fbs_z['M49_Country_Code'] = fbs_z['M49_Country_Code'].apply(_format_m49_quote_fbs)
            fbs_z['country'] = fbs_z['country'].astype(str).str.strip()
            fbs_z = fbs_z[fbs_z['country'].isin(universe.countries)]
            fbs_z['iso3'] = fbs_z['country'].map(universe.iso3_by_country)
            fbs_z = fbs_z.dropna(subset=['iso3'])
            fbs_z['iso3'] = fbs_z['iso3'].astype(str)
            fbs_z['year'] = pd.to_numeric(fbs_z['year'], errors='coerce')
            fbs_z = fbs_z.dropna(subset=['year'])
            fbs_z['year'] = fbs_z['year'].astype(int)
            fbs_z['item_raw'] = fbs_z['item_raw'].astype(str).str.strip()
            fbs_z = fbs_z[fbs_z['item_raw'].isin(fbs_source_items)]
            fbs_z['element_norm'] = fbs_z['element'].astype(str).str.strip().str.lower()
            fbs_z = fbs_z[fbs_z['element_norm'] == 'production']
            fbs_z['commodity'] = fbs_z['item_raw'].map(maps.production_by_item).fillna(fbs_z['item_raw'])
            fbs_prod_df = _group(fbs_z, 'production_t')
            if not fbs_prod_df.empty:
                production_df = pd.concat([production_df, fbs_prod_df], ignore_index=True)

    # Area
    area_items = maps.area_item_to_comm or maps.production_by_item
    area = z[z['item_raw'].isin(area_items.keys())].copy()
    area['target_elem'] = area['item_raw'].map(maps.area_element_by_item)
    area['element_norm'] = area['element'].astype(str).str.strip().str.lower()
    area['target_norm'] = area['target_elem'].fillna('').astype(str).str.strip().str.lower()
    mask_specific = area['target_elem'].notna() & (area['element_norm'] == area['target_norm'])
    mask_default = area['target_elem'].isna() & area['element_norm'].str.contains('area harvested', na=False)
    area = area[mask_specific | mask_default]
    area['commodity'] = area['item_raw'].map(area_items).fillna(area['item_raw'])
    area['value'] = area.apply(lambda r: _convert_area_unit(float(r['value']), r['unit']), axis=1)
    area_df = _group(area, 'area_ha')

    # Yield (crop)
    yields = z[z['item_raw'].isin(maps.yield_item_to_comm.keys())].copy()
    yields['target_elem'] = yields['item_raw'].map(maps.yield_element_by_item)
    yields['element_norm'] = yields['element'].astype(str).str.strip().str.lower()
    yields['target_norm'] = yields['target_elem'].fillna('').astype(str).str.strip().str.lower()
    mask_specific = yields['target_elem'].notna() & (yields['element_norm'] == yields['target_norm'])
    mask_default = yields['target_elem'].isna() & yields['element_norm'].str.contains('yield', na=False)
    yields = yields[mask_specific | mask_default]
    yields['commodity'] = yields['item_raw'].map(maps.yield_item_to_comm).fillna(yields['item_raw'])
    yields['value'] = yields.apply(lambda r: _convert_yield_unit(float(r['value']), r['unit']), axis=1)
    yield_df = _group(yields, 'yield_t_per_ha')

    # Livestock yields for meat/milk/eggs
    # Meat: Yield/Carcass Weight.
    # Milk/eggs: Production/Stock yield.
    livestock_yield_list = []
    
    # 1. Meat: carcass weight (t/head).
    carcass = z[(z['item_raw'].isin(maps.yield_item_to_comm.keys())) &
                (z['element'].str.contains('carcass', case=False, na=False))].copy()
    if not carcass.empty:
        carcass['commodity'] = carcass['item_raw'].map(maps.yield_item_to_comm).fillna(carcass['item_raw'])
        carcass['value'] = carcass.apply(lambda r: _convert_carcass_unit(float(r['value']), r['unit']), axis=1)
        carcass_grouped = _group(carcass, 'yield_t_per_head')
        livestock_yield_list.append(carcass_grouped)
    
    # 2. Milk/egg yields matched using Item_Yield_Map and Item_Yield_Element.
    # Find livestock Items with defined Yield_Element.
    livestock_yield_items = {k: v for k, v in maps.yield_item_to_comm.items() 
                            if k in maps.yield_element_by_item}
    if livestock_yield_items:
        livestock_yields = z[z['item_raw'].isin(livestock_yield_items.keys())].copy()
        livestock_yields['target_elem'] = livestock_yields['item_raw'].map(maps.yield_element_by_item)
        livestock_yields['element_norm'] = livestock_yields['element'].astype(str).str.strip().str.lower()
        livestock_yields['target_norm'] = livestock_yields['target_elem'].fillna('').astype(str).str.strip().str.lower()
        
        # Keep matching Elements, excluding carcass weights already processed above.
        mask_match = (livestock_yields['target_elem'].notna() & 
                     (livestock_yields['element_norm'] == livestock_yields['target_norm']) &
                     ~livestock_yields['element'].str.contains('carcass', case=False, na=False))
        livestock_yields = livestock_yields[mask_match]
        
        if not livestock_yields.empty:
            livestock_yields['commodity'] = livestock_yields['item_raw'].map(maps.yield_item_to_comm).fillna(livestock_yields['item_raw'])
            # Convert units by Item type.
            livestock_yields['value'] = livestock_yields.apply(
                lambda r: _convert_livestock_yield_unit(float(r['value']), r['unit'], r['item_raw']), 
                axis=1
            )
            livestock_yield_grouped = _group(livestock_yields, 'yield_t_per_head')
            livestock_yield_list.append(livestock_yield_grouped)
    
    # Combine all livestock yield data.
    if livestock_yield_list:
        livestock_yield_df = pd.concat(livestock_yield_list, ignore_index=True)
        # Average multiple sources for the same commodity.
        # Retain M49_Country_Code.
        group_cols = ['country','iso3','year','commodity']
        if 'M49_Country_Code' in livestock_yield_df.columns:
            group_cols = ['M49_Country_Code'] + group_cols
        livestock_yield_df = livestock_yield_df.groupby(group_cols, as_index=False)['yield_t_per_head'].mean()
    else:
        livestock_yield_df = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','yield_t_per_head'])

    # Slaughter
    sl_items = maps.slaughter_item_to_comm
    slaughter = z[z['item_raw'].isin(sl_items.keys())].copy()
    slaughter['target_elem'] = slaughter['item_raw'].map(maps.slaughter_element_by_item)
    slaughter['target_norm'] = slaughter['target_elem'].fillna('').astype(str).str.strip().str.lower()
    slaughter['element_norm'] = slaughter['element'].astype(str).str.strip().str.lower()
    mask_slaughter = slaughter['target_norm'].str.contains('slaughter', na=False)
    slaughter = slaughter[mask_slaughter & (slaughter['element_norm'] == slaughter['target_norm'])]
    slaughter['commodity'] = slaughter['item_raw'].map(sl_items).fillna(slaughter['item_raw'])
    slaughter_df = _group(slaughter, 'slaughter_head')

    # Stock
    stock_items = maps.stock_item_to_comm
    stock = z[z['item_raw'].isin(stock_items.keys())].copy()
    stock['target_elem'] = stock['item_raw'].map(maps.stock_element_by_item)
    stock['target_norm'] = stock['target_elem'].fillna('').astype(str).str.strip().str.lower()
    stock['element_norm'] = stock['element'].astype(str).str.strip().str.lower()
    stock = stock[stock['target_elem'].notna() & (stock['element_norm'] == stock['target_norm'])]
    stock['commodity'] = stock['item_raw'].map(stock_items).fillna(stock['item_raw'])
    stock_df = _group(stock, 'stock_head')

    return {
        'production': production_df,
        'yield': yield_df,
        'area': area_df,
        'slaughter': slaughter_df,
        'livestock_yield': livestock_yield_df,  # Rename carcass_yield to livestock_yield.
        'stock': stock_df,
    }


def _extend_future_years(df: pd.DataFrame,
                         value_col: str,
                         universe: Universe,
                         commodity_required: bool = True) -> pd.DataFrame:
    """Extend historical data to future years, retaining M49_Country_Code."""
    if df is None or df.empty:
        return df
    required_cols = {'country', 'iso3', 'year'}
    if commodity_required:
        required_cols.add('commodity')
    if not required_cols.issubset(df.columns):
        return df
    df = df.copy()
    
    # Check for an M49 column.
    has_m49 = 'M49_Country_Code' in df.columns
    
    hist_mask = df['year'] <= 2020
    if not hist_mask.any():
        return df
    base_year = 2020 if (df['year'] == 2020).any() else int(df['year'].max())
    future_years = [y for y in universe.years if y > base_year]
    if not future_years:
        return df
    
    # Include M49 when present.
    key_cols = ['country', 'iso3'] + (['commodity'] if commodity_required else [])
    if has_m49:
        key_cols = ['M49_Country_Code'] + key_cols
    
    base = df[df['year'] == base_year][key_cols + [value_col]].copy()
    frames = [df]
    for y in future_years:
        missing_keys = df[df['year'] == y][key_cols]
        if not missing_keys.empty:
            continue
        tmp = base.copy()
        tmp['year'] = y
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(key_cols + ['year'])
    out = out[out['year'].isin(universe.years)]
    out = out.sort_values(key_cols + ['year']).reset_index(drop=True)
    return out


def load_fertilizer_statistics(fert_xlsx: str,
                               universe: Universe,
                               maps: EmisItemMappings) -> Dict[str, pd.DataFrame]:
    eff_cols_name = 'fertilizer_efficiency_kgN_per_ha'
    amt_cols_name = 'fertilizer_n_input_t'
    # Add M49_Country_Code.
    empty = {
        'efficiency': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity', eff_cols_name]),
        'amount': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity', amt_cols_name]),
    }
    if not os.path.exists(fert_xlsx):
        return empty
    df = _lc(_read_excel(fert_xlsx, sheet_name="data"))
    c_m49 = _maybe_find_col(df, ['M49 Code', 'M49'])
    c_item = _maybe_find_col(df, ['Item'])
    c_prod_item = _maybe_find_col(df, ['Production_Item', 'Production Item'])
    eff_cols = [c for c in df.columns
                if isinstance(c, str) and c.startswith('N_FertEffi_') and str(c)[-4:].isdigit()]
    amt_cols = [c for c in df.columns
                if isinstance(c, str) and c.startswith('N_contentModi_') and str(c)[-4:].isdigit()]
    if not (c_m49 and (c_item or c_prod_item) and eff_cols):
        return empty
    idx_m49 = df.columns.get_loc(c_m49)
    idx_item = df.columns.get_loc(c_item) if c_item else None
    idx_prod_item = df.columns.get_loc(c_prod_item) if c_prod_item else None
    eff_cols_idx = [(col, df.columns.get_loc(col)) for col in eff_cols]
    amt_cols_idx = [(col, df.columns.get_loc(col)) for col in amt_cols]
    records_eff: List[Dict[str, Any]] = []
    records_amt: List[Dict[str, Any]] = []
    def _clean_str(val: Any) -> Optional[str]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        if not s or s.lower() in {'nan', 'no'}:
            return None
        return s

    for row in df.itertuples(index=False, name=None):
        code_raw = row[idx_m49]
        parsed = _parse_m49_code(code_raw)
        if not parsed:
            continue
        m49_normalized = parsed
        country = m49_normalized
        item_name = _clean_str(row[idx_item]) if idx_item is not None else None
        prod_item_name = _clean_str(row[idx_prod_item]) if idx_prod_item is not None else None
        commodity: Optional[str] = None

        def _resolve(name: Optional[str]) -> Optional[str]:
            if not name:
                return None
            cand = maps.fertilizer_by_item.get(name)
            if cand and cand in universe.commodities:
                return cand
            cand = maps.production_by_item.get(name)
            if cand and cand in universe.commodities:
                return cand
            if name in universe.commodities:
                return name
            return None

        commodity = _resolve(item_name)
        if commodity is None:
            commodity = _resolve(prod_item_name)
        if commodity is None:
            continue
        iso3 = universe.iso3_by_country.get(country)
        if not iso3:
            continue
        for col, col_idx in eff_cols_idx:
            year = int(str(col)[-4:])
            if year not in universe.years:
                continue
            val = pd.to_numeric(row[col_idx], errors='coerce')
            if pd.isna(val):
                continue
            records_eff.append({
                'M49_Country_Code': m49_normalized,  # Use standardized M49 codes.
                'country': country,
                'iso3': iso3,
                'commodity': commodity,
                'year': year,
                eff_cols_name: float(val),
            })
        for col, col_idx in amt_cols_idx:
            year = int(str(col)[-4:])
            if year not in universe.years:
                continue
            val = pd.to_numeric(row[col_idx], errors='coerce')
            if pd.isna(val):
                continue
            records_amt.append({
                'M49_Country_Code': m49_normalized,  # Use standardized M49 codes.
                'country': country,
                'iso3': iso3,
                'commodity': commodity,
                'year': year,
                amt_cols_name: float(val) / 1000.0,  # kgN -> tN
            })
    eff_df = pd.DataFrame(records_eff)
    amt_df = pd.DataFrame(records_amt)
    eff_df = _extend_future_years(eff_df, eff_cols_name, universe)
    amt_df = _extend_future_years(amt_df, amt_cols_name, universe)
    return {'efficiency': eff_df, 'amount': amt_df}


def load_feed_requirement_per_head(feed_xlsx: str,
                                   universe: Universe,
                                   maps: EmisItemMappings,
                                   feed_requirement_scheme: Optional[str] = None) -> pd.DataFrame:
    # Add M49_Country_Code.
    columns = [
        'M49_Country_Code','country','iso3','year','commodity',
        'feed_requirement_kg_per_head','feed_requirement_kg_per_head_lower','feed_requirement_kg_per_head_upper'
    ]
    if not os.path.exists(feed_xlsx):
        print(f"[ERROR] 关键文件不存在: {feed_xlsx}")
        raise FileNotFoundError(f"Feed requirement file not found: {feed_xlsx}")
    scheme_norm = str(feed_requirement_scheme or 'IPCC').strip().lower()
    if scheme_norm == 'gleam':
        try:
            df = _lc(_read_excel(feed_xlsx, sheet_name='total_kgDM_per_head_GLEAM'))
        except Exception:
            return pd.DataFrame(columns=columns)
        c_species = _maybe_find_col(df, ['Species'])
        c_area = _maybe_find_col(df, ['M49_Country_Code'])
        year_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit()]
        if not all([c_species, c_area]) or not year_cols:
            raise KeyError("total_kgDM_per_head_GLEAM 表结构异常（需 Species/M49_Country_Code/Yxxxx 列）")
        records: List[Dict[str, Any]] = []
        for row in df.itertuples(index=False):
            code_raw = getattr(row, c_area, None)
            parsed = _parse_m49_code(code_raw)
            if not parsed:
                continue
            m49_normalized = parsed
            country = m49_normalized
            species = str(getattr(row, c_species)).strip()
            commodity = maps.feed_item_to_comm.get(species)
            if not commodity:
                continue
            iso3 = universe.iso3_by_country.get(country)
            if not iso3:
                continue
            for col in year_cols:
                year = int(col[1:])
                if year not in universe.years:
                    continue
                val = pd.to_numeric(getattr(row, col), errors='coerce')
                if pd.isna(val):
                    continue
                m49_normalized = f"'{str(parsed).zfill(3)}" if parsed is not None else None
                records.append({
                    'M49_Country_Code': m49_normalized,
                    'country': country,
                    'iso3': iso3,
                    'commodity': commodity,
                    'year': year,
                    'feed_requirement_kg_per_head': float(val),
                    'feed_requirement_kg_per_head_lower': None,
                    'feed_requirement_kg_per_head_upper': None,
                })
    else:
        try:
            df = _lc(_read_excel(feed_xlsx, sheet_name='total_kgDM_per_head_IPCC'))
        except Exception:
            return pd.DataFrame(columns=columns)
        c_species = _maybe_find_col(df, ['Species'])
        c_area = _maybe_find_col(df, ['M49_Country_Code'])
        c_year_col = _maybe_find_col(df, ['year'])
        c_total_col = _maybe_find_col(df, ['total_kgDM_per_head'])
        c_lower_col = _maybe_find_col(df, ['lower_kgDM_per_head'])
        c_upper_col = _maybe_find_col(df, ['upper_kgDM_per_head'])
        if not all([c_species, c_area, c_year_col, c_total_col]):
            raise KeyError("total_kgDM_per_head_IPCC 表结构已更新（需 year/total/lower/upper 列），请检查输入文件")
        records = []
        for row in df.itertuples(index=False):
            code_raw = getattr(row, c_area, None)
            parsed = _parse_m49_code(code_raw)
            if not parsed:
                continue
            m49_normalized = parsed
            country = m49_normalized
            species = str(getattr(row, c_species)).strip()
            commodity = maps.feed_item_to_comm.get(species)
            if not commodity:
                continue
            iso3 = universe.iso3_by_country.get(country)
            if not iso3:
                continue
            year_val = pd.to_numeric(getattr(row, c_year_col), errors='coerce')
            if pd.isna(year_val):
                continue
            year = int(year_val)
            if year not in universe.years:
                continue
            val = pd.to_numeric(getattr(row, c_total_col), errors='coerce')
            if pd.isna(val):
                continue
            lower_val = pd.to_numeric(getattr(row, c_lower_col), errors='coerce') if c_lower_col else np.nan
            upper_val = pd.to_numeric(getattr(row, c_upper_col), errors='coerce') if c_upper_col else np.nan
            m49_normalized = f"'{str(parsed).zfill(3)}" if parsed is not None else None
            records.append({
                'M49_Country_Code': m49_normalized,
                'country': country,
                'iso3': iso3,
                'commodity': commodity,
                'year': year,
                'feed_requirement_kg_per_head': float(val),
                'feed_requirement_kg_per_head_lower': None if pd.isna(lower_val) else float(lower_val),
                'feed_requirement_kg_per_head_upper': None if pd.isna(upper_val) else float(upper_val),
            })
    df_out = pd.DataFrame(records, columns=columns)
    if df_out.empty:
        return df_out
    df_out = _extend_future_years(df_out, 'feed_requirement_kg_per_head', universe)
    return df_out


def load_manure_management_ratio(manure_csv: str,
                                 universe: Universe,
                                 maps: EmisItemMappings) -> pd.DataFrame:
    """
    Load livestock manure management ratios.
    WARNING: keep only commodities defined in dict_v3 (drop 'All Animals' and similar)
    """
    # Add M49_Country_Code.
    columns = ['M49_Country_Code','country','iso3','year','commodity','manure_management_ratio']
    if not os.path.exists(manure_csv):
        print(f"[ERROR] 关键文件不存在: {manure_csv}")
        raise FileNotFoundError(f"Manure CSV file not found: {manure_csv}")
    df_raw = _read_csv(manure_csv)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val = _find_col(df, ['Value'])
    if not all([c_area, c_year, c_item, c_elem, c_val]):
        return pd.DataFrame(columns=columns)
    keep_cols = [c_area, c_year, c_item, c_elem, c_val]
    if 'M49_Country_Code' in df.columns:
        keep_cols.append('M49_Country_Code')
    z = df[keep_cols].copy()
    z = z.rename(columns={c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'})
    z['value'] = pd.to_numeric(z['value'], errors='coerce')
    z = z.dropna(subset=['value'])
    z = _attach_country_from_m49(df, z, universe, context=f"Manure management ratio ({manure_csv})")
    # Preserve M49_Country_Code and ensure index alignment.
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df.loc[z.index, 'M49_Country_Code'].values
    
    # Normalize M49 to an apostrophe plus three digits.
    if 'M49_Country_Code' in z.columns:
        def _format_m49_quote(val):
            if pd.isna(val):
                return val
            return _norm_m49(val)
        z['M49_Country_Code'] = z['M49_Country_Code'].apply(_format_m49_quote)
    
    z['country'] = z['country'].astype(str).str.strip()
    
    # WARNING: keep only the 198 valid countries
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    z = z.dropna(subset=['iso3'])
    z['iso3'] = z['iso3'].astype(str)
    z['year'] = pd.to_numeric(z['year'], errors='coerce')
    z = z.dropna(subset=['year'])
    z['year'] = z['year'].astype(int)
    z['element_norm'] = z['element'].astype(str).str.strip().str.lower()
    treated_label = 'manure management (manure treated, n content)'
    excreted_label = 'amount excreted in manure (n content)'
    z = z[z['element_norm'].isin([treated_label, excreted_label])]
    if z.empty:
        return pd.DataFrame(columns=columns)
    z['item_clean'] = z['item_raw'].astype(str).str.strip()
    
    # WARNING: use dict_v3 mapping and keep only items that map
    before_map = len(z)
    z['commodity'] = z['item_clean'].map(maps.stock_item_to_comm)
    if z['commodity'].isna().any():
        z.loc[z['commodity'].isna(), 'commodity'] = z.loc[z['commodity'].isna(), 'item_clean'].map(maps.slaughter_item_to_comm)
    z = z.dropna(subset=['commodity'])
    after_map = len(z)
    if before_map > after_map:
        print(f"[INFO] 过滤掉 {before_map - after_map} 行非dict_v3定义的manure Item")
    
    # WARNING: do not filter using universe.commodities!
    # Manure uses Item_Stock_Map names; universe.commodities uses Item_Production_Map names.
    # Commodity was already validated by maps.stock_item_to_comm/slaughter_item_to_comm.
    
    # Preserve M49_Country_Code in the pivot.
    index_cols = ['country','iso3','year','commodity']
    if 'M49_Country_Code' in z.columns:
        index_cols = ['M49_Country_Code'] + index_cols
    
    pivot = z.pivot_table(index=index_cols,
                          columns='element_norm',
                          values='value',
                          aggfunc='sum',
                          fill_value=np.nan)
    pivot = pivot.reset_index()
    if treated_label not in pivot.columns or excreted_label not in pivot.columns:
        return pd.DataFrame(columns=columns)
    pivot['manure_management_ratio'] = pivot[treated_label] / pivot[excreted_label].replace(0, np.nan)
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna(subset=['manure_management_ratio'])
    
    # Include M49_Country_Code when selecting columns.
    out_cols = ['country','iso3','year','commodity','manure_management_ratio']
    if 'M49_Country_Code' in pivot.columns:
        out_cols = ['M49_Country_Code'] + out_cols
    out = pivot[out_cols].copy()
    out = _extend_future_years(out, 'manure_management_ratio', universe)
    return out
