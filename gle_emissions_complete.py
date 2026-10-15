# -*- coding: utf-8 -*-
"""
gle_emissions_complete.py
==========================
Complete livestock emissions calculation module.
Implement the full chain from supply-demand model production to emissions.
Includes four emissions processes:
1. Enteric fermentation (CH4)
2. Manure management (CH4, N2O)
3. Manure applied to soils (N2O)
4. Manure left on pasture (N2O)
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from config_paths import get_results_base
from runtime_data_cache import read_excel_cached, read_csv_cached

# Configure logging.
logger = logging.getLogger(__name__)
_CALCULATOR_CACHE: Dict[Tuple[str, str, str, str, str, int], "LivestockEmissionsCalculator"] = {}


def _get_debug_level() -> int:
    raw = os.environ.get("NZF_DEBUG_LEVEL", "0").strip()
    try:
        return max(0, int(float(raw)))
    except Exception:
        return 0


class _DebugMessageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _get_debug_level() >= 2:
            return True
        msg = record.getMessage()
        if "[DEBUG" in msg or "【DEBUG】" in msg:
            return False
        return True


if not any(isinstance(f, _DebugMessageFilter) for f in logger.filters):
    logger.addFilter(_DebugMessageFilter())


def _normalize_m49_for_comparison(m49_val) -> str:
    """
    Normalize M49 codes to a common format for comparison.
    Convert any representation to 'xxx: a leading apostrophe followed by three digits.
    """
    if m49_val is None or pd.isna(m49_val):
        return ''
    s = str(m49_val).strip()
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


class LivestockEmissionsCalculator:
    """Livestock emissions calculator."""
    
    def __init__(self, 
                 gle_params_path: str,
                 hist_production_path: str,
                 hist_emissions_path: str,
                 hist_manure_stock_path: str,
                 dict_v3_path: str,
                 scenario_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the calculator.
        
        Args:
            gle_params_path: Path to GLE_parameters.xlsx.
            hist_production_path: Path to Production_Crops_Livestock_E_All_Data_NOFLAG.csv.
            hist_emissions_path: Path to Emissions_livestock_dairy_split.csv, already split into dairy/non-dairy.
            hist_manure_stock_path: Path to Environment_LivestockManure_with_ratio.csv.
            dict_v3_path: Path to dict_v3.xlsx.
        """
        self.gle_params_path = gle_params_path
        self.hist_production_path = hist_production_path
        self.hist_emissions_path = hist_emissions_path
        self.hist_manure_stock_path = hist_manure_stock_path
        self.dict_v3_path = dict_v3_path
        self.scenario_params = scenario_params if isinstance(scenario_params, dict) else {}
        self._yield_base_year = 2020
        self._yield_multiplier_map = self._build_yield_multiplier_map(self.scenario_params)
        
        # Store calculated results for export to production_summary.
        self._computed_slaughter = []  # List[pd.DataFrame] storing calculated slaughter data
        self._computed_stock = []      # List[pd.DataFrame] storing calculated stock data
        self._computed_carcass_yield = []  # List[pd.DataFrame] storing carcass_yield data for meat
        self._computed_dairy_yield = []   # List[pd.DataFrame] storing dairy_yield data for milk/eggs
        self._manure_ratio_fallback_counts = {'region': 0, 'global': 0, 'default': 0}
        self._hist_prod_row_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._hist_prod_region_mean: Dict[Tuple[str, str, str, str], float] = {}
        self._hist_prod_global_mean: Dict[Tuple[str, str, str], float] = {}
        self._hist_unit_by_item_element: Dict[Tuple[str, str], str] = {}
        self._hist_year_cols: List[str] = []
        
        # Load data.
        self._load_mappings()
        self._load_parameters()
        self._load_historical_data()
        self._build_hist_production_lookup()

    def reset_runtime_outputs(self) -> None:
        self._computed_slaughter = []
        self._computed_stock = []
        self._computed_carcass_yield = []
        self._computed_dairy_yield = []
        self._manure_ratio_fallback_counts = {'region': 0, 'global': 0, 'default': 0}

    @staticmethod
    def _get_base_item_name(item_emis: str) -> Optional[str]:
        base_item_map = {
            'Cattle, dairy': 'Cattle',
            'Cattle, non-dairy': 'Cattle',
            'Buffaloes, dairy': 'Buffaloes',
            'Buffaloes, non-dairy': 'Buffaloes',
            'Goats, dairy': 'Goats',
            'Goats, non-dairy': 'Goats',
            'Sheep, dairy': 'Sheep',
            'Sheep, non-dairy': 'Sheep',
            'Camels, dairy': 'Camels',
            'Camels, non-dairy': 'Camels',
        }
        return base_item_map.get(str(item_emis).strip())

    def _build_yield_multiplier_map(self, scenario_params: Dict[str, Any]) -> Dict[Tuple[str, str, int], float]:
        """Normalize yield_multiplier keys to (m49, Item_Emis, year)."""
        out: Dict[Tuple[str, str, int], float] = {}
        raw = scenario_params.get('yield_multiplier', {}) if isinstance(scenario_params, dict) else {}
        for key, mult in raw.items():
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            country, commodity, year = key
            m49 = _normalize_m49_for_comparison(country)
            if not m49 or commodity is None:
                continue
            try:
                year_int = int(year)
            except Exception:
                continue
            try:
                mult_val = float(mult)
            except Exception:
                continue
            out[(m49, str(commodity).strip(), year_int)] = mult_val
        return out

    def _get_yield_multiplier(self, m49_code: Any, commodity: str, year: int) -> float:
        if not self._yield_multiplier_map:
            return 1.0
        m49 = _normalize_m49_for_comparison(m49_code)
        if not m49 or commodity is None:
            return 1.0
        try:
            year_int = int(year)
        except Exception:
            year_int = self._yield_base_year
        comm = str(commodity).strip()
        val = self._yield_multiplier_map.get((m49, comm, year_int))
        if val is None and year_int != self._yield_base_year:
            val = self._yield_multiplier_map.get((m49, comm, self._yield_base_year))
        return float(val) if val is not None else 1.0

    def _get_emission_factor_multiplier(self,
                                        scenario_params: Optional[Dict[str, Any]],
                                        m49_code: Any,
                                        item_emis: str,
                                        process: str,
                                        year: int,
                                        ghg: Optional[str] = None) -> float:
        if not scenario_params:
            return 1.0
        m49 = _normalize_m49_for_comparison(m49_code)
        if not m49:
            return 1.0
        item_key = str(item_emis).strip()
        proc_key = str(process).strip()
        ghg_key = str(ghg).strip() if ghg else None
        try:
            year_key = int(year)
        except Exception:
            year_key = year
        keys = []
        if ghg_key:
            keys.extend([
                (m49, item_key, proc_key, ghg_key, year_key),
                (m49, item_key, 'All', ghg_key, year_key),
                (m49, 'All', proc_key, ghg_key, year_key),
                (m49, 'All', 'All', ghg_key, year_key),
                (m49, item_key, proc_key, 'All', year_key),
                (m49, item_key, 'All', 'All', year_key),
                (m49, 'All', proc_key, 'All', year_key),
                (m49, 'All', 'All', 'All', year_key),
            ])
        keys.extend([
            (m49, item_key, proc_key, year_key),
            (m49, item_key, 'All', year_key),
            (m49, 'All', proc_key, year_key),
            (m49, 'All', 'All', year_key),
        ])
        ef_mult_dict = scenario_params.get('emission_factor_multiplier', {})
        for key in keys:
            if key in ef_mult_dict:
                try:
                    return float(ef_mult_dict.get(key, 1.0))
                except Exception:
                    return 1.0
        legacy_dict = scenario_params.get('ef_multiplier_by', {})
        for key in keys:
            if key in legacy_dict:
                try:
                    return float(legacy_dict.get(key, 1.0))
                except Exception:
                    return 1.0
        return 1.0

    def _get_emission_factor_absolute(self,
                                      scenario_params: Optional[Dict[str, Any]],
                                      m49_code: Any,
                                      item_emis: str,
                                      process: str,
                                      ghg: str,
                                      year: int) -> Optional[float]:
        if not scenario_params:
            return None
        ef_abs_dict = scenario_params.get('emission_factor_absolute_by', {})
        if not ef_abs_dict:
            return None
        m49 = _normalize_m49_for_comparison(m49_code)
        if not m49:
            return None
        item_key = str(item_emis).strip()
        proc_key = str(process).strip()
        ghg_key = str(ghg).strip() if ghg else 'All'
        try:
            year_key = int(year)
        except Exception:
            year_key = year
        keys = [
            (m49, item_key, proc_key, ghg_key, year_key),
            (m49, item_key, 'All', ghg_key, year_key),
            (m49, 'All', proc_key, ghg_key, year_key),
            (m49, 'All', 'All', ghg_key, year_key),
            (m49, item_key, proc_key, 'All', year_key),
            (m49, item_key, 'All', 'All', year_key),
            (m49, 'All', proc_key, 'All', year_key),
            (m49, 'All', 'All', 'All', year_key),
            (m49, item_key, proc_key, year_key),
            (m49, item_key, 'All', year_key),
            (m49, 'All', proc_key, year_key),
            (m49, 'All', 'All', year_key),
        ]
        for key in keys:
            if key in ef_abs_dict:
                try:
                    return float(ef_abs_dict.get(key))
                except Exception:
                    return None
        return None

    def _get_emission_factor_bound(self,
                                   scenario_params: Optional[Dict[str, Any]],
                                   m49_code: Any,
                                   item_emis: str,
                                   process: str,
                                   ghg: str,
                                   year: int) -> Optional[Tuple[float, float, bool, bool, float]]:
        if not scenario_params:
            return None
        ef_bound_dict = scenario_params.get('emission_factor_bound_by', {})
        if not ef_bound_dict:
            return None
        m49 = _normalize_m49_for_comparison(m49_code)
        if not m49:
            return None
        item_key = str(item_emis).strip()
        proc_key = str(process).strip()
        ghg_key = str(ghg).strip() if ghg else 'All'
        try:
            year_key = int(year)
        except Exception:
            year_key = year
        keys = [
            (m49, item_key, proc_key, ghg_key, year_key),
            (m49, item_key, 'All', ghg_key, year_key),
            (m49, 'All', proc_key, ghg_key, year_key),
            (m49, 'All', 'All', ghg_key, year_key),
            (m49, item_key, proc_key, 'All', year_key),
            (m49, item_key, 'All', 'All', year_key),
            (m49, 'All', proc_key, 'All', year_key),
            (m49, 'All', 'All', 'All', year_key),
            (m49, item_key, proc_key, year_key),
            (m49, item_key, 'All', year_key),
            (m49, 'All', proc_key, year_key),
            (m49, 'All', 'All', year_key),
        ]
        for key in keys:
            if key in ef_bound_dict:
                try:
                    return ef_bound_dict.get(key)
                except Exception:
                    return None
        return None

    @staticmethod
    def _apply_ef_bound(base_ef: float, bound: Tuple[float, float, bool, bool, float]) -> float:
        lo, hi, lo_is_y2020, hi_is_y2020, u = bound
        try:
            lo = float(lo)
            hi = float(hi)
            u = float(u)
        except Exception:
            return base_ef
        lo_val = base_ef * lo if lo_is_y2020 else lo
        hi_val = base_ef * hi if hi_is_y2020 else hi
        if hi_val < lo_val:
            lo_val, hi_val = hi_val, lo_val
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        return lo_val + (hi_val - lo_val) * u
    def _load_mappings(self):
        """Load mappings from dict_v3."""
        emis_item = read_excel_cached(self.dict_v3_path, sheet_name='Emis_item')
        
        # Retain dict_v3 for subsequent use, particularly feed_requirement creation.
        self.dict_v3 = {'Emis_item': emis_item}
        
        # Load regional mappings.
        try:
            region_df = read_excel_cached(self.dict_v3_path, sheet_name='region')
            self.m49_to_region = {}
            for _, row in region_df.iterrows():
                m49_code = row.get('M49_Country_Code')
                region = row.get('Region_market_full')
                if pd.isna(region):
                    region = row.get('Region_market_agg')
                if pd.isna(region):
                    region = row.get('Region_agg2')
                if pd.notna(m49_code) and pd.notna(region):
                    # Standardize M49 to 'xxx: a leading apostrophe and three digits.
                    m49_str = _normalize_m49_for_comparison(m49_code)
                    if m49_str:
                        self.m49_to_region[m49_str] = str(region).strip()
            logger.info(f" 加载了 {len(self.m49_to_region)} 个国家的区域映射(Region_market_full)")
            self.region_to_m49_codes = {}
            for m49_code, region_name in self.m49_to_region.items():
                self.region_to_m49_codes.setdefault(str(region_name), []).append(str(m49_code))
            
            # Select valid countries where Region_label_new != 'no'.
            try:
                region_df_full = read_excel_cached(self.dict_v3_path, sheet_name='region')
                valid_countries = region_df_full[
                    region_df_full['Region_label_new'].astype(str).str.lower() != 'no'
                ]['M49_Country_Code'].dropna().unique()
                # Standardize M49 to 'xxx: a leading apostrophe and three digits.
                self.valid_m49_codes = set()
                for m49 in valid_countries:
                    try:
                        m49_str = str(m49).strip().lstrip("'\"")
                        if m49_str.isdigit():
                            m49_str = f"'{m49_str.zfill(3)}"
                        self.valid_m49_codes.add(m49_str)
                    except (ValueError, TypeError):
                        continue
                logger.info(f" 筛选出 {len(self.valid_m49_codes)} 个有效国家（Region_label_new != 'no'）")
            except Exception as e2:
                logger.warning(f"WARNING: failed to filter valid years: {e2}")
                self.valid_m49_codes = set(self.m49_to_region.keys())
        except Exception as e:
            logger.warning(f"WARNING: failed to load process mapping: {e}")
            self.m49_to_region = {}
            self.region_to_m49_codes = {}
            self.valid_m49_codes = set()
        
        # Select livestock emissions processes.
        livestock_processes = [
            'Enteric fermentation',
            'Manure management', 
            'Manure applied to soils',
            'Manure left on pasture'
        ]
        
        self.emis_mappings = emis_item[emis_item['Process'].isin(livestock_processes)].copy()
        
        # Build mapping dictionaries.
        self.item_production_map = {}
        self.item_yield_map = {}
        self.item_slaughtered_map = {}
        self.item_slaughtered_ratio_map = {}
        self.item_stock_map = {}
        
        # Reverse mapping: Item_Production_Map -> Item_Emis.
        self.production_to_emis = {}  # Look up Item_Emis from production-data names.
        
        for _, row in self.emis_mappings.iterrows():
            # Item_Emis in dict_v3 contains the commodity names used by the model.
            item_emis = row['Item_Emis']
            
            # Build forward and reverse production mappings.
            if pd.notna(row.get('Item_Production_Map')):
                prod_item = row['Item_Production_Map']
                self.item_production_map[item_emis] = {
                    'Item': prod_item,
                    'Element': row.get('Item_Production_Element', 'Production')
                }
                # Reverse mapping: FAOSTAT name -> model name.
                self.production_to_emis[prod_item] = item_emis
            
            # Yield mapping
            if pd.notna(row.get('Item_Yield_Map')):
                self.item_yield_map[item_emis] = {
                    'Item': row['Item_Yield_Map'],
                    'Element': row.get('Item_Yield_Element', 'Yield')
                }
            
            # Slaughtered mapping
            if pd.notna(row.get('Item_Slaughtered_Map')):
                self.item_slaughtered_map[item_emis] = {
                    'Item': row['Item_Slaughtered_Map'],
                    'Element': row.get('Item_Slaughtered_Element', 'Producing Animals/Slaughtered')
                }
            
            # Slaughtered ratio mapping
            if pd.notna(row.get('Item_SlaughteredRatio_Map')):
                self.item_slaughtered_ratio_map[item_emis] = {
                    'Item': row['Item_SlaughteredRatio_Map'],
                    'Element': row.get('Item_SlaughteredRatio_Element', 'Producing/Slaughtered ratio')
                }
            
            # Stock mapping
            if pd.notna(row.get('Item_Stock_Map')):
                self.item_stock_map[item_emis] = {
                    'Item': row['Item_Stock_Map'],
                    'Element': row.get('Item_Stock_Element', 'Stocks')
                }
    
    def _load_parameters(self):
        """Load the GLE parameter table."""
        self.gle_params = read_excel_cached(self.gle_params_path)
        
        # Check parameter column names, especially year columns.
        year_cols = [col for col in self.gle_params.columns if str(col).startswith('Y') or str(col).startswith('2')]
        logger.info(f"[INFO] GLE_parameters年份列: {year_cols[:5]}...{year_cols[-3:] if len(year_cols) > 5 else year_cols}")
        
        # Select rows with Select=1.
        self.gle_params = self.gle_params[self.gle_params['Select'] == 1].copy()
        # Standardize M49 to 'xxx: a leading apostrophe and three digits.
        if 'M49_Country_Code' in self.gle_params.columns:
            self.gle_params['M49_Country_Code'] = self.gle_params['M49_Country_Code'].apply(
                _normalize_m49_for_comparison
            )
        
    def _load_historical_data(self):
        """Load historical data."""
        # Production data
        self.hist_production = read_csv_cached(self.hist_production_path, encoding='latin1')
        
        # Emissions data
        self.hist_emissions = read_csv_cached(self.hist_emissions_path, encoding='latin1')
        
        # Manure and stock data
        self.hist_manure_stock = read_csv_cached(self.hist_manure_stock_path)
        
        # Standardize M49 to 'xxx: a leading apostrophe and three digits.
        # Match the format passed by S4_0_main.py.
        def _normalize_m49(s):
            """Normalize M49 to 'xxx format (4??'004', 76.0??'076')"""
            def normalize_single(val):
                if pd.isna(val):
                    return None
                return _normalize_m49_for_comparison(val)
            return s.apply(normalize_single)

        # hist_production uses 'Area Code (M49)' or 'M49_Country_Code'
        if 'Area Code (M49)' in self.hist_production.columns:
            self.hist_production['Area Code (M49)'] = _normalize_m49(self.hist_production['Area Code (M49)'])
        elif 'M49_Country_Code' in self.hist_production.columns:
            self.hist_production['Area Code (M49)'] = _normalize_m49(self.hist_production['M49_Country_Code'])

        if 'M49_Country_Code' in self.hist_emissions.columns:
            self.hist_emissions['M49_Country_Code'] = _normalize_m49(self.hist_emissions['M49_Country_Code'])

        if 'M49_Country_Code' in self.hist_manure_stock.columns:
            self.hist_manure_stock['M49_Country_Code'] = _normalize_m49(self.hist_manure_stock['M49_Country_Code'])

    def _build_hist_livestock_split_ratio_lookup(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Build stock-based dairy/non-dairy split ratios for merged historical livestock items."""
        cache = getattr(self, '_hist_livestock_split_ratio_lookup', None)
        if cache is not None:
            return cache

        split_targets = {
            'Buffalo': ('Buffalo, dairy', 'Buffalo, non-dairy'),
            'Camels': ('Camel, dairy', 'Camel, non-dairy'),
            'Goats': ('Goats, dairy', 'Goats, non-dairy'),
            'Sheep': ('Sheep, dairy', 'Sheep, non-dairy'),
        }
        required_cols = {'M49_Country_Code', 'Item', 'Element'}
        if self.hist_manure_stock.empty or not required_cols.issubset(self.hist_manure_stock.columns):
            self._hist_livestock_split_ratio_lookup = {}
            return self._hist_livestock_split_ratio_lookup

        stock_df = self.hist_manure_stock[
            self.hist_manure_stock['Element'].astype(str).eq('Stocks')
        ].copy()
        stock_year_cols = [
            col for col in stock_df.columns
            if isinstance(col, str) and col.startswith('Y') and col[1:].isdigit()
        ]
        if stock_df.empty or not stock_year_cols:
            self._hist_livestock_split_ratio_lookup = {}
            return self._hist_livestock_split_ratio_lookup

        ratio_lookup: Dict[str, Dict[str, pd.DataFrame]] = {}
        for base_item, (dairy_item, non_dairy_item) in split_targets.items():
            dairy_df = (
                stock_df[stock_df['Item'].astype(str).eq(dairy_item)]
                .set_index('M49_Country_Code')[stock_year_cols]
                .apply(pd.to_numeric, errors='coerce')
            )
            non_dairy_df = (
                stock_df[stock_df['Item'].astype(str).eq(non_dairy_item)]
                .set_index('M49_Country_Code')[stock_year_cols]
                .apply(pd.to_numeric, errors='coerce')
            )

            dairy_df, non_dairy_df = dairy_df.align(non_dairy_df, fill_value=0.0)
            total_df = dairy_df + non_dairy_df

            dairy_ratio = dairy_df.div(total_df).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            non_dairy_ratio = non_dairy_df.div(total_df).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            no_ratio_mask = total_df.isna() | total_df.eq(0)
            dairy_ratio = dairy_ratio.mask(no_ratio_mask, 0.0)
            non_dairy_ratio = non_dairy_ratio.mask(no_ratio_mask, 1.0)

            ratio_lookup[base_item] = {
                dairy_item: dairy_ratio,
                non_dairy_item: non_dairy_ratio,
            }

        self._hist_livestock_split_ratio_lookup = ratio_lookup
        return ratio_lookup

    @staticmethod
    def _resolve_hist_split_ratio_col(ratio_cols: List[str], target_year_col: str) -> Optional[str]:
        if target_year_col in ratio_cols:
            return target_year_col
        if not (isinstance(target_year_col, str) and target_year_col.startswith('Y') and target_year_col[1:].isdigit()):
            return None

        available_years = sorted(
            int(col[1:]) for col in ratio_cols
            if isinstance(col, str) and col.startswith('Y') and col[1:].isdigit()
        )
        if not available_years:
            return None

        target_year = int(target_year_col[1:])
        if target_year < available_years[0]:
            return f"Y{available_years[0]}"
        if target_year > available_years[-1]:
            return f"Y{available_years[-1]}"
        return None

    def _split_historical_livestock_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split merged Buffalo/Camels/Goats/Sheep rows before dict_v3 Item filtering."""
        if df.empty or 'Item' not in df.columns or 'M49_Country_Code' not in df.columns:
            return df

        item_aliases = {
            'Buffalo': 'Buffalo',
            'Buffaloes': 'Buffalo',
            'Camels': 'Camels',
            'Goats': 'Goats',
            'Sheep': 'Sheep',
        }
        split_targets = {
            'Buffalo': ('Buffalo, dairy', 'Buffalo, non-dairy'),
            'Camels': ('Camel, dairy', 'Camel, non-dairy'),
            'Goats': ('Goats, dairy', 'Goats, non-dairy'),
            'Sheep': ('Sheep, dairy', 'Sheep, non-dairy'),
        }

        item_series = df['Item'].astype(str).str.strip()
        rows_to_split = item_series.isin(item_aliases.keys())
        if not rows_to_split.any():
            return df

        ratio_lookup = self._build_hist_livestock_split_ratio_lookup()
        year_cols = [
            col for col in df.columns
            if isinstance(col, str) and col.startswith('Y') and col[1:].isdigit()
        ]
        if not year_cols:
            return df

        split_source = df[rows_to_split].copy()
        keep_df = df[~rows_to_split].copy()
        split_rows = []

        for _, row in split_source.iterrows():
            base_item = item_aliases.get(str(row['Item']).strip())
            if not base_item:
                continue
            dairy_item, non_dairy_item = split_targets[base_item]
            ratio_frames = ratio_lookup.get(base_item, {})
            dairy_ratio_df = ratio_frames.get(dairy_item)
            non_dairy_ratio_df = ratio_frames.get(non_dairy_item)
            country_code = _normalize_m49_for_comparison(row['M49_Country_Code'])

            if dairy_ratio_df is not None and country_code in dairy_ratio_df.index:
                dairy_ratio_values = dairy_ratio_df.loc[country_code]
                non_dairy_ratio_values = non_dairy_ratio_df.loc[country_code]
            else:
                dairy_ratio_values = pd.Series(0.0, index=year_cols, dtype=float)
                non_dairy_ratio_values = pd.Series(1.0, index=year_cols, dtype=float)

            dairy_row = row.to_dict()
            dairy_row['Item'] = dairy_item
            non_dairy_row = row.to_dict()
            non_dairy_row['Item'] = non_dairy_item

            for year_col in year_cols:
                raw_value = pd.to_numeric(row.get(year_col), errors='coerce')
                if pd.isna(raw_value):
                    dairy_row[year_col] = np.nan
                    non_dairy_row[year_col] = np.nan
                    continue

                ratio_col = self._resolve_hist_split_ratio_col(list(dairy_ratio_values.index), year_col)
                dairy_ratio = pd.to_numeric(dairy_ratio_values.get(ratio_col), errors='coerce')
                non_dairy_ratio = pd.to_numeric(non_dairy_ratio_values.get(ratio_col), errors='coerce')
                dairy_ratio = 0.0 if pd.isna(dairy_ratio) else float(dairy_ratio)
                non_dairy_ratio = 1.0 if pd.isna(non_dairy_ratio) else float(non_dairy_ratio)

                dairy_row[year_col] = raw_value * dairy_ratio
                non_dairy_row[year_col] = raw_value * non_dairy_ratio

            split_rows.append(dairy_row)
            split_rows.append(non_dairy_row)

        if not split_rows:
            return keep_df
        split_df = pd.DataFrame(split_rows)
        return pd.concat([keep_df, split_df], ignore_index=True)

    def _build_hist_production_lookup(self):
        """Prebuild historical yield lookups to avoid repeated full-table filtering on hot paths."""
        self._hist_prod_row_lookup = {}
        self._hist_prod_region_mean = {}
        self._hist_prod_global_mean = {}
        self._hist_unit_by_item_element = {}
        self._hist_year_cols = []
        if self.hist_production is None or self.hist_production.empty:
            return
        if 'Area Code (M49)' not in self.hist_production.columns:
            return

        year_cols = [
            col for col in self.hist_production.columns
            if isinstance(col, str) and col.startswith('Y') and col[1:].isdigit()
        ]
        if not year_cols:
            return
        self._hist_year_cols = year_cols

        relevant_pairs = set()
        for mapping in self.item_yield_map.values():
            item = str(mapping.get('Item', '') or '').strip()
            element = str(mapping.get('Element', 'Yield') or 'Yield').strip()
            if not item:
                continue
            relevant_pairs.add((item, element))
            if element == 'Yield':
                relevant_pairs.add((item, 'Production'))
                relevant_pairs.add((item, 'Milk Animals'))
        if not relevant_pairs:
            return

        hp = self.hist_production.copy()
        hp['Item'] = hp['Item'].astype(str).str.strip()
        hp['Element'] = hp['Element'].astype(str).str.strip()
        pair_mask = [(item, element) in relevant_pairs for item, element in zip(hp['Item'], hp['Element'])]
        keep_cols = ['Area Code (M49)', 'Item', 'Element'] + year_cols
        unit_col = 'Unit' if 'Unit' in hp.columns else None
        if unit_col:
            keep_cols.append(unit_col)
        hist_subset = hp.loc[pair_mask, keep_cols].copy()
        if hist_subset.empty:
            return
        hist_subset['ratio_region'] = hist_subset['Area Code (M49)'].map(self.m49_to_region)

        if unit_col:
            unit_df = hist_subset[['Item', 'Element', unit_col]].dropna(subset=[unit_col]).copy()
            if not unit_df.empty:
                unit_df[unit_col] = unit_df[unit_col].astype(str)
                self._hist_unit_by_item_element = (
                    unit_df.groupby(['Item', 'Element'], as_index=False)[unit_col]
                    .first()
                    .set_index(['Item', 'Element'])[unit_col]
                    .to_dict()
                )

        for rec in hist_subset.to_dict('records'):
            key = (
                str(rec.get('Area Code (M49)') or '').strip(),
                str(rec.get('Item') or '').strip(),
                str(rec.get('Element') or '').strip(),
            )
            if not all(key):
                continue
            if key in self._hist_prod_row_lookup:
                continue
            row_vals = {col: rec.get(col) for col in year_cols}
            row_vals['_unit'] = str(rec.get(unit_col) or '') if unit_col else ''
            self._hist_prod_row_lookup[key] = row_vals

        melt_id_cols = ['Area Code (M49)', 'ratio_region', 'Item', 'Element']
        hist_long = hist_subset.melt(
            id_vars=melt_id_cols,
            value_vars=year_cols,
            var_name='year_col',
            value_name='value',
        )
        hist_long['value'] = pd.to_numeric(hist_long['value'], errors='coerce')
        hist_long = hist_long[np.isfinite(hist_long['value']) & (hist_long['value'] > 0)].copy()
        if hist_long.empty:
            return

        self._hist_prod_global_mean = (
            hist_long.groupby(['Item', 'Element', 'year_col'])['value']
            .mean()
            .to_dict()
        )
        hist_region = hist_long[hist_long['ratio_region'].astype(str).str.strip() != ''].copy()
        if not hist_region.empty:
            self._hist_prod_region_mean = (
                hist_region.groupby(['ratio_region', 'Item', 'Element', 'year_col'])['value']
                .mean()
                .to_dict()
            )

    def _lookup_hist_row_value(self, m49_code: Any, item: str, element: str, year_col: str) -> Tuple[Optional[float], str]:
        m49_norm = _normalize_m49_for_comparison(m49_code)
        key = (m49_norm, str(item).strip(), str(element).strip())
        row = self._hist_prod_row_lookup.get(key)
        if not row:
            return None, ''
        value = pd.to_numeric(row.get(year_col), errors='coerce')
        if pd.notna(value) and float(value) > 0:
            return float(value), str(row.get('_unit', '') or '')
        return None, str(row.get('_unit', '') or '')

    def _lookup_hist_region_mean(self, m49_code: Any, item: str, element: str, year_col: str) -> Optional[float]:
        region = self.m49_to_region.get(_normalize_m49_for_comparison(m49_code))
        if not region:
            return None
        value = self._hist_prod_region_mean.get((str(region), str(item).strip(), str(element).strip(), year_col))
        return float(value) if value is not None else None

    def _lookup_hist_global_mean(self, item: str, element: str, year_col: str) -> Optional[float]:
        value = self._hist_prod_global_mean.get((str(item).strip(), str(element).strip(), year_col))
        return float(value) if value is not None else None
        
    def _load_historical_emissions(self, years: List[int]) -> Dict[str, Optional[pd.DataFrame]]:
        """
         Read historical emissions from Emissions_livestock_dairy_split.csv.
        The source file already separates dairy/non-dairy buffalo, camels, sheep, and goats.
        
        Args:
            years: Years to read.
            
        Returns:
            Historical emissions grouped by process.
        """
        logger.info(f"\n[DEBUG] Loading historical emissions for years: {years}")
        
        # Historical emissions CSV Element values include the GHG type.
        # Read CH4 and N2O data separately.
        # Use standard dict_v3 Process names, including spaces.
        element_mappings = {
            # Enteric fermentation includes CH4 only.
            'Enteric fermentation (Emissions CH4)': ('Enteric fermentation', 'CH4'),
            # Manure management includes CH4 and N2O.
            'Manure management (Emissions CH4)': ('Manure management', 'CH4'),
            'Manure management (Emissions N2O)': ('Manure management', 'N2O'),
            # Manure applied to soils and left on pasture include N2O only.
            'Manure applied to soils (Emissions N2O)': ('Manure applied to soils', 'N2O'),
            'Manure left on pasture (Emissions N2O)': ('Manure left on pasture', 'N2O'),
        }
        
        # Result container: {process: {ghg_type: [dfs]}}.
        # Use standard Process names.
        results_by_ghg = {
            'Enteric fermentation': {'CH4': [], 'N2O': [], 'CO2': []},
            'Manure management': {'CH4': [], 'N2O': [], 'CO2': []},
            'Manure applied to soils': {'CH4': [], 'N2O': [], 'CO2': []},
            'Manure left on pasture': {'CH4': [], 'N2O': [], 'CO2': []},
        }
        
        # Select emissions data with Select=1.
        df = self.hist_emissions.copy()
        if 'Select' in df.columns:
            df = df[df['Select'] == 1]
        
        # Exclude countries with Region_label_new='no'.
        if 'M49_Country_Code' in df.columns:
            region_df = read_excel_cached(self.dict_v3_path, sheet_name='region')
            # Standardize M49 to 'xxx: a leading apostrophe and three digits.
            # Normalize leading-zero dict_v3 codes and potentially integer CSV codes consistently.
            def _normalize_m49_local(s):
                def normalize_single(val):
                    if pd.isna(val):
                        return None
                    return _normalize_m49_for_comparison(val)
                return s.apply(normalize_single)
            
            region_df['M49_Country_Code'] = _normalize_m49_local(region_df['M49_Country_Code'])
            # df (hist_emissions) was normalized in _load_historical_data; do not repeat.
            
            valid_m49_codes = region_df[region_df['Region_label_new'] != 'no']['M49_Country_Code'].unique()
            rows_before = len(df)
            df = df[df['M49_Country_Code'].isin(valid_m49_codes)].copy()
            rows_after = len(df)
            if rows_before > rows_after:
                logger.info(f"[INFO] 历史排放数据过滤掉 {rows_before - rows_after} 行无效国家")
        
        # Select livestock-related emissions processes.
        if 'Element' not in df.columns:
            logger.info("WARNING: historical emissions are missing the Element column; cannot read")
            return {k: None for k in results_by_ghg.keys()}
        
        df_livestock = df[df['Element'].isin(element_mappings.keys())].copy()
        
        if df_livestock.empty:
            logger.info("WARNING: historical emissions block not found in the source file")
            logger.info(f"  可用的Element: {df['Element'].unique()[:10]}")
            return {k: None for k in results_by_ghg.keys()}
        
        logger.info(f"[DEBUG] Found {len(df_livestock)} rows of livestock emissions (before Item filter)")

        merged_item_mask = df_livestock['Item'].astype(str).str.strip().isin(
            {'Buffalo', 'Buffaloes', 'Camels', 'Goats', 'Sheep'}
        )
        merged_row_count = int(merged_item_mask.sum())
        if merged_row_count > 0:
            logger.info(f"[INFO] Detected {merged_row_count} merged livestock history rows; splitting before dict_v3 Item filter")
            df_livestock = self._split_historical_livestock_items(df_livestock)
            logger.info(f"[DEBUG] After merged-item split: {len(df_livestock)} rows")
        
        # Filter Items using Item_Emis from dict_v3.
        # Keep only Items defined in dict_v3, excluding undefined entries such as 'Sheep and Goats'.
        if 'Item' in df_livestock.columns and hasattr(self, 'emis_mappings') and not self.emis_mappings.empty:
            valid_items = set(self.emis_mappings['Item_Emis'].dropna().unique())
            # Historical CSV Items use FAOSTAT names and need mapping to Item_Emis.
            # They may already use Item_Emis format, so first try direct filtering.
            items_before = len(df_livestock)
            items_unique_before = df_livestock['Item'].nunique()
            
            df_livestock = df_livestock[df_livestock['Item'].isin(valid_items)].copy()
            
            items_after = len(df_livestock)
            items_unique_after = df_livestock['Item'].nunique()
            
            if items_before > items_after:
                invalid_items = set(df[df['Element'].isin(element_mappings.keys())]['Item'].unique()) - valid_items
                logger.info(f"[INFO]  按dict_v3过滤Item: {items_before} ?? {items_after} 行 ({items_unique_before} ?? {items_unique_after} 种商品)")
                if invalid_items:
                    logger.info(f"   过滤掉的无效Item ({len(invalid_items)}种): {list(invalid_items)[:10]}")
        
        logger.info(f"[DEBUG] Found {len(df_livestock)} rows of livestock emissions (after Item filter)")
        
        # Convert to long format.
        year_cols = [c for c in df_livestock.columns if isinstance(c, str) and c.startswith('Y')]
        id_cols = [c for c in df_livestock.columns if c not in year_cols]
        
        df_long = df_livestock.melt(id_vars=id_cols, value_vars=year_cols,
                                     var_name='Year', value_name='Value')
        df_long['Year'] = df_long['Year'].str.lstrip('Y').astype(int)
        df_long = df_long[df_long['Year'].isin(years)]
        
        logger.info(f"[DEBUG] After filtering years {years}, got {len(df_long)} rows")
        
        # Group by process and GHG type.
        for element_name, (process, ghg_type) in element_mappings.items():
            process_df = df_long[df_long['Element'] == element_name].copy()
            
            if process_df.empty:
                continue
            
            logger.info(f"[DEBUG] {element_name}: {len(process_df)} rows")
            
            # Standardize column names.
            process_df = process_df.rename(columns={
                'M49_Country_Code': 'M49_Country_Code',
                'Item': 'Item',  # Preserve the Item column name; its CSV values are FAOSTAT names.
                'Year': 'year',
                'Value': f'{ghg_type}_kt'
            })
            
            # Add other GHG columns filled with zero.
            if ghg_type == 'CH4':
                process_df['N2O_kt'] = 0.0
                process_df['CO2_kt'] = 0.0
            elif ghg_type == 'N2O':
                process_df['CH4_kt'] = 0.0
                process_df['CO2_kt'] = 0.0
            else:  # CO2
                process_df['CH4_kt'] = 0.0
                process_df['N2O_kt'] = 0.0
            
            results_by_ghg[process][ghg_type].append(process_df[[
                'M49_Country_Code', 'Item', 'year', 'CH4_kt', 'N2O_kt', 'CO2_kt'
            ]])
        
        # Merge GHG data within each process, then combine processes.
        final_results = {}
        for process, ghg_dict in results_by_ghg.items():
            all_ghg_dfs = []
            for ghg_type, dfs in ghg_dict.items():
                if dfs:
                    all_ghg_dfs.extend(dfs)
            
            if all_ghg_dfs:
                # Merge different GHGs for the same process.
                combined = pd.concat(all_ghg_dfs, ignore_index=True)
                # Group by country, commodity, and year; sum CH4/N2O/CO2.
                result_df = combined.groupby(
                    ['M49_Country_Code', 'Item', 'year'], as_index=False
                ).agg({
                    'CH4_kt': 'sum',
                    'N2O_kt': 'sum',
                    'CO2_kt': 'sum'
                })
                # Item already contains FAOSTAT names; no renaming is needed.
                result_df['process'] = process  # Add the standard process name.
                final_results[process] = result_df
                logger.info(f"[DEBUG] {process}: final {len(final_results[process])} rows")
            else:
                final_results[process] = None
        
        return final_results
        
    def get_parameter_value(self, 
                           m49_code: int,
                           item: str,
                           process: str,
                           param_name: str,
                           year: int,
                           base_year: int = 2020) -> float:
        """
        Get a parameter value.
        
        Args:
            m49_code: M49 country code.
            item: Species name.
            process: Emissions process.
            param_name: Parameter name.
            year: Year.
            base_year: Baseline year for future BASE scenarios.
            
        Returns:
            Parameter value.
        """
        # Filter parameters.
        mask = (
            (self.gle_params['M49_Country_Code'] == m49_code) &
            (self.gle_params['Item'] == item) &
            (self.gle_params['Process'] == process) &
            (self.gle_params['paramName'] == param_name)
        )
        
        param_df = self.gle_params[mask]
        
        if param_df.empty:
            return 0.0
        
        # Read historical values directly.
        year_col = f'Y{year}'
        if year <= base_year and year_col in param_df.columns:
            value = param_df[year_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        # Use 2020 values for future BASE scenarios.
        base_col = f'Y{base_year}'
        if base_col in param_df.columns:
            value = param_df[base_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        return 0.0
    
    def get_parameter_value_by_species(self,
                                      m49_code: int,
                                      item: str,
                                      process: str,
                                      param_name: str,
                                      species: str,
                                      year: int,
                                      base_year: int = 2020) -> float:
        """
        Get a parameter value by gas species (CH4/N2O/CO2).
        
        Args:
            m49_code: M49 country code.
            item: Species name.
            process: Emissions process.
            param_name: Parameter name.
            species: Gas type ('CH4', 'N2O', 'CO2').
            year: Year.
            base_year: Baseline year.
            
        Returns:
            Parameter value.
        """
        # Filter parameters, including gas species.
        mask = (
            (self.gle_params['M49_Country_Code'] == m49_code) &
            (self.gle_params['Item'] == item) &
            (self.gle_params['Process'] == process) &
            (self.gle_params['paramName'] == param_name)
        )
        
        # GLE_parameters.xlsx distinguishes CH4/N2O using paramMMS, not Species.
        if 'paramMMS' in self.gle_params.columns:
            mask = mask & (self.gle_params['paramMMS'] == species)
        elif 'Species' in self.gle_params.columns:
            mask = mask & (self.gle_params['Species'] == species)
        
        param_df = self.gle_params[mask]
        
        if param_df.empty:
            return 0.0
        
        # Read historical values directly.
        year_col = f'Y{year}'
        if year <= base_year and year_col in param_df.columns:
            value = param_df[year_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        # Use 2020 values for future BASE scenarios.
        base_col = f'Y{base_year}'
        if base_col in param_df.columns:
            value = param_df[base_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        return 0.0

    def _build_parameter_table(self,
                               process: str,
                               param_name: str,
                               year: int,
                               *,
                               value_col: str,
                               base_year: int = 2020,
                               species: Optional[str] = None) -> pd.DataFrame:
        """Prepare a process/item parameter table for vectorized joins."""
        cache = getattr(self, '_parameter_table_cache', None)
        if cache is None:
            cache = {}
            self._parameter_table_cache = cache
        cache_key = (str(process), str(param_name), int(year), str(value_col), species, int(base_year))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if self.gle_params is None or self.gle_params.empty:
            return pd.DataFrame(columns=['M49_Country_Code', 'Item', value_col])
        params = self.gle_params[
            (self.gle_params['Process'] == process) &
            (self.gle_params['paramName'] == param_name)
        ].copy()
        if species is not None:
            if 'paramMMS' in params.columns:
                params = params[params['paramMMS'] == species].copy()
            elif 'Species' in params.columns:
                params = params[params['Species'] == species].copy()
        if params.empty:
            return pd.DataFrame(columns=['M49_Country_Code', 'Item', value_col])
        year_col = f'Y{year}'
        base_col = f'Y{base_year}'
        if year <= base_year and year_col in params.columns:
            series = params[year_col]
        elif base_col in params.columns:
            series = params[base_col]
        elif year_col in params.columns:
            series = params[year_col]
        else:
            return pd.DataFrame(columns=['M49_Country_Code', 'Item', value_col])
        params[value_col] = pd.to_numeric(series, errors='coerce')
        params = params.dropna(subset=[value_col])
        params = params[['M49_Country_Code', 'Item', value_col]].drop_duplicates(
            subset=['M49_Country_Code', 'Item'],
            keep='first'
        )
        cache[cache_key] = params
        return params

    def _get_hist_ratio_lookup_maps(
        self,
        *,
        element: str,
        year: int,
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[str, float]]:
        """Cache exact/region/global ratio lookups from hist_manure_stock for one element/year."""
        cache = getattr(self, '_hist_ratio_lookup_cache', None)
        if cache is None:
            cache = {}
            self._hist_ratio_lookup_cache = cache

        target_year = min(int(year), 2020)
        cache_key = (str(element), target_year)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        year_col = f'Y{target_year}'
        required_cols = {'M49_Country_Code', 'Item', 'Element', year_col}
        if self.hist_manure_stock.empty or not required_cols.issubset(self.hist_manure_stock.columns):
            empty = ({}, {}, {})
            cache[cache_key] = empty
            return empty

        df = self.hist_manure_stock[
            self.hist_manure_stock['Element'].astype(str).eq(str(element))
        ][['M49_Country_Code', 'Item', year_col]].copy()
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=[year_col])
        if df.empty:
            empty = ({}, {}, {})
            cache[cache_key] = empty
            return empty

        df['M49_Country_Code'] = df['M49_Country_Code'].apply(_normalize_m49_for_comparison)
        df['Item'] = df['Item'].astype(str)
        exact_map = {
            (str(m49), str(item)): float(value)
            for m49, item, value in df[['M49_Country_Code', 'Item', year_col]].itertuples(index=False, name=None)
        }

        region_map: Dict[Tuple[str, str], float] = {}
        if self.m49_to_region:
            df['ratio_region'] = df['M49_Country_Code'].map(self.m49_to_region)
            region_df = df.dropna(subset=['ratio_region']).groupby(['ratio_region', 'Item'], as_index=False)[year_col].mean()
            region_map = {
                (str(region), str(item)): float(value)
                for region, item, value in region_df[['ratio_region', 'Item', year_col]].itertuples(index=False, name=None)
            }

        global_df = df.groupby('Item', as_index=False)[year_col].mean()
        global_map = {
            str(item): float(value)
            for item, value in global_df[['Item', year_col]].itertuples(index=False, name=None)
        }

        cache[cache_key] = (exact_map, region_map, global_map)
        return cache[cache_key]

    def _lookup_hist_ratio_series(
        self,
        m49_series: pd.Series,
        item_series: pd.Series,
        *,
        year: int,
        element: str,
        default_value: float,
        allow_region_global: bool = False,
        count_fallbacks: bool = False,
    ) -> np.ndarray:
        """Fast per-row ratio lookup using cached dicts instead of repeated DataFrame scans."""
        target_year = min(int(year), 2020)
        target_exact, target_region, target_global = self._get_hist_ratio_lookup_maps(element=element, year=target_year)
        if target_year != 2020:
            base_exact, _, _ = self._get_hist_ratio_lookup_maps(element=element, year=2020)
        else:
            base_exact = target_exact

        results = np.empty(len(m49_series), dtype=float)
        region_fallbacks = 0
        global_fallbacks = 0
        default_fallbacks = 0

        for idx, (m49_raw, item_raw) in enumerate(zip(m49_series, item_series)):
            m49 = _normalize_m49_for_comparison(m49_raw)
            item = str(item_raw).strip()
            base_item = self._get_base_item_name(item)

            value = target_exact.get((m49, item))
            if value is None and base_item:
                value = target_exact.get((m49, base_item))
            if value is None and target_year != 2020:
                value = base_exact.get((m49, item))
                if value is None and base_item:
                    value = base_exact.get((m49, base_item))

            if value is None and allow_region_global:
                region = self.m49_to_region.get(m49)
                if region:
                    value = target_region.get((region, item))
                    if value is None and base_item:
                        value = target_region.get((region, base_item))
                    if value is not None:
                        region_fallbacks += 1
                if value is None:
                    value = target_global.get(item)
                    if value is None and base_item:
                        value = target_global.get(base_item)
                    if value is not None:
                        global_fallbacks += 1

            if value is None:
                value = default_value
                if allow_region_global:
                    default_fallbacks += 1

            results[idx] = float(value)

        if count_fallbacks and allow_region_global:
            self._manure_ratio_fallback_counts['region'] += region_fallbacks
            self._manure_ratio_fallback_counts['global'] += global_fallbacks
            self._manure_ratio_fallback_counts['default'] += default_fallbacks

        return results

    def _prepare_production_agg(
        self,
        production_df: pd.DataFrame,
        *,
        year: int,
        allowed_items: List[str],
        label: str,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Prepare one-year production aggregated by country-item for vectorized livestock calculations."""
        logger.info(f"\n=== {label} 调试 (year={year}) ===")
        logger.info(f"[DEBUG] 输入production_df: {len(production_df)} 行")
        if not production_df.empty:
            logger.info(f"[DEBUG] 列名: {list(production_df.columns)}")
            logger.info(f"[DEBUG] 年份分布: {production_df['year'].unique() if 'year' in production_df.columns else 'N/A'}")

        if 'year' in production_df.columns:
            production_df = production_df[production_df['year'] == year].copy()
            logger.info(f"[DEBUG] 筛选year={year}后: {len(production_df)} 行")
            if production_df.empty:
                logger.warning(f" WARNING: no data after filtering year={year}; skipping")
                return pd.DataFrame(), None
        else:
            logger.warning(" WARNING: production_df没有year列！")

        commodity_col = 'Commodity' if 'Commodity' in production_df.columns else (
            'commodity' if 'commodity' in production_df.columns else 'Item'
        )
        if commodity_col not in production_df.columns:
            logger.info("WARNING: commodity column not found")
            return pd.DataFrame(), None
        if 'M49_Country_Code' not in production_df.columns or 'production_t' not in production_df.columns:
            logger.warning("WARNING: production_df缺少M49_Country_Code或production_t列")
            return pd.DataFrame(), commodity_col

        allowed_set = {str(item).strip() for item in allowed_items if str(item).strip()}
        work = production_df[['M49_Country_Code', commodity_col, 'production_t']].copy()
        work[commodity_col] = work[commodity_col].astype(str).str.strip()
        work = work[work[commodity_col].isin(allowed_set)].copy()
        logger.info(f"{label}候选商品筛选后: {len(work)} 行")
        if work.empty:
            return pd.DataFrame(), commodity_col

        work['M49_Country_Code'] = work['M49_Country_Code'].apply(_normalize_m49_for_comparison)
        work = work[work['M49_Country_Code'].astype(str).str.strip() != ''].copy()
        if hasattr(self, 'valid_m49_codes') and self.valid_m49_codes:
            work = work[work['M49_Country_Code'].isin(self.valid_m49_codes)].copy()
        work['production_t'] = pd.to_numeric(work['production_t'], errors='coerce')
        work = work.dropna(subset=['production_t'])
        work = work[work['production_t'] > 0].copy()
        if work.empty:
            return pd.DataFrame(), commodity_col

        grouped = (
            work.groupby(['M49_Country_Code', commodity_col], as_index=False)['production_t']
            .sum()
            .rename(columns={commodity_col: 'Item'})
        )
        logger.info(f"{label}聚合后: {len(grouped)} 行, 国家={grouped['M49_Country_Code'].nunique()}, 商品={grouped['Item'].nunique()}")
        return grouped, commodity_col
    
    def calculate_milk_animals(self,
                               production_df: pd.DataFrame,
                               year: int) -> pd.DataFrame:
        """
        Calculate milk-producing animal counts from milk production, including cattle, sheep/goats, camels, etc.
        
        Args:
            production_df: Production DataFrame whose Commodity column uses FAOSTAT Item_Production_Map names.
                          For example, 'Raw milk of cattle', not 'Cattle, dairy'.
            year: Year.
            
        Returns:
            DataFrame containing milk-producing animal counts.
        """
        # Identify all dairy commodities from the mapping.
        # Dairy includes milk-producing buffalo/camels/cattle/goats/sheep and egg-laying Chickens, layers.
        dairy_production_items = []  # FAOSTAT production-data names
        for item_emis, mapping in self.item_production_map.items():
            # Select milk or egg commodities and exclude non-dairy entries.
            item_lower = item_emis.lower()
            if ('dairy' in item_lower and 'non-dairy' not in item_lower) or 'layers' in item_lower:
                fao_item = mapping['Item']  # Names from Item_Production_Map
                dairy_production_items.append(fao_item)
        
        logger.info(f"识别的dairy FAOSTAT商品 ({len(dairy_production_items)} 种): {dairy_production_items}")

        grouped, _ = self._prepare_production_agg(
            production_df,
            year=year,
            allowed_items=dairy_production_items,
            label='calculate_milk_animals',
        )
        if grouped.empty:
            logger.warning("WARNING: production_df has no dairy items")
            return pd.DataFrame()

        grouped['Item_Emis'] = grouped['Item'].map(self.production_to_emis)
        missing_prod_map = grouped['Item_Emis'].isna().sum()
        if missing_prod_map:
            logger.info(f"  WARNING: {missing_prod_map} 条dairy记录未找到 production mapping")
        grouped = grouped.dropna(subset=['Item_Emis']).copy()
        if grouped.empty:
            return pd.DataFrame()

        yield_map_rows = []
        for item_emis, mapping in self.item_yield_map.items():
            yield_map_rows.append({
                'Item_Emis': str(item_emis),
                'Yield_Item': str(mapping.get('Item', '') or '').strip(),
                'Yield_Element': str(mapping.get('Element', 'Yield') or 'Yield').strip(),
            })
        yield_map_df = pd.DataFrame(yield_map_rows).drop_duplicates(subset=['Item_Emis'])
        grouped = grouped.merge(yield_map_df, on='Item_Emis', how='left')
        missing_yield_map = grouped['Yield_Item'].isna().sum()
        if missing_yield_map:
            logger.info(f"  WARNING: {missing_yield_map} 条dairy记录缺少 yield mapping")
        grouped = grouped.dropna(subset=['Yield_Item']).copy()
        if grouped.empty:
            return pd.DataFrame()

        lookup_keys = grouped[['M49_Country_Code', 'Item_Emis', 'Yield_Item', 'Yield_Element']].drop_duplicates().copy()
        lookup_keys['yield_t_per_head'] = [
            self.get_yield_value(m49_code, yield_item, yield_element, year)
            for m49_code, _, yield_item, yield_element in lookup_keys.itertuples(index=False, name=None)
        ]
        grouped = grouped.merge(
            lookup_keys,
            on=['M49_Country_Code', 'Item_Emis', 'Yield_Item', 'Yield_Element'],
            how='left',
        )

        default_yields = {
            'Buffalo, dairy': 1.5,
            'Camel, dairy': 1.2,
            'Goats, dairy': 0.5,
            'Sheep, dairy': 0.4,
            'Cattle, dairy': 6.0,
            'Chickens, layers': 0.015,
        }
        missing_yield_mask = pd.to_numeric(grouped['yield_t_per_head'], errors='coerce').fillna(0.0) <= 0
        if missing_yield_mask.any():
            grouped.loc[missing_yield_mask, 'yield_t_per_head'] = (
                grouped.loc[missing_yield_mask, 'Item_Emis']
                .map(default_yields)
                .fillna(1.0)
            )
            logger.info(f"  [DAIRY_FIX] {int(missing_yield_mask.sum())} 条dairy记录使用默认yield")

        grouped['yield_multiplier'] = [
            self._get_yield_multiplier(m49_code, item_emis, year)
            for m49_code, item_emis in grouped[['M49_Country_Code', 'Item_Emis']].itertuples(index=False, name=None)
        ]
        if year >= 2020:
            reference = self.scenario_params.get('animal_yield_reference', {})
            for row_index, row in grouped.iterrows():
                reference_value = reference.get((_normalize_m49_for_comparison(row.M49_Country_Code), str(row.Item_Emis)))
                if reference_value is not None:
                    grouped.at[row_index, 'yield_t_per_head'] = float(reference_value)
        grouped['yield_t_per_head'] = pd.to_numeric(grouped['yield_t_per_head'], errors='coerce') * pd.to_numeric(grouped['yield_multiplier'], errors='coerce').fillna(1.0)
        grouped = grouped[grouped['yield_t_per_head'] > 0].copy()
        if grouped.empty:
            return pd.DataFrame()

        grouped['year'] = int(year)
        grouped['producing_animals'] = grouped['production_t'] / grouped['yield_t_per_head']
        result = grouped[['M49_Country_Code', 'Item', 'Item_Emis', 'year', 'production_t', 'yield_t_per_head']].copy()
        layer_mask = result['Item_Emis'].astype(str).str.lower().str.contains('layer', na=False)
        result.loc[~layer_mask, 'milk_animals'] = grouped.loc[~layer_mask, 'producing_animals'].to_numpy()
        result.loc[layer_mask, 'laying'] = grouped.loc[layer_mask, 'producing_animals'].to_numpy()

        logger.info(f"计算得到 {len(result)} 个dairy动物流量记录")
        return result
    
    def calculate_meat_animals(self,
                               production_df: pd.DataFrame,
                               year: int) -> pd.DataFrame:
        """
        Calculate slaughtered animal counts from meat production.
        
        Args:
            production_df: Production DataFrame whose Commodity column uses FAOSTAT Item_Production_Map names.
                          For example, 'Meat of pig', not 'Swine'.
            year: Year.
            
        Returns:
            DataFrame containing slaughtered animal counts.
        """
        # Identify meat commodities from the reverse mapping.
        # Meat commodities have Item_Cat2='Meat' or 'non-dairy' in Item_Emis.
        meat_production_items = []  # FAOSTAT production-data names
        for item_emis, mapping in self.item_production_map.items():
            # Check keywords in commodity names.
            item_lower = item_emis.lower()
            if ('non-dairy' in item_lower or 
                'broiler' in item_lower or
                item_emis in ['Asses', 'Ducks', 'Horses', 'Llamas', 'Mules and hinnies', 
                             'Swine', 'Turkeys']):
                fao_item = mapping['Item']  # Names from Item_Production_Map
                meat_production_items.append(fao_item)
        
        logger.info(f"识别的肉类FAOSTAT商品 ({len(meat_production_items)} 种): {meat_production_items}")

        grouped, _ = self._prepare_production_agg(
            production_df,
            year=year,
            allowed_items=meat_production_items,
            label='calculate_meat_animals',
        )
        if grouped.empty:
            logger.info("WARNING: production_df has no carcass items")
            return pd.DataFrame()

        grouped['Item_Emis'] = grouped['Item'].map(self.production_to_emis)
        missing_prod_map = grouped['Item_Emis'].isna().sum()
        if missing_prod_map:
            logger.info(f"  WARNING: {missing_prod_map} 条meat记录未找到 production mapping")
        grouped = grouped.dropna(subset=['Item_Emis']).copy()
        if grouped.empty:
            return pd.DataFrame()

        yield_map_rows = []
        for item_emis, mapping in self.item_yield_map.items():
            yield_map_rows.append({
                'Item_Emis': str(item_emis),
                'Yield_Item': str(mapping.get('Item', '') or '').strip(),
                'Yield_Element': str(mapping.get('Element', 'Yield') or 'Yield').strip(),
            })
        yield_map_df = pd.DataFrame(yield_map_rows).drop_duplicates(subset=['Item_Emis'])
        grouped = grouped.merge(yield_map_df, on='Item_Emis', how='left')
        missing_yield_map = grouped['Yield_Item'].isna().sum()
        if missing_yield_map:
            logger.info(f"  WARNING: {missing_yield_map} 条meat记录缺少 carcass mapping")
        grouped = grouped.dropna(subset=['Yield_Item']).copy()
        if grouped.empty:
            return pd.DataFrame()

        lookup_keys = grouped[['M49_Country_Code', 'Item_Emis', 'Yield_Item', 'Yield_Element']].drop_duplicates().copy()
        lookup_keys['carcass_weight_t'] = [
            self.get_carcass_weight(m49_code, yield_item, yield_element, year)
            for m49_code, _, yield_item, yield_element in lookup_keys.itertuples(index=False, name=None)
        ]
        grouped = grouped.merge(
            lookup_keys,
            on=['M49_Country_Code', 'Item_Emis', 'Yield_Item', 'Yield_Element'],
            how='left',
        )

        grouped['yield_multiplier'] = [
            self._get_yield_multiplier(m49_code, item_emis, year)
            for m49_code, item_emis in grouped[['M49_Country_Code', 'Item_Emis']].itertuples(index=False, name=None)
        ]
        if year >= 2020:
            reference = self.scenario_params.get('animal_yield_reference', {})
            for row_index, row in grouped.iterrows():
                reference_value = reference.get((_normalize_m49_for_comparison(row.M49_Country_Code), str(row.Item_Emis)))
                if reference_value is not None:
                    grouped.at[row_index, 'carcass_weight_t'] = float(reference_value)
        grouped['carcass_weight_t'] = pd.to_numeric(grouped['carcass_weight_t'], errors='coerce') * pd.to_numeric(grouped['yield_multiplier'], errors='coerce').fillna(1.0)
        missing_yield = (grouped['carcass_weight_t'].fillna(0.0) <= 0).sum()
        if missing_yield:
            logger.warning(f"  WARNING: {int(missing_yield)} 条meat记录仍缺少 yield value，将跳过")
        grouped = grouped[grouped['carcass_weight_t'] > 0].copy()
        if grouped.empty:
            return pd.DataFrame()

        grouped['year'] = int(year)
        grouped['slaughtered'] = grouped['production_t'] / grouped['carcass_weight_t']
        result = grouped[['M49_Country_Code', 'Item', 'Item_Emis', 'year', 'production_t', 'carcass_weight_t', 'slaughtered']].copy()

        logger.info(f"计算得到 {len(result)} 个肉类动物流量记录")
        return result
    
    def calculate_egg_animals(self,
                             production_df: pd.DataFrame,
                             year: int) -> pd.DataFrame:
        """
        Calculate laying animal counts from egg production.
        
        Args:
            production_df: Egg-production DataFrame with FAOSTAT Item_Production_Map names in Commodity.
                          For example, 'Eggs from hens', not 'Chickens, eggs'.
            year: Year.
            
        Returns:
            DataFrame containing laying animal counts.
        """
        # Information
        logger.info(f"\n=== calculate_egg_animals 调试 (year={year}) ===")
        
        # Determine the commodity column name.
        commodity_col = 'Commodity' if 'Commodity' in production_df.columns else (
            'commodity' if 'commodity' in production_df.columns else 'Item'
        )
        
        # Egg commodities using FAOSTAT production-data names
        egg_production_items = []
        for item_emis, mapping in self.item_production_map.items():
            if 'egg' in item_emis.lower():
                fao_item = mapping['Item']
                egg_production_items.append(fao_item)
        
        logger.info(f"识别的蛋类FAOSTAT商品: {egg_production_items}")
        
        # Check which egg commodities appear in production_df.
        available_items = production_df[commodity_col].unique()
        matching_items = set(available_items) & set(egg_production_items)
        logger.info(f"production_df中的蛋类商品: {list(matching_items)}")
        
        results = []
        
        # Calculate only for valid countries.
        for m49_code in production_df['M49_Country_Code'].unique():
            # Normalize M49 format before comparing.
            if hasattr(self, 'valid_m49_codes') and self.valid_m49_codes:
                m49_normalized = _normalize_m49_for_comparison(m49_code)
                if m49_normalized not in self.valid_m49_codes:
                    continue
            
            country_prod = production_df[production_df['M49_Country_Code'] == m49_code]
            
            # Iterate over all egg commodities using FAOSTAT Item_Production_Map names.
            for fao_item in matching_items:
                egg_data = country_prod[country_prod[commodity_col] == fao_item]
                
                if egg_data.empty:
                    continue
                
                egg_production = egg_data['production_t'].sum()
                
                if egg_production <= 0:
                    continue
                
                # Reverse-look up Item_Emis.
                if fao_item not in self.production_to_emis:
                    logger.info(f"  WARNING: {fao_item} not found in production mapping")
                    continue
                
                item_emis = self.production_to_emis[fao_item]
                
                # Get laying rate using Item_Yield_Map.
                if item_emis not in self.item_yield_map:
                    logger.info(f"  WARNING: {item_emis} missing yield mapping (layers)")
                    continue
                
                laying_fao_item = self.item_yield_map[item_emis]['Item']
                laying_element = self.item_yield_map[item_emis].get('Element', 'Yield')
                laying_rate = self.get_laying_rate(m49_code, laying_fao_item, laying_element, year)
                
                if laying_rate > 0:
                    # Laying animal count = egg production / laying rate.
                    laying = egg_production / laying_rate
                    
                    results.append({
                        'M49_Country_Code': m49_code,
                        'Item': fao_item,
                        'Item_Emis': item_emis,
                        'year': year,
                        'production_t': egg_production,
                        'laying_rate_t_per_head': laying_rate,
                        'laying': laying
                    })
                else:
                    logger.warning(f"  WARNING: {item_emis} ({laying_fao_item}) has no production data")
        
        logger.info(f"计算得到 {len(results)} 个蛋类动物流量记录")
        return pd.DataFrame(results)
    
    def get_carcass_weight(self, m49_code: int, item: str, element: str, year: int) -> float:
        """
        Get carcass weight (meat yield).
        Priority: country-specific -> regional mean -> global mean.
        
         FAOSTAT carcass weight is originally in kg/head;
        this method converts and returns t/head (tonnes per head).
        
        Args:
            m49_code: M49 country code.
            item: FAOSTAT Item name from Item_Yield_Map.
            element: Element name from Item_Yield_Element.
            year: Year.
            
        Returns:
            float: Carcass weight (t/head).
        """
        year_col = f'Y{year}' if year <= 2020 else 'Y2020'
        
        # Convert FAOSTAT carcass weight from kg/head to t/head.
        KG_TO_TONNE = 1.0 / 1000.0

        value, _ = self._lookup_hist_row_value(m49_code, item, element, year_col)
        if value is not None:
            return value * KG_TO_TONNE

        region = self.m49_to_region.get(_normalize_m49_for_comparison(m49_code))
        region_avg = self._lookup_hist_region_mean(m49_code, item, element, year_col)
        if region_avg is not None and region_avg > 0:
            region_avg_t = region_avg * KG_TO_TONNE
            logger.debug(f"   {item} M49={m49_code} 使用{region}区域平均胴体重: {region_avg_t:.4f} t/head")
            return region_avg_t

        global_avg = self._lookup_hist_global_mean(item, element, year_col)
        if global_avg is not None and global_avg > 0:
            global_avg_t = global_avg * KG_TO_TONNE
            logger.debug(f"   {item} M49={m49_code} 使用全球平均胴体重: {global_avg_t:.4f} t/head")
            return global_avg_t

        return 0.0
    
    def get_laying_rate(self, m49_code: int, item: str, element: str, year: int) -> float:
        """
        Get the laying rate.
        Priority: country-specific -> regional mean -> global mean.
        
        Args:
            m49_code: M49 country code.
            item: FAOSTAT Item name from Item_Yield_Map.
            element: Element name from Item_Yield_Element.
            year: Year.
        """
        year_col = f'Y{year}' if year <= 2020 else 'Y2020'
        
        value, _ = self._lookup_hist_row_value(m49_code, item, element, year_col)
        if value is not None:
            return value

        region = self.m49_to_region.get(_normalize_m49_for_comparison(m49_code))
        region_avg = self._lookup_hist_region_mean(m49_code, item, element, year_col)
        if region_avg is not None and region_avg > 0:
            logger.debug(f"   {item} M49={m49_code} 使用{region}区域平均产蛋率: {region_avg:.4f} t/head")
            return region_avg

        global_avg = self._lookup_hist_global_mean(item, element, year_col)
        if global_avg is not None and global_avg > 0:
            logger.debug(f"   {item} M49={m49_code} 使用全球平均产蛋率: {global_avg:.4f} t/head")
            return global_avg

        return 0.0
    
    def get_yield_value(self, m49_code: int, item: str, element: str, year: int) -> float:
        """
        Get milk/egg yield.
        Priority: country-specific -> regional mean -> global mean.
        
         FAOSTAT milk yield is usually supplied in hg/head or kg/head;
        this method converts and returns t/head (tonnes per head).
        
        Args:
            m49_code: M49 country code.
            item: FAOSTAT Item name from Item_Yield_Map.
            element: Element name from Item_Yield_Element.
            year: Year.
            
        Returns:
            float: Yield (t/head/year).
        """
        year_col = f'Y{year}' if year <= 2020 else 'Y2020'
        
        def _convert_yield_unit(value: float, unit_str: str, item_name: str) -> float:
            """Convert FAOSTAT yield units to t/head."""
            if not np.isfinite(value) or value <= 0:
                return 0.0
            u = (unit_str or '').strip().lower()
            item_lower = (item_name or '').lower()
            
            # Milk yield is usually in hg/head or kg/head.
            if 'milk' in item_lower:
                if u in {'hg/an', 'hg/animal', 'hg', 'hectogram', '100g/an'}:
                    # hg (100 g) -> t: divide by 10,000.
                    return value / 10000.0
                if u in {'kg/an', 'kg/animal', 'kg', 'kilogram'}:
                    # kg -> t: divide by 1,000.
                    return value / 1000.0
                if u in {'t/an', 't/animal', 't', 'tonne'}:
                    return value
                # Default to hg/head, the standard FAOSTAT milk-yield unit.
                return value / 10000.0
            
            # Egg yield
            if 'egg' in item_lower:
                if u in {'kg/an', 'kg/animal', 'kg'}:
                    return value / 1000.0
                if u in {'t/an', 't/animal', 't'}:
                    return value
                # Default to kg/head.
                return value / 1000.0
            
            # Other livestock yields, if this branch is reached
            if 'kg' in u:
                return value / 1000.0
            if 'hg' in u or '100g' in u:
                return value / 10000.0
            
            # Conservatively assume kg units.
            return value / 1000.0
        
        value, unit_str = self._lookup_hist_row_value(m49_code, item, element, year_col)
        if value is not None:
            return _convert_yield_unit(float(value), str(unit_str), item)

        if element == 'Yield':
            prod_value, _ = self._lookup_hist_row_value(m49_code, item, 'Production', year_col)
            animals_value, _ = self._lookup_hist_row_value(m49_code, item, 'Milk Animals', year_col)
            if prod_value is not None and animals_value is not None and prod_value > 0 and animals_value > 0:
                calculated_yield = float(prod_value) / float(animals_value)
                logger.info(f"  [YIELD_CALC] {item} M49={m49_code}: 从Production({prod_value:.0f}t) / Milk Animals({animals_value:.0f}head) 计算yield = {calculated_yield:.4f} t/head")
                return calculated_yield

        region = self.m49_to_region.get(_normalize_m49_for_comparison(m49_code))
        region_avg = self._lookup_hist_region_mean(m49_code, item, element, year_col)
        unit_fallback = self._hist_unit_by_item_element.get((str(item).strip(), str(element).strip()), '')
        if region_avg is not None and region_avg > 0:
            region_avg_t = _convert_yield_unit(region_avg, str(unit_fallback), item)
            logger.debug(f"   {item} M49={m49_code} 使用{region}区域平均产率: {region_avg_t:.6f} t/head")
            return float(region_avg_t)

        global_avg = self._lookup_hist_global_mean(item, element, year_col)
        if global_avg is not None and global_avg > 0:
            global_avg_t = _convert_yield_unit(global_avg, str(unit_fallback), item)
            logger.debug(f"   {item} M49={m49_code} 使用全球平均产率: {global_avg_t:.6f} t/head")
            return float(global_avg_t)

        return 0.0
    def calculate_stock_from_animals(self,
                                    animals_df: pd.DataFrame,
                                    year: int) -> pd.DataFrame:
        """
        Calculate standing stocks from animal flows.
        
        Args:
            animals_df: DataFrame containing producing_animals/slaughtered/milk_animals/laying.
            year: Year.
            
        Returns:
            DataFrame containing stocks.
        """
        logger.info(f"\n--- 计算存栏 (year={year}) ---")
        logger.info(f"输入animals_df: {len(animals_df)} 行")
        if not animals_df.empty:
            logger.info(f"列名: {list(animals_df.columns)}")
        
        work = animals_df.copy()
        work['M49_Country_Code'] = work['M49_Country_Code'].apply(_normalize_m49_for_comparison)
        work['Item'] = work['Item'].astype(str)
        if 'Item_Emis' not in work.columns:
            work['Item_Emis'] = work['Item']
        work['Item_Emis'] = work['Item_Emis'].astype(str)

        for col in ['milk_animals', 'slaughtered', 'laying', 'producing_animals']:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0.0)
            else:
                work[col] = 0.0

        ratio_item_map = {
            item_emis: mapping.get('Item')
            for item_emis, mapping in self.item_slaughtered_ratio_map.items()
            if mapping.get('Item') is not None
        }
        ratio_element_map = {
            item_emis: mapping.get('Element', 'Producing/Slaughtered ratio')
            for item_emis, mapping in self.item_slaughtered_ratio_map.items()
        }
        work['ratio_item'] = work['Item_Emis'].map(ratio_item_map)
        work['ratio_element'] = work['Item_Emis'].map(ratio_element_map).fillna('Producing/Slaughtered ratio')
        work['ratio_region'] = work['M49_Country_Code'].map(self.m49_to_region)

        exact_ratio_df, region_ratio_df, global_ratio_df = self._get_slaughter_ratio_lookup_frames(year)
        if not exact_ratio_df.empty:
            work = work.merge(
                exact_ratio_df,
                on=['M49_Country_Code', 'ratio_item', 'ratio_element'],
                how='left',
            )
        else:
            work['exact_ratio'] = np.nan
        if not region_ratio_df.empty:
            work = work.merge(
                region_ratio_df,
                on=['ratio_region', 'ratio_item', 'ratio_element'],
                how='left',
            )
        else:
            work['region_ratio'] = np.nan
        if not global_ratio_df.empty:
            work = work.merge(
                global_ratio_df,
                on=['ratio_item', 'ratio_element'],
                how='left',
            )
        else:
            work['global_ratio'] = np.nan

        work['ratio_val'] = work['exact_ratio']
        missing_ratio = work['ratio_val'].isna()
        if missing_ratio.any():
            work.loc[missing_ratio, 'ratio_val'] = work.loc[missing_ratio, 'region_ratio']
        missing_ratio = work['ratio_val'].isna()
        if missing_ratio.any():
            work.loc[missing_ratio, 'ratio_val'] = work.loc[missing_ratio, 'global_ratio']
        work['ratio_val'] = pd.to_numeric(work['ratio_val'], errors='coerce').fillna(1.0)
        work.loc[work['ratio_val'] <= 0, 'ratio_val'] = 1.0

        # B04 reference ratios preserve observed 2020 production and stock.
        # Ratios are animal flows / standing herd and may legitimately exceed 1.
        reference = (self.scenario_params or {}).get('stock_ratio_reference', {})
        if reference and year >= 2020:
            override = pd.Series([reference.get((str(c),str(j)), np.nan)
                                  for c,j in zip(work.M49_Country_Code,work.Item_Emis)], index=work.index)
            identified = override.notna()
            if ((override[identified] <= 0) | ~np.isfinite(override[identified])).any():
                raise ValueError('Invalid B04 reference stock conversion ratio')
            work.loc[identified,'ratio_val'] = override[identified]

        milk = work['milk_animals'].to_numpy(dtype=float)
        slaughtered = work['slaughtered'].to_numpy(dtype=float)
        laying = work['laying'].to_numpy(dtype=float)
        producing = work['producing_animals'].to_numpy(dtype=float)
        ratio = work['ratio_val'].to_numpy(dtype=float)

        conditions = [milk > 0, slaughtered > 0, laying > 0, producing > 0]
        stock_values = [milk / ratio, slaughtered / ratio, laying / ratio, producing / ratio]
        animal_types = ['milk', 'meat', 'egg', 'producing']
        work['stock'] = np.select(conditions, stock_values, default=0.0)
        work['animal_type'] = np.select(conditions, animal_types, default='')

        result_df = work.loc[work['stock'] > 0, ['M49_Country_Code', 'Item', 'Item_Emis']].copy()
        result_df['year'] = year
        result_df['stock'] = work.loc[work['stock'] > 0, 'stock'].to_numpy()
        result_df['animal_type'] = work.loc[work['stock'] > 0, 'animal_type'].to_numpy()
        # B04: independent background herds enter both feed and emissions once.
        if year >= 2020:
            background = self.scenario_params.get('background_stock_reference', {})
            if background:
                extra = pd.DataFrame([dict(M49_Country_Code=_normalize_m49_for_comparison(c),
                    Item=j,Item_Emis=j,year=year,stock=float(heads),animal_type='background')
                    for (c,j),heads in background.items() if float(heads)>0])
                if not extra.empty:
                    bg_keys=set(zip(extra.M49_Country_Code,extra.Item_Emis))
                    keep=[(c,j) not in bg_keys for c,j in zip(result_df.M49_Country_Code,result_df.Item_Emis)]
                    result_df=pd.concat([result_df.loc[keep],extra],ignore_index=True)

        logger.info(f"输出stock_df: {len(result_df)} 行")
        if not result_df.empty:
            logger.info(f"  奶类: {len(result_df[result_df['animal_type']=='milk'])} 行")
            logger.info(f"  肉类: {len(result_df[result_df['animal_type']=='meat'])} 行")
            logger.info(f"  蛋类: {len(result_df[result_df['animal_type']=='egg'])} 行")
        
        return result_df

    def _get_slaughter_ratio_lookup_frames(
        self,
        year: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cache country/regional/global slaughter-ratio lookups by year."""
        cache = getattr(self, '_slaughter_ratio_lookup_cache', None)
        if cache is None:
            cache = {}
            self._slaughter_ratio_lookup_cache = cache

        target_year = int(year) if int(year) <= 2020 else 2020
        cached = cache.get(target_year)
        if cached is not None:
            return cached

        year_col = f'Y{target_year}'
        required_cols = {'Area Code (M49)', 'Item', 'Element', year_col}
        if self.hist_production.empty or not required_cols.issubset(self.hist_production.columns):
            empty_exact = pd.DataFrame(columns=['M49_Country_Code', 'ratio_item', 'ratio_element', 'exact_ratio'])
            empty_region = pd.DataFrame(columns=['ratio_region', 'ratio_item', 'ratio_element', 'region_ratio'])
            empty_global = pd.DataFrame(columns=['ratio_item', 'ratio_element', 'global_ratio'])
            cache[target_year] = (empty_exact, empty_region, empty_global)
            return cache[target_year]

        ratio_hist = self.hist_production[['Area Code (M49)', 'Item', 'Element', year_col]].copy()
        ratio_hist[year_col] = pd.to_numeric(ratio_hist[year_col], errors='coerce')
        ratio_hist = ratio_hist.dropna(subset=[year_col]).copy()
        if ratio_hist.empty:
            empty_exact = pd.DataFrame(columns=['M49_Country_Code', 'ratio_item', 'ratio_element', 'exact_ratio'])
            empty_region = pd.DataFrame(columns=['ratio_region', 'ratio_item', 'ratio_element', 'region_ratio'])
            empty_global = pd.DataFrame(columns=['ratio_item', 'ratio_element', 'global_ratio'])
            cache[target_year] = (empty_exact, empty_region, empty_global)
            return cache[target_year]

        exact_ratio_df = (
            ratio_hist.rename(
                columns={
                    'Area Code (M49)': 'M49_Country_Code',
                    'Item': 'ratio_item',
                    'Element': 'ratio_element',
                    year_col: 'exact_ratio',
                }
            )[['M49_Country_Code', 'ratio_item', 'ratio_element', 'exact_ratio']]
            .drop_duplicates(subset=['M49_Country_Code', 'ratio_item', 'ratio_element'])
        )

        region_map_df = pd.DataFrame(
            {
                'M49_Country_Code': list(self.m49_to_region.keys()),
                'ratio_region': list(self.m49_to_region.values()),
            }
        )
        region_ratio_df = pd.DataFrame(columns=['ratio_region', 'ratio_item', 'ratio_element', 'region_ratio'])
        if not region_map_df.empty and not exact_ratio_df.empty:
            region_ratio_df = (
                exact_ratio_df.merge(region_map_df, on='M49_Country_Code', how='left')
                .dropna(subset=['ratio_region'])
                .groupby(['ratio_region', 'ratio_item', 'ratio_element'], as_index=False)['exact_ratio']
                .mean()
                .rename(columns={'exact_ratio': 'region_ratio'})
            )

        global_ratio_df = (
            exact_ratio_df.groupby(['ratio_item', 'ratio_element'], as_index=False)['exact_ratio']
            .mean()
            .rename(columns={'exact_ratio': 'global_ratio'})
        )

        cache[target_year] = (exact_ratio_df, region_ratio_df, global_ratio_df)
        return cache[target_year]
    
    def get_slaughtered_ratio(self, m49_code: int, item: str, element: str, year: int) -> float:
        """
        Get the producing/slaughtered ratio.
        Priority: country-specific -> regional mean -> global mean -> default 1.0.
        
        Args:
            m49_code: M49 country code.
            item: FAOSTAT Item name from Item_SlaughteredRatio_Map.
            element: Element from Item_SlaughteredRatio_Element, e.g. 'Producing/Slaughtered ratio'.
            year: Year.
        """
        year_col = f'Y{year}' if year <= 2020 else 'Y2020'
        
        # 1. Try country-specific data.
        hist_ratio = self.hist_production[
            (self.hist_production['Area Code (M49)'] == m49_code) &
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        if not hist_ratio.empty and year_col in hist_ratio.columns:
            value = hist_ratio[year_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        # 2. If missing, try the regional mean.
        region = self.m49_to_region.get(m49_code)
        if region:
            region_m49_codes = [m49 for m49, reg in self.m49_to_region.items() if reg == region]
            
            if region_m49_codes:
                region_data = self.hist_production[
                    (self.hist_production['Area Code (M49)'].isin(region_m49_codes)) &
                    (self.hist_production['Item'] == item) &
                    (self.hist_production['Element'] == element)
                ]
                
                if not region_data.empty and year_col in region_data.columns:
                    valid_values = region_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(valid_values) > 0:
                        region_avg = valid_values.mean()
                        logger.debug(f"   {item} M49={m49_code} 使用{region}区域平均屠宰比例: {region_avg:.4f} (基于{len(valid_values)}个国家)")
                        return float(region_avg)
        
        # 3. If regional data are also missing, use the global mean.
        global_data = self.hist_production[
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        if not global_data.empty and year_col in global_data.columns:
            valid_values = global_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(valid_values) > 0:
                global_avg = valid_values.mean()
                logger.debug(f"   {item} M49={m49_code} 使用全球平均屠宰比例: {global_avg:.4f} (基于{len(valid_values)}个国家)")
                return float(global_avg)
        
        return 1.0  # Default ratio: 1.
    
    def calculate_enteric_fermentation(self,
                                      stock_df: pd.DataFrame,
                                      year: int,
                                      scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate enteric fermentation emissions.
        
        Args:
            stock_df: DataFrame containing livestock stocks.
            year: Year.
            scenario_params: Scenario parameters.
            
        Returns:
            Emissions DataFrame.
        """
        if stock_df is None or stock_df.empty:
            return pd.DataFrame()

        work = stock_df.copy()
        work['M49_Country_Code'] = work['M49_Country_Code'].apply(_normalize_m49_for_comparison)
        if 'Item_Emis' not in work.columns:
            work['Item_Emis'] = work['Item']
        work['Item_Emis'] = work['Item_Emis'].astype(str)
        work['stock'] = pd.to_numeric(work.get('stock', 0.0), errors='coerce').fillna(0.0)
        work = work[work['stock'] > 0].copy()
        if work.empty:
            return pd.DataFrame()

        ef_table = self._build_parameter_table(
            'Enteric fermentation',
            'Emission factor',
            year,
            value_col='emission_factor',
        )
        work = work.merge(
            ef_table,
            left_on=['M49_Country_Code', 'Item_Emis'],
            right_on=['M49_Country_Code', 'Item'],
            how='left',
            suffixes=('', '_param')
        )
        work = work.drop(columns=['Item_param'], errors='ignore')
        work['emission_factor'] = pd.to_numeric(work['emission_factor'], errors='coerce').fillna(0.0)

        if scenario_params:
            adjusted = []
            for m49_code, item_emis, ef in zip(work['M49_Country_Code'], work['Item_Emis'], work['emission_factor']):
                ef_val = float(ef)
                ef_bound = self._get_emission_factor_bound(
                    scenario_params,
                    m49_code,
                    item_emis,
                    'Enteric fermentation',
                    'CH4',
                    year
                )
                if ef_bound is not None:
                    ef_val = self._apply_ef_bound(ef_val, ef_bound)
                else:
                    ef_abs = self._get_emission_factor_absolute(
                        scenario_params,
                        m49_code,
                        item_emis,
                        'Enteric fermentation',
                        'CH4',
                        year
                    )
                    if ef_abs is not None:
                        ef_val = ef_abs
                    else:
                        ef_val *= self._get_emission_factor_multiplier(
                            scenario_params,
                            m49_code,
                            item_emis,
                            'Enteric fermentation',
                            year,
                            ghg='CH4'
                        )
                adjusted.append(ef_val)
            work['emission_factor'] = np.asarray(adjusted, dtype=float)

        work['CH4_kt'] = work['stock'] * work['emission_factor'] / 1_000_000.0

        return pd.DataFrame({
            'M49_Country_Code': work['M49_Country_Code'],
            'Item': work['Item_Emis'],
            'year': int(year),
            'process': 'Enteric fermentation',
            'stock': work['stock'],
            'emission_factor': work['emission_factor'],
            'CH4_kt': work['CH4_kt'],
            'N2O_kt': 0.0,
            'CO2_kt': 0.0
        })
    
    def calculate_manure_management(self,
                                   stock_df: pd.DataFrame,
                                   year: int,
                                   scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate manure management emissions.
        
        Args:
            stock_df: DataFrame containing livestock stocks.
            year: Year.
            scenario_params: Scenario parameters.
            
        Returns:
            Emissions DataFrame.
        """
        # Defensively consolidate duplicate stock records to prevent double-counted emissions.
        if stock_df is None:
            return pd.DataFrame()

        # Ensure key columns exist.
        key_cols = ['M49_Country_Code', 'Item', 'Item_Emis', 'year']
        if 'country' in stock_df.columns:
            key_cols.append('country')
        if set(['M49_Country_Code', 'Item']).issubset(stock_df.columns):
            pre_rows = len(stock_df)
            try:
                if 'Item_Emis' in stock_df.columns:
                    group_cols = key_cols
                else:
                    # Group by Item if Item_Emis is absent.
                    group_cols = ['M49_Country_Code', 'Item', 'year']
                    if 'country' in stock_df.columns:
                        group_cols.append('country')

                agg_dict = {}
                # Aggregate stock using 'first', not 'sum', to avoid double counting.
                # Duplicates arise from overlapping milk/meat filters, e.g. Cattle, non-dairy entering both categories.
                if 'stock' in stock_df.columns:
                    agg_dict['stock'] = 'first'  # Use first to avoid duplicate summation.
                # Use first for other numeric columns too, preserving values without accidental summation.
                for c in stock_df.columns:
                    if c not in group_cols and c not in agg_dict and pd.api.types.is_numeric_dtype(stock_df[c]):
                        agg_dict[c] = 'first'

                if agg_dict:
                    stock_df = stock_df.groupby(group_cols, as_index=False).agg(agg_dict)
                else:
                    stock_df = stock_df.drop_duplicates(subset=group_cols)

                post_rows = len(stock_df)
                if pre_rows != post_rows:
                    logger.info(f" [WARN] calculate_manure_management: 检测到{pre_rows - post_rows}行重复stock记录 ({pre_rows} -> {post_rows})")
                    logger.info(f"    这可能是由于商品同时被milk和meat filter捕获（如'Cattle, non-dairy'）")
            except Exception as e:
                logger.info(f"[WARN] calculate_manure_management: 合并stock出错: {e}")

        work = stock_df.copy()
        work['M49_Country_Code'] = work['M49_Country_Code'].apply(_normalize_m49_for_comparison)
        if 'Item_Emis' not in work.columns:
            work['Item_Emis'] = work['Item']
        work['Item_Emis'] = work['Item_Emis'].astype(str)
        work['stock'] = pd.to_numeric(work.get('stock', 0.0), errors='coerce').fillna(0.0)
        work = work[work['stock'] > 0].copy()
        if work.empty:
            return pd.DataFrame()

        n_rate_table = self._build_parameter_table(
            'Manure management',
            'N.excretion.rate',
            year,
            value_col='n_excretion_rate',
        )
        work = work.merge(
            n_rate_table,
            left_on=['M49_Country_Code', 'Item_Emis'],
            right_on=['M49_Country_Code', 'Item'],
            how='left',
            suffixes=('', '_param')
        )
        work = work.drop(columns=['Item_param'], errors='ignore')
        work['n_excretion_rate'] = pd.to_numeric(work['n_excretion_rate'], errors='coerce').fillna(0.0)
        work['annual_n_excretion'] = work['stock'] * work['n_excretion_rate'] * 365.0

        mm_ratio = self._lookup_hist_ratio_series(
            work['M49_Country_Code'],
            work['Item_Emis'],
            year=year,
            element='Manure management ratio',
            default_value=0.5,
            allow_region_global=True,
            count_fallbacks=True,
        )
        mm_ratio = pd.Series(mm_ratio, index=work.index, dtype='float64')

        def _lookup_manure_ratio_scenario_value(raw_map, m49_code, item_emis, country, default):
            if not raw_map:
                return default
            m49_key = _normalize_m49_for_comparison(m49_code)
            item_key = str(item_emis).strip()
            try:
                year_key = int(year)
            except Exception:
                year_key = year
            keys = []
            if m49_key:
                keys.extend([
                    (m49_key, item_key, year_key),
                    (m49_key, 'All', year_key),
                ])
            country_key = str(country or '').strip()
            if country_key:
                keys.extend([
                    (country_key, item_key, year_key),
                    (country_key, 'All', year_key),
                ])
            for key in keys:
                if key in raw_map:
                    try:
                        return float(raw_map.get(key))
                    except Exception:
                        return default
            return default

        if scenario_params and 'manure_management_ratio_multiplier' in scenario_params:
            mm_mult_dict = scenario_params.get('manure_management_ratio_multiplier', {})
            mm_mults = [
                _lookup_manure_ratio_scenario_value(mm_mult_dict, m49_code, item_emis, country, 1.0)
                for m49_code, item_emis, country in zip(
                    work['M49_Country_Code'],
                    work['Item_Emis'],
                    work.get('country', [''] * len(work)),
                )
            ]
            mm_ratio = mm_ratio * np.asarray(mm_mults, dtype=float)

        if scenario_params and 'manure_management_ratio_absolute_by' in scenario_params:
            mm_abs_dict = scenario_params.get('manure_management_ratio_absolute_by', {})
            mm_abs_vals = pd.Series(
                [
                    _lookup_manure_ratio_scenario_value(mm_abs_dict, m49_code, item_emis, country, np.nan)
                    for m49_code, item_emis, country in zip(
                        work['M49_Country_Code'],
                        work['Item_Emis'],
                        work.get('country', [''] * len(work)),
                    )
                ],
                index=work.index,
                dtype='float64',
            )
            mm_ratio = mm_ratio.where(mm_abs_vals.isna(), mm_abs_vals)

        mm_ratio = pd.to_numeric(mm_ratio, errors='coerce').fillna(0.5).clip(lower=0.0, upper=1.0)
        work['mm_ratio'] = mm_ratio
        work['managed_manure_n'] = work['annual_n_excretion'] * work['mm_ratio']

        ef_ch4_table = self._build_parameter_table(
            'Manure management',
            'Emission factor',
            year,
            value_col='ef_ch4',
            species='CH4',
        )
        ef_n2o_table = self._build_parameter_table(
            'Manure management',
            'Emission factor',
            year,
            value_col='ef_n2o',
            species='N2O',
        )
        work = work.merge(
            ef_ch4_table,
            left_on=['M49_Country_Code', 'Item_Emis'],
            right_on=['M49_Country_Code', 'Item'],
            how='left',
            suffixes=('', '_ch4')
        )
        work = work.drop(columns=['Item_ch4'], errors='ignore')
        work = work.merge(
            ef_n2o_table,
            left_on=['M49_Country_Code', 'Item_Emis'],
            right_on=['M49_Country_Code', 'Item'],
            how='left',
            suffixes=('', '_n2o')
        )
        work = work.drop(columns=['Item_n2o'], errors='ignore')
        work['ef_ch4'] = pd.to_numeric(work['ef_ch4'], errors='coerce').fillna(0.0)
        work['ef_n2o'] = pd.to_numeric(work['ef_n2o'], errors='coerce').fillna(0.0)

        if scenario_params:
            adj_ch4: List[float] = []
            adj_n2o: List[float] = []
            for m49_code, item_emis, ef_ch4, ef_n2o in zip(
                work['M49_Country_Code'], work['Item_Emis'], work['ef_ch4'], work['ef_n2o']
            ):
                ef_ch4_val = float(ef_ch4)
                ef_n2o_val = float(ef_n2o)

                ef_bound_ch4 = self._get_emission_factor_bound(
                    scenario_params, m49_code, item_emis, 'Manure management', 'CH4', year
                )
                ef_bound_n2o = self._get_emission_factor_bound(
                    scenario_params, m49_code, item_emis, 'Manure management', 'N2O', year
                )
                ef_abs_ch4 = self._apply_ef_bound(ef_ch4_val, ef_bound_ch4) if ef_bound_ch4 is not None else None
                ef_abs_n2o = self._apply_ef_bound(ef_n2o_val, ef_bound_n2o) if ef_bound_n2o is not None else None
                if ef_abs_ch4 is None:
                    ef_abs_ch4 = self._get_emission_factor_absolute(
                        scenario_params, m49_code, item_emis, 'Manure management', 'CH4', year
                    )
                if ef_abs_n2o is None:
                    ef_abs_n2o = self._get_emission_factor_absolute(
                        scenario_params, m49_code, item_emis, 'Manure management', 'N2O', year
                    )
                if ef_abs_ch4 is not None:
                    ef_ch4_val = float(ef_abs_ch4)
                else:
                    ef_ch4_val *= self._get_emission_factor_multiplier(
                        scenario_params, m49_code, item_emis, 'Manure management', year, ghg='CH4'
                    )
                if ef_abs_n2o is not None:
                    ef_n2o_val = float(ef_abs_n2o)
                else:
                    ef_n2o_val *= self._get_emission_factor_multiplier(
                        scenario_params, m49_code, item_emis, 'Manure management', year, ghg='N2O'
                    )
                adj_ch4.append(ef_ch4_val)
                adj_n2o.append(ef_n2o_val)
            work['ef_ch4'] = np.asarray(adj_ch4, dtype=float)
            work['ef_n2o'] = np.asarray(adj_n2o, dtype=float)

        work['CH4_kt'] = work['managed_manure_n'] * work['ef_ch4'] / 1_000_000.0
        work['N2O_kt'] = work['managed_manure_n'] * work['ef_n2o'] / 1_000_000.0

        return pd.DataFrame({
            'M49_Country_Code': work['M49_Country_Code'],
            'Item': work['Item_Emis'],
            'year': int(year),
            'process': 'Manure management',
            'stock': work['stock'],
            'annual_n_excretion': work['annual_n_excretion'],
            'mm_ratio': work['mm_ratio'],
            'managed_manure_n': work['managed_manure_n'],
            'CH4_kt': work['CH4_kt'],
            'N2O_kt': work['N2O_kt'],
            'CO2_kt': 0.0
        })
    
    def get_manure_management_ratio(self, m49_code: int, item_emis: str, year: int) -> float:
        """
        Get the manure management ratio.
        
         Environment_LivestockManure_with_ratio.csv uses Item_Emis model names.
         FIX: fall back to the base Item name if a dairy/non-dairy-suffixed Item is not found.
        """
        # Define dairy/non-dairy to base Item mappings.
        base_item_map = {
            'Cattle, dairy': 'Cattle',
            'Cattle, non-dairy': 'Cattle',
            'Buffaloes, dairy': 'Buffaloes',
            'Buffaloes, non-dairy': 'Buffaloes',
            'Goats, dairy': 'Goats',
            'Goats, non-dairy': 'Goats',
            'Sheep, dairy': 'Sheep',
            'Sheep, non-dairy': 'Sheep',
            'Camels, dairy': 'Camels',
            'Camels, non-dairy': 'Camels',
        }
        m49_norm = _normalize_m49_for_comparison(m49_code)
        
        def _query_ratio(item_name: str, target_year: int) -> float:
            """Internal helper: query the ratio for a specified Item."""
            hist_mm = self.hist_manure_stock[
                (self.hist_manure_stock['M49_Country_Code'] == m49_norm) &
                (self.hist_manure_stock['Item'] == item_name) &
                (self.hist_manure_stock['Element'] == 'Manure management ratio')
            ]
            
            if not hist_mm.empty:
                year_col = f'Y{target_year}'
                if year_col in hist_mm.columns:
                    value = hist_mm[year_col].iloc[0]
                    if pd.notna(value):
                        return float(value)
            return None

        def _query_mean_ratio(item_name: str, target_year: int, region: Optional[str]) -> Optional[float]:
            """Look up regional/global means."""
            year_col = f'Y{target_year}'
            df = self.hist_manure_stock[
                (self.hist_manure_stock['Item'] == item_name) &
                (self.hist_manure_stock['Element'] == 'Manure management ratio')
            ]
            if df.empty or year_col not in df.columns:
                return None
            if region:
                region_m49_codes = [m49 for m49, reg in self.m49_to_region.items() if reg == region]
                if region_m49_codes:
                    df = df[df['M49_Country_Code'].isin(region_m49_codes)]
            vals = df[year_col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                return None
            return float(vals.mean())
        
        # 1. Try the original Item name first.
        target_year = min(year, 2020)  # Use actual years historically and 2020 for future years.
        
        ratio = _query_ratio(item_emis, target_year)
        if ratio is not None:
            return ratio
        
        # 2. If missing and the Item has a dairy/non-dairy suffix, try its base name.
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio = _query_ratio(base_item, target_year)
            if ratio is not None:
                logger.warning(f" 未找到{item_emis}的manure management ratio，使用基础Item '{base_item}' 的值: {ratio:.6f} (M49={m49_code}, year={year})")
                return ratio
        
        # 3. If still missing, try any available 2020 value.
        ratio_2020 = _query_ratio(item_emis, 2020)
        if ratio_2020 is not None:
            return ratio_2020

        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio_2020 = _query_ratio(base_item, 2020)
            if ratio_2020 is not None:
                logger.warning(f" 未找到{item_emis}的2020 ratio，使用基础Item '{base_item}' 的值: {ratio_2020:.6f} (M49={m49_code})")
                return ratio_2020

        # 4. Regional mean (Region_market_full)
        region = self.m49_to_region.get(m49_norm)
        ratio_region = _query_mean_ratio(item_emis, target_year, region)
        if ratio_region is not None:
            self._manure_ratio_fallback_counts['region'] += 1
            logger.warning(f" 未找到{item_emis}的国家ratio，使用区域均值: {ratio_region:.6f} (Region={region}, M49={m49_code}, year={year})")
            return ratio_region
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio_region = _query_mean_ratio(base_item, target_year, region)
            if ratio_region is not None:
                self._manure_ratio_fallback_counts['region'] += 1
                logger.warning(f" 未找到{item_emis}的国家ratio，使用区域基础Item均值: {ratio_region:.6f} (Region={region}, M49={m49_code}, year={year})")
                return ratio_region

        # 5. Global mean for this Item
        ratio_global = _query_mean_ratio(item_emis, target_year, None)
        if ratio_global is not None:
            self._manure_ratio_fallback_counts['global'] += 1
            logger.warning(f" 未找到{item_emis}的区域ratio，使用全球均值: {ratio_global:.6f} (M49={m49_code}, year={year})")
            return ratio_global
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio_global = _query_mean_ratio(base_item, target_year, None)
            if ratio_global is not None:
                self._manure_ratio_fallback_counts['global'] += 1
                logger.warning(f" 未找到{item_emis}的区域ratio，使用全球基础Item均值: {ratio_global:.6f} (M49={m49_code}, year={year})")
                return ratio_global

        # 6. If still missing, default to 0.5.
        self._manure_ratio_fallback_counts['default'] += 1
        logger.warning(f" 完全未找到{item_emis}的manure management ratio，使用默认值0.5 (M49={m49_code}, year={year})")
        return 0.5
    
    def calculate_manure_applied_to_soils(self,
                                         mm_results: pd.DataFrame,
                                         year: int,
                                         scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate emissions from manure applied to soils.
        
        Args:
            mm_results: Manure management calculation results.
            year: Year.
            scenario_params: Scenario parameters.
            
        Returns:
            Emissions DataFrame.
        """
        if mm_results is None or mm_results.empty:
            return pd.DataFrame()

        work = mm_results.copy()
        work['M49_Country_Code'] = work['M49_Country_Code'].apply(_normalize_m49_for_comparison)
        if 'Item_Emis' not in work.columns:
            work['Item_Emis'] = work['Item']
        work['Item_Emis'] = work['Item_Emis'].astype(str)
        work['managed_manure_n'] = pd.to_numeric(work['managed_manure_n'], errors='coerce').fillna(0.0)

        work['ma_ratio'] = self._lookup_hist_ratio_series(
            work['M49_Country_Code'],
            work['Item_Emis'],
            year=year,
            element='Manure applied ratio',
            default_value=0.3,
        )
        work['applied_manure_n'] = work['managed_manure_n'] * work['ma_ratio']

        ef_table = self._build_parameter_table(
            'Manure applied to soils',
            'Emission factor',
            year,
            value_col='ef_n2o',
        )
        work = work.merge(
            ef_table,
            left_on=['M49_Country_Code', 'Item_Emis'],
            right_on=['M49_Country_Code', 'Item'],
            how='left',
            suffixes=('', '_param')
        )
        work = work.drop(columns=['Item_param'], errors='ignore')
        work['ef_n2o'] = pd.to_numeric(work['ef_n2o'], errors='coerce').fillna(0.0)

        if scenario_params:
            adjusted = []
            for m49_code, item_emis, ef_n2o in zip(work['M49_Country_Code'], work['Item_Emis'], work['ef_n2o']):
                ef_n2o_val = float(ef_n2o)
                ef_bound = self._get_emission_factor_bound(
                    scenario_params,
                    m49_code,
                    item_emis,
                    'Manure applied to soils',
                    'N2O',
                    year
                )
                if ef_bound is not None:
                    ef_n2o_val = self._apply_ef_bound(ef_n2o_val, ef_bound)
                else:
                    ef_abs = self._get_emission_factor_absolute(
                        scenario_params,
                        m49_code,
                        item_emis,
                        'Manure applied to soils',
                        'N2O',
                        year
                    )
                    if ef_abs is not None:
                        ef_n2o_val = ef_abs
                    else:
                        ef_n2o_val *= self._get_emission_factor_multiplier(
                            scenario_params,
                            m49_code,
                            item_emis,
                            'Manure applied to soils',
                            year,
                            ghg='N2O'
                        )
                adjusted.append(ef_n2o_val)
            work['ef_n2o'] = np.asarray(adjusted, dtype=float)

        work['N2O_kt'] = work['applied_manure_n'] * work['ef_n2o'] / 1_000_000.0
        return pd.DataFrame({
            'M49_Country_Code': work['M49_Country_Code'],
            'Item': work['Item_Emis'],
            'year': int(year),
            'process': 'Manure applied to soils',
            'applied_manure_n': work['applied_manure_n'],
            'ma_ratio': work['ma_ratio'],
            'CH4_kt': 0.0,
            'N2O_kt': work['N2O_kt'],
            'CO2_kt': 0.0
        })
    
    def get_manure_applied_ratio(self, m49_code: int, item_emis: str, year: int) -> float:
        """
        Get the manure applied ratio.
        
         Environment_LivestockManure_with_ratio.csv uses Item_Emis model names.
         FIX: fall back to the base Item name if a dairy/non-dairy-suffixed Item is not found.
        """
        base_item_map = {
            'Cattle, dairy': 'Cattle', 'Cattle, non-dairy': 'Cattle',
            'Buffaloes, dairy': 'Buffaloes', 'Buffaloes, non-dairy': 'Buffaloes',
            'Goats, dairy': 'Goats', 'Goats, non-dairy': 'Goats',
            'Sheep, dairy': 'Sheep', 'Sheep, non-dairy': 'Sheep',
            'Camels, dairy': 'Camels', 'Camels, non-dairy': 'Camels',
        }
        
        def _query_ratio(item_name: str, target_year: int) -> float:
            hist_ma = self.hist_manure_stock[
                (self.hist_manure_stock['M49_Country_Code'] == m49_code) &
                (self.hist_manure_stock['Item'] == item_name) &
                (self.hist_manure_stock['Element'] == 'Manure applied ratio')
            ]
            if not hist_ma.empty:
                year_col = f'Y{target_year}'
                if year_col in hist_ma.columns:
                    value = hist_ma[year_col].iloc[0]
                    if pd.notna(value):
                        return float(value)
            return None
        
        target_year = min(year, 2020)
        
        # Try the original Item name.
        ratio = _query_ratio(item_emis, target_year)
        if ratio is not None:
            return ratio
        
        # Try the base Item name.
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio = _query_ratio(base_item, target_year)
            if ratio is not None:
                return ratio
        
        # Try 2020.
        ratio_2020 = _query_ratio(item_emis, 2020)
        if ratio_2020 is not None:
            return ratio_2020
        
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio_2020 = _query_ratio(base_item, 2020)
            if ratio_2020 is not None:
                return ratio_2020
        
        return 0.3  # Default value
    
    def calculate_manure_left_on_pasture(self,
                                        mm_results: pd.DataFrame,
                                        year: int,
                                        scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate emissions from manure left on pasture.
        
        Args:
            mm_results: Manure management calculation results.
            year: Year.
            scenario_params: Scenario parameters.
            
        Returns:
            Emissions DataFrame.
        """
        if mm_results is None or mm_results.empty:
            return pd.DataFrame()

        work = mm_results.copy()
        work['M49_Country_Code'] = work['M49_Country_Code'].apply(_normalize_m49_for_comparison)
        if 'Item_Emis' not in work.columns:
            work['Item_Emis'] = work['Item']
        work['Item_Emis'] = work['Item_Emis'].astype(str)
        work['annual_n_excretion'] = pd.to_numeric(work['annual_n_excretion'], errors='coerce').fillna(0.0)
        work['mm_ratio'] = pd.to_numeric(work['mm_ratio'], errors='coerce').fillna(0.0)

        lp_ratio_sum = self._lookup_hist_ratio_series(
            work['M49_Country_Code'],
            work['Item_Emis'],
            year=year,
            element='Manure left_pasture_management ratio sum',
            default_value=0.7,
        )
        work['lp_ratio'] = np.clip(lp_ratio_sum - work['mm_ratio'].to_numpy(dtype=float), 0.0, 1.0)
        work['left_pasture_n'] = work['annual_n_excretion'] * work['lp_ratio']

        ef_table = self._build_parameter_table(
            'Manure left on pasture',
            'Emission factor',
            year,
            value_col='ef_n2o',
        )
        work = work.merge(
            ef_table,
            left_on=['M49_Country_Code', 'Item_Emis'],
            right_on=['M49_Country_Code', 'Item'],
            how='left',
            suffixes=('', '_param')
        )
        work = work.drop(columns=['Item_param'], errors='ignore')
        work['ef_n2o'] = pd.to_numeric(work['ef_n2o'], errors='coerce').fillna(0.0)

        if scenario_params:
            adjusted = []
            for m49_code, item_emis, ef_n2o in zip(work['M49_Country_Code'], work['Item_Emis'], work['ef_n2o']):
                ef_n2o_val = float(ef_n2o)
                ef_bound = self._get_emission_factor_bound(
                    scenario_params,
                    m49_code,
                    item_emis,
                    'Manure left on pasture',
                    'N2O',
                    year
                )
                if ef_bound is not None:
                    ef_n2o_val = self._apply_ef_bound(ef_n2o_val, ef_bound)
                else:
                    ef_abs = self._get_emission_factor_absolute(
                        scenario_params,
                        m49_code,
                        item_emis,
                        'Manure left on pasture',
                        'N2O',
                        year
                    )
                    if ef_abs is not None:
                        ef_n2o_val = ef_abs
                    else:
                        ef_n2o_val *= self._get_emission_factor_multiplier(
                            scenario_params,
                            m49_code,
                            item_emis,
                            'Manure left on pasture',
                            year,
                            ghg='N2O'
                        )
                adjusted.append(ef_n2o_val)
            work['ef_n2o'] = np.asarray(adjusted, dtype=float)

        work['N2O_kt'] = work['left_pasture_n'] * work['ef_n2o'] / 1_000_000.0
        return pd.DataFrame({
            'M49_Country_Code': work['M49_Country_Code'],
            'Item': work['Item_Emis'],
            'year': int(year),
            'process': 'Manure left on pasture',
            'left_pasture_n': work['left_pasture_n'],
            'lp_ratio': work['lp_ratio'],
            'CH4_kt': 0.0,
            'N2O_kt': work['N2O_kt'],
            'CO2_kt': 0.0
        })
    
    def get_left_pasture_ratio_sum(self, m49_code: int, item_emis: str, year: int) -> float:
        """
        Get the sum of left_pasture_management ratios.
        
         Environment_LivestockManure_with_ratio.csv uses Item_Emis model names.
         FIX: fall back to the base Item name if a dairy/non-dairy-suffixed Item is not found.
        """
        base_item_map = {
            'Cattle, dairy': 'Cattle', 'Cattle, non-dairy': 'Cattle',
            'Buffaloes, dairy': 'Buffaloes', 'Buffaloes, non-dairy': 'Buffaloes',
            'Goats, dairy': 'Goats', 'Goats, non-dairy': 'Goats',
            'Sheep, dairy': 'Sheep', 'Sheep, non-dairy': 'Sheep',
            'Camels, dairy': 'Camels', 'Camels, non-dairy': 'Camels',
        }
        
        def _query_ratio(item_name: str, target_year: int) -> float:
            hist_lp = self.hist_manure_stock[
                (self.hist_manure_stock['M49_Country_Code'] == m49_code) &
                (self.hist_manure_stock['Item'] == item_name) &
                (self.hist_manure_stock['Element'] == 'Manure left_pasture_management ratio sum')
            ]
            if not hist_lp.empty:
                year_col = f'Y{target_year}'
                if year_col in hist_lp.columns:
                    value = hist_lp[year_col].iloc[0]
                    if pd.notna(value):
                        return float(value)
            return None
        
        target_year = min(year, 2020)
        
        # Try the original Item name.
        ratio = _query_ratio(item_emis, target_year)
        if ratio is not None:
            return ratio
        
        # Try the base Item name.
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio = _query_ratio(base_item, target_year)
            if ratio is not None:
                return ratio
        
        # Try 2020.
        ratio_2020 = _query_ratio(item_emis, 2020)
        if ratio_2020 is not None:
            return ratio_2020
        
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio_2020 = _query_ratio(base_item, 2020)
            if ratio_2020 is not None:
                return ratio_2020
        
        return 0.7  # Default value
    
    def merge_dairy_nondairy(self, emissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine dairy and non-dairy species categories.
        
        Args:
            emissions_df: Emissions DataFrame.
            
        Returns:
            Combined DataFrame.
        """
        # Return directly if the DataFrame is empty or has no Item column.
        if emissions_df.empty or 'Item' not in emissions_df.columns:
            return emissions_df
        
        # Define species pairs to combine.
        merge_pairs = {
            ('Buffalo, dairy', 'Buffalo, non-dairy'): 'Buffalo',
            ('Camel, dairy', 'Camel, non-dairy'): 'Camels',
            ('Goats, dairy', 'Goats, non-dairy'): 'Goats',
            ('Sheep, dairy', 'Sheep, non-dairy'): 'Sheep'
        }
        
        result_list = []
        
        for (dairy_item, nondairy_item), merged_item in merge_pairs.items():
            # Find dairy and non-dairy data.
            dairy_data = emissions_df[emissions_df['Item'] == dairy_item].copy()
            nondairy_data = emissions_df[emissions_df['Item'] == nondairy_item].copy()
            
            if dairy_data.empty and nondairy_data.empty:
                continue
            
            # Combine data.
            merged_data = pd.concat([dairy_data, nondairy_data], ignore_index=True)
            
            # Aggregate by M49_Country_Code, year, and process.
            agg_dict = {
                'CH4_kt': 'sum',
                'N2O_kt': 'sum',
                'CO2_kt': 'sum'
            }
            
            # Add other numeric columns.
            for col in merged_data.columns:
                if col not in ['M49_Country_Code', 'Item', 'year', 'process', 
                              'CH4_kt', 'N2O_kt', 'CO2_kt'] and \
                   pd.api.types.is_numeric_dtype(merged_data[col]):
                    agg_dict[col] = 'sum'
            
            merged = merged_data.groupby(
                ['M49_Country_Code', 'year', 'process'],
                as_index=False
            ).agg(agg_dict)
            
            merged['Item'] = merged_item
            result_list.append(merged)
        
        # Preserve original species that do not need combining.
        items_to_remove = []
        for (dairy, nondairy), _ in merge_pairs.items():
            items_to_remove.extend([dairy, nondairy])
        
        other_data = emissions_df[~emissions_df['Item'].isin(items_to_remove)].copy()
        result_list.append(other_data)
        
        # Combine all results.
        final_result = pd.concat(result_list, ignore_index=True)
        
        return final_result
    
    def extract_livestock_parameters(self, years: List[int], hist_cutoff_year: int = 2020) -> Dict[str, pd.DataFrame]:
        """
        Extract livestock parameters for production_summary.
        
        Read historical years from Environment_LivestockManure_with_ratio.csv.
        Use baseline values from the last historical year (2020) for future years.
        
        Returns:
            Dictionary of DataFrames containing individual parameters.
        """
        try:
            # Read historical manure stock data.
            manure_df = read_csv_cached(self.hist_manure_stock_path)
            logger.info(f"[DEBUG] 读取历史livestock参数: {len(manure_df)} 行")
            
            # Standardize column names.
            manure_df.columns = [c.strip() for c in manure_df.columns]
            
            # Required parameter columns
            param_dfs = {}
            
            # Stock parameters
            if 'Stocks' in manure_df.columns:
                stock_df = manure_df[['Area', 'Item', 'Year', 'Stocks']].copy()
                stock_df = stock_df.rename(columns={
                    'Area': 'country',
                    'Item': 'commodity',
                    'Year': 'year',
                    'Stocks': 'stock_head'
                })
                param_dfs['stock'] = stock_df
            
            # Create feed requirements below, after GLE results replace stocks.
            # This ensures use of GLE-calculated stock_df with M49 codes, rather than CSV stocks with country names.
            
            # Manure management ratio
            if 'Manure_management_ratio' in manure_df.columns:
                manure_ratio_df = manure_df[['Area', 'Item', 'Year', 'Manure_management_ratio']].copy()
                manure_ratio_df = manure_ratio_df.rename(columns={
                    'Area': 'country',
                    'Item': 'commodity',
                    'Year': 'year',
                    'Manure_management_ratio': 'manure_management_ratio'
                })
                param_dfs['manure_ratio'] = manure_ratio_df
            
            
            # Use GLE-calculated slaughter, stock, and carcass_yield when available.
            # run_full_calculation stores these data during calculation.
            
            if self._computed_slaughter:
                computed_slaughter_df = pd.concat(self._computed_slaughter, ignore_index=True)
                param_dfs['slaughter'] = computed_slaughter_df
                logger.info(f"[DEBUG] 导出GLE计算的slaughter: {len(computed_slaughter_df)} 行")
            
            if self._computed_stock:
                computed_stock_df = pd.concat(self._computed_stock, ignore_index=True)
                # Replace CSV-read stocks with calculated results.
                param_dfs['stock'] = computed_stock_df
                logger.info(f"[DEBUG] 导出GLE计算的stock: {len(computed_stock_df)} 行")
            
            if self._computed_carcass_yield:
                computed_yield_df = pd.concat(self._computed_carcass_yield, ignore_index=True)
                param_dfs['carcass_yield'] = computed_yield_df
                logger.info(f"[DEBUG] 导出GLE计算的carcass_yield (meat类): {len(computed_yield_df)} 行")
            
            # Export dairy yield for milk/eggs.
            if self._computed_dairy_yield:
                computed_dairy_yield_df = pd.concat(self._computed_dairy_yield, ignore_index=True)
                param_dfs['dairy_yield'] = computed_dairy_yield_df
                logger.info(f"[DEBUG] 导出GLE计算的dairy_yield (milk/egg类): {len(computed_dairy_yield_df)} 行")
            
            # Feed requirements from dict_v3 defaults
            # Create after replacing stocks with GLE results to ensure M49 codes are used.
            if hasattr(self, 'dict_v3') and 'stock' in param_dfs:
                emis_item_df = self.dict_v3.get('Emis_item', pd.DataFrame())
                if not emis_item_df.empty and 'Feed_GE' in emis_item_df.columns:
                    # Build the feed requirement lookup.
                    feed_lookup = emis_item_df.set_index('Item_Emis')['Feed_GE'].to_dict()
                    
                    # Use the latest stock_df: GLE results with M49 codes.
                    stock_df_for_feed = param_dfs['stock']
                    feed_req_df = stock_df_for_feed[['country', 'commodity', 'year']].copy()
                    feed_req_df['feed_requirement_kg_per_head'] = feed_req_df['commodity'].map(feed_lookup)
                    param_dfs['feed_requirement'] = feed_req_df
                    logger.info(f"[DEBUG] 导出GLE计算的feed_requirement: {len(feed_req_df)} 行 (基于GLE计算的stock)")
            
            # Extend parameters to future years using the 2020 baseline.
            # No extension is needed if _computed_* already contains future data.
            future_years = [y for y in years if y > hist_cutoff_year]
            if future_years:
                logger.info(f"[DEBUG] 检查是否需要扩展参数: {future_years}")
                for param_name, param_df in list(param_dfs.items()):
                    if param_df.empty:
                        continue
                    
                    # Check whether future-year data already exist.
                    existing_future = param_df[param_df['year'].isin(future_years)]
                    if not existing_future.empty:
                        logger.info(f"  [{param_name}] 已有 {len(existing_future)} 行未来年份数据，跳过扩展")
                        continue
                    
                    # Extract 2020 data as the baseline.
                    baseline_2020 = param_df[param_df['year'] == hist_cutoff_year].copy()
                    if baseline_2020.empty:
                        continue
                    
                    # Copy for each future year.
                    extended_rows = []
                    for future_year in future_years:
                        future_data = baseline_2020.copy()
                        future_data['year'] = future_year
                        extended_rows.append(future_data)
                    
                    if extended_rows:
                        param_dfs[param_name] = pd.concat([param_df] + extended_rows, ignore_index=True)
                        logger.info(f"  [{param_name}] 扩展后: {len(param_dfs[param_name])} 行")
            
            return param_dfs
            
        except Exception as e:
            logger.info(f"[ERROR] 提取livestock参数失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_full_calculation(self,
                            production_df: pd.DataFrame,
                            years: List[int],
                            scenario_params: Optional[Dict] = None,
                            hist_cutoff_year: int = 2020) -> Dict[str, pd.DataFrame]:
        """
        Run the full emissions calculation workflow.
        
        Args:
            production_df: DataFrame containing projected future production.
            years: List of calculation years.
            scenario_params: Scenario parameters.
            hist_cutoff_year: Read emissions directly from historical data for years up to and including this cutoff.
            
        Returns:
            Dictionary containing results for all emissions processes.
        """
        # WARNING: do not filter Region here - production_df should already be filtered
        # Region filtering belongs in S4_0_main.py when constructing production_for_livestock.
        self.reset_runtime_outputs()
        
        all_results = {
            'Enteric fermentation': [],
            'Manure management': [],
            'Manure applied to soils': [],
            'Manure left on pasture': []
        }
        
        # Separate historical and future years.
        hist_years = [y for y in years if y <= hist_cutoff_year]
        future_years = [y for y in years if y > hist_cutoff_year]
        
        # 1. Historical years: read directly from the emissions CSV.
        if hist_years:
            logger.info(f"\n{'='*60}")
            logger.info(f"Loading historical emissions for years {min(hist_years)}-{max(hist_years)}...")
            logger.info(f"{'='*60}")
            hist_emissions = self._load_historical_emissions(hist_years)
            if hist_emissions:
                for process, emis_df in hist_emissions.items():
                    if emis_df is not None and not emis_df.empty:
                        all_results[process].append(emis_df)
                        logger.info(f"  [OK] {process}: {len(emis_df)} 行历史数据")
        
        # 2. Future years: calculate emissions.
        logger.info(f"\n[DEBUG] future_years = {future_years}")
        logger.info(f"[DEBUG] production_df shape: {production_df.shape}")
        if not production_df.empty:
            logger.info(f"[DEBUG] production_df years: {production_df['year'].unique() if 'year' in production_df.columns else 'NO year column'}")
            logger.info(f"[DEBUG] production_df columns: {list(production_df.columns)}")
            
            # Print the 2080 data overview.
            if 'year' in production_df.columns:
                year_2080_data = production_df[production_df['year'] == 2080]
                logger.info(f"[DEBUG] 2080年数据行数: {len(year_2080_data)}")
                if not year_2080_data.empty:
                    logger.info(f"[DEBUG] 2080年前5行:")
                    logger.info(year_2080_data.head())
                    if 'Commodity' in year_2080_data.columns:
                        logger.info(f"[DEBUG] 2080年商品: {year_2080_data['Commodity'].unique()}")

            # Do not normalize production_df M49 codes
            # production_df should already have correct format from S4_0_main
            # Normalizing here would break matching with dict_v3 (which has '004, '008, etc.)
        
        for year in future_years:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing year {year}...")
            logger.info(f"{'='*60}")
            
            # WARNING: debug: inspect the production statistics
            if 'year' in production_df.columns:
                year_prod = production_df[production_df['year'] == year]
                logger.info(f"[DEBUG] Year {year} 产量数据: {len(year_prod)} 行")
                if not year_prod.empty:
                    logger.info(f"  - 商品数: {year_prod['Commodity'].nunique() if 'Commodity' in year_prod.columns else 'N/A'}")
                    logger.info(f"  - 国家数: {year_prod['M49_Country_Code'].nunique() if 'M49_Country_Code' in year_prod.columns else 'N/A'}")
                    if 'production_t' in year_prod.columns:
                        total_prod = year_prod['production_t'].sum()
                        logger.info(f"  - 总产量: {total_prod:.0f} tons")
                else:
                    logger.info(f"  WARNING: country set has no production data")
            
            # 1. Calculate dairy animals, including milk producers and laying hens.
            # dict_v3 classifies Chickens, layers as Dairy.
            logger.info(f"\n=== 调用calculate_milk_animals (year={year}) ===")
            dairy_animals_df = self.calculate_milk_animals(production_df, year)
            logger.info(f"[返回结果] dairy_animals_df: {len(dairy_animals_df) if not dairy_animals_df.empty else 0} 行")
            if not dairy_animals_df.empty:
                logger.info(f"  - 包含列: {list(dairy_animals_df.columns)}")
                logger.info(f"  - 前3行数据:")
                logger.info(dairy_animals_df.head(3))
            
            # 2. Calculate meat animals, excluding dairy animals.
            logger.info(f"\n=== 调用calculate_meat_animals (year={year}) ===")
            meat_animals_df = self.calculate_meat_animals(production_df, year)
            logger.info(f"[返回结果] meat_animals_df: {len(meat_animals_df) if not meat_animals_df.empty else 0} 行")
            if not meat_animals_df.empty:
                logger.info(f"  - 包含列: {list(meat_animals_df.columns)}")
                logger.info(f"  - 前3行数据:")
                logger.info(meat_animals_df.head(3))
            
            # Save calculated dairy yield extracted from dairy_animals_df.
            if not dairy_animals_df.empty and 'yield_t_per_head' in dairy_animals_df.columns:
                dairy_yield_save = dairy_animals_df[['M49_Country_Code', 'Item_Emis', 'year', 'yield_t_per_head']].copy()
                dairy_yield_save = dairy_yield_save.rename(columns={
                    'Item_Emis': 'commodity'
                })
                # Retain M49_Country_Code and also create country.
                dairy_yield_save['country'] = dairy_yield_save['M49_Country_Code']
                self._computed_dairy_yield.append(dairy_yield_save)
                logger.info(f"  [保存] dairy_yield: {len(dairy_yield_save)} 行")
            
            # 3. Combine all animal flows.
            all_animals_list = []
            if not dairy_animals_df.empty:
                all_animals_list.append(dairy_animals_df)
            if not meat_animals_df.empty:
                all_animals_list.append(meat_animals_df)
            
            if not all_animals_list:
                logger.info(f"WARNING: Year {year}: no production data for this country; skipping")
                continue
            
            all_animals_df = pd.concat(all_animals_list, ignore_index=True)
            logger.info(f"\n总计 {len(all_animals_df)} 个动物流量记录")
            
            # 5. Calculate stocks.
            stock_df = self.calculate_stock_from_animals(all_animals_df, year)
            
            if stock_df.empty:
                logger.info(f"WARNING: Year {year}: stock_df is empty; skipping emissions")
                continue
            
            # Save calculated stocks.
            # Preserve M49_Country_Code and provide country for compatibility.
            if 'stock' in stock_df.columns:
                stock_save = stock_df[['M49_Country_Code', 'Item_Emis', 'year', 'stock']].copy()
                # Preserve M49_Country_Code and copy it to country as the country identifier.
                stock_save = stock_save.rename(columns={
                    'Item_Emis': 'commodity',
                    'stock': 'stock_head'
                })
                # Also provide country, identical to M49, for compatibility.
                stock_save['country'] = stock_save['M49_Country_Code']
                self._computed_stock.append(stock_save)
            
            # Save slaughter data extracted from meat_animals_df.
            # Preserve M49_Country_Code.
            if not meat_animals_df.empty and 'slaughtered' in meat_animals_df.columns:
                slaughter_save = meat_animals_df[['M49_Country_Code', 'Item_Emis', 'year', 'slaughtered']].copy()
                slaughter_save = slaughter_save.rename(columns={
                    'Item_Emis': 'commodity',
                    'slaughtered': 'slaughter_head'
                })
                # Retain M49_Country_Code and also create country.
                slaughter_save['country'] = slaughter_save['M49_Country_Code']
                
                # Add carcass_yield from the carcass_weight_t column.
                if 'carcass_weight_t' in meat_animals_df.columns:
                    yield_save = meat_animals_df[['M49_Country_Code', 'Item_Emis', 'year', 'carcass_weight_t']].copy()
                    yield_save = yield_save.rename(columns={
                        'Item_Emis': 'commodity',
                        'carcass_weight_t': 'yield_t_per_head'
                    })
                    # Retain M49_Country_Code and also create country.
                    yield_save['country'] = yield_save['M49_Country_Code']
                    self._computed_carcass_yield.append(yield_save)
                self._computed_slaughter.append(slaughter_save)
            
            logger.info(f"计算得到 {len(stock_df)} 个存栏记录")
            
            # 6. Calculate enteric fermentation.
            ef_results = self.calculate_enteric_fermentation(stock_df, year, scenario_params)
            if not ef_results.empty:
                all_results['Enteric fermentation'].append(ef_results)
                logger.info(f"  [OK] Enteric fermentation: {len(ef_results)} 行")
            
            # 7. Calculate manure management.
            mm_results = self.calculate_manure_management(stock_df, year, scenario_params)
            # Defensively remove exact mm_results duplicates by country/Item/year/three gas values.
            try:
                if mm_results is not None and not mm_results.empty:
                    before_mm = len(mm_results)
                    dup_cols = ['M49_Country_Code', 'Item', 'year', 'CH4_kt', 'N2O_kt', 'CO2_kt']
                    if set(dup_cols).issubset(mm_results.columns):
                        mm_results = mm_results.drop_duplicates(subset=dup_cols, keep='first').reset_index(drop=True)
                        after_mm = len(mm_results)
                        if before_mm != after_mm:
                            logger.info(f"[INFO] run_full_calculation: 去除 mm_results 完全重复行 {before_mm} -> {after_mm}")
            except Exception as e:
                logger.info(f"[WARN] run_full_calculation: 去重 mm_results 失败: {e}")
            # outputs disabled
            if False:
                try:
                    import datetime
                    dbg_dir = Path(get_results_base()) / 'debug_outputs' / 'GLE'
                    dbg_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

                    if not mm_results.empty:
                        # write full mm_results snapshot for this year
                        try:
                            mm_results.to_csv(dbg_dir / f'mm_results_snapshot_{year}_{ts}.csv', index=False)
                        except Exception:
                            pass

                        # detect exact duplicate rows in mm_results
                        try:
                            dup_cols = ['M49_Country_Code', 'Item', 'year', 'CH4_kt', 'N2O_kt', 'CO2_kt']
                            if set(dup_cols).issubset(mm_results.columns):
                                dup_mask_all = mm_results.duplicated(subset=dup_cols, keep=False)
                                if dup_mask_all.any():
                                    try:
                                        mm_results[dup_mask_all].to_csv(dbg_dir / f'mm_results_duplicates_{year}_{ts}.csv', index=False)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        # write a small summary file for quick review
                        try:
                            with open(dbg_dir / f'mm_debug_summary_{year}_{ts}.txt', 'w', encoding='utf-8') as fh:
                                fh.write(f"year={year}\n")
                                fh.write(f"mm_results rows total: {len(mm_results)}\n")
                                try:
                                    # count duplicates (exact)
                                    dup_count = int(mm_results.duplicated(subset=dup_cols, keep=False).sum()) if set(dup_cols).issubset(mm_results.columns) else 0
                                except Exception:
                                    dup_count = -1
                                fh.write(f"duplicates_exact_count: {dup_count}\n")
                        except Exception:
                            pass
                except Exception:
                    # Do not interrupt normal flow if diagnostics fail
                    pass
            if not mm_results.empty:
                all_results['Manure management'].append(mm_results)
                logger.info(f"  [OK] Manure management: {len(mm_results)} 行")
            
            # 8. Calculate manure applied to soils.
            mas_results = self.calculate_manure_applied_to_soils(mm_results, year, scenario_params)
            if not mas_results.empty:
                all_results['Manure applied to soils'].append(mas_results)
                logger.info(f"  [OK] Manure applied to soils: {len(mas_results)} 行")
            
            # 9. Calculate manure left on pasture.
            mlp_results = self.calculate_manure_left_on_pasture(mm_results, year, scenario_params)
            if not mlp_results.empty:
                all_results['Manure left on pasture'].append(mlp_results)
                logger.info(f"  [OK] Manure left on pasture: {len(mlp_results)} 行")
        
        # Combine all years.
        final_results = {}
        for process, results_list in all_results.items():
            if results_list:
                combined = pd.concat(results_list, ignore_index=True)
                # Do not combine dairy and non-dairy; dict_v3 requires strict separation.
                # S4_1_results already separates historical data; future data are calculated separately.
                # combined = self.merge_dairy_nondairy(combined) # Disabled because it conflicts with the requirement.
                final_results[process] = combined
                
                # WARNING: debug: look at the per-process data distribution
                if 'year' in combined.columns:
                    year_counts = combined['year'].value_counts().sort_index()
                    logger.info(f"\n[DEBUG] {process} 年份分布:")
                    for yr, cnt in year_counts.items():
                        logger.info(f"  - {yr}: {cnt} 行")
        
        return final_results


def _get_cached_livestock_calculator(
    *,
    gle_params_path: str,
    hist_production_path: str,
    hist_emissions_path: str,
    hist_manure_stock_path: str,
    dict_v3_path: str,
    scenario_params: Optional[Dict[str, Any]] = None,
) -> LivestockEmissionsCalculator:
    yield_mult = scenario_params.get('yield_multiplier') if isinstance(scenario_params, dict) else None
    key = (
        str(Path(gle_params_path).resolve()),
        str(Path(hist_production_path).resolve()),
        str(Path(hist_emissions_path).resolve()),
        str(Path(hist_manure_stock_path).resolve()),
        str(Path(dict_v3_path).resolve()),
        tuple(sorted((repr(k),float(v)) for k,v in (yield_mult or {}).items())),
        tuple(sorted((repr(k),float(v)) for k,v in (scenario_params or {}).get('stock_ratio_reference', {}).items())),
        tuple(sorted((repr(k),float(v)) for k,v in (scenario_params or {}).get('animal_yield_reference', {}).items())),
        tuple(sorted((repr(k),float(v)) for k,v in (scenario_params or {}).get('background_stock_reference', {}).items())),
    )
    calculator = _CALCULATOR_CACHE.get(key)
    if calculator is None:
        calculator = LivestockEmissionsCalculator(
            gle_params_path=gle_params_path,
            hist_production_path=hist_production_path,
            hist_emissions_path=hist_emissions_path,
            hist_manure_stock_path=hist_manure_stock_path,
            dict_v3_path=dict_v3_path,
            scenario_params=scenario_params,
        )
        _CALCULATOR_CACHE[key] = calculator
    return calculator


def calculate_stock_from_optimized_production(
    production_df: pd.DataFrame,
    years: List[int],
    gle_params_path: str,
    hist_production_path: str,
    hist_emissions_path: str,
    hist_manure_stock_path: str,
    dict_v3_path: str,
    universe: Any,
    scenario_params: Optional[Dict[str, Any]] = None,
    hist_cutoff_year: int = 2020
) -> pd.DataFrame:
    """
    Calculate livestock stocks from optimized production.
    
    Complete calculation chain:
        Qs (optimized production) 
        -> calculate_milk_animals / calculate_meat_animals (production -> animal flows)
        -> calculate_stock_from_animals (animal flows -> stocks)
        -> stock_df, used by build_feed_demand_from_stock.
    
    Args:
        production_df: Optimized production with [country, year, commodity, production_t] and M49_Country_Code.
        years: List of calculation years.
        gle_params_path: Path to GLE parameters.
        hist_production_path: Historical production data path.
        hist_emissions_path: Historical emissions data path.
        hist_manure_stock_path: Historical manure/stock data path.
        dict_v3_path: Path to dict_v3.
        universe: Universe object containing country mappings.
        scenario_params: Scenario parameters for yield adjustments, etc.
        hist_cutoff_year: Last historical year.
        
    Returns:
        stock_df: Stock DataFrame with [country, iso3, year, commodity, stock_head],
                  compatible with build_feed_demand_from_stock inputs.
    """
    logger.info("=" * 60)
    logger.info("开始从优化后产量计算存栏 (calculate_stock_from_optimized_production)")
    logger.info("=" * 60)
    
    # Validate input parameters.
    logger.info("\n [GLE诊断] 输入参数检查:")
    logger.info(f"  - production_df是否为空: {production_df is None or production_df.empty}")
    if production_df is not None and not production_df.empty:
        logger.info(f"  - production_df形状: {production_df.shape}")
        logger.info(f"  - production_df列名: {list(production_df.columns)}")
        logger.info(f"  - production_df年份: {sorted(production_df['year'].unique()) if 'year' in production_df.columns else 'N/A'}")
        logger.info(f"  - production_df国家数: {production_df['country'].nunique() if 'country' in production_df.columns else 'N/A'}")
        logger.info(f"  - production_df商品数: {production_df['commodity'].nunique() if 'commodity' in production_df.columns else 'N/A'}")
        if 'Commodity' in production_df.columns:
            logger.info(f"  - Commodity列存在: ")
            logger.info(f"  - Commodity样例: {production_df['Commodity'].head(5).tolist()}")
        else:
            logger.info(f"  - Commodity列存在: ")
        if 'production_t' in production_df.columns:
            logger.info(f"  - production_t总和: {production_df['production_t'].sum():,.0f} t")
            logger.info(f"  - production_t非零行数: {(production_df['production_t'] > 0).sum()}/{len(production_df)}")
    logger.info(f"  - years参数: {years}")
    logger.info(f"  - hist_cutoff_year: {hist_cutoff_year}\n")
    
    if production_df is None or production_df.empty:
        logger.warning(" production_df 为空，返回空存栏数据")
        return pd.DataFrame(columns=['country', 'iso3', 'year', 'commodity', 'stock_head'])
    
    calculator = _get_cached_livestock_calculator(
        gle_params_path=gle_params_path,
        hist_production_path=hist_production_path,
        hist_emissions_path=hist_emissions_path,
        hist_manure_stock_path=hist_manure_stock_path,
        dict_v3_path=dict_v3_path,
        scenario_params=scenario_params,
    )
    calculator.reset_runtime_outputs()
    
    # Use the optimized production chain for years >= hist_cutoff_year to preserve base-period consistency.
    dynamic_years = sorted({int(y) for y in years if int(y) >= hist_cutoff_year})
    hist_years = [y for y in years if y < hist_cutoff_year]
    
    all_stock_results = []
    
    # 1. Future years: calculate stocks from optimized production.
    logger.info(f"处理年份 {dynamic_years}，从优化产量计算存栏...")
    
    for year in dynamic_years:
        logger.info(f"\n{'='*60}")
        logger.info(f" 处理年份 {year}...")
        logger.info(f"{'='*60}")
        
        # 1a. Calculate dairy animals, including dairy cattle and laying hens.
        dairy_animals_df = calculator.calculate_milk_animals(production_df, year)
        logger.info(f"  Dairy动物计算结果: {len(dairy_animals_df)} 行")
        
        # 1b. Calculate meat animals, including beef cattle and pigs.
        meat_animals_df = calculator.calculate_meat_animals(production_df, year)
        logger.info(f"  Meat动物计算结果: {len(meat_animals_df)} 行")
        
        # 1c. Combine all animal flows.
        all_animals_list = []
        if not dairy_animals_df.empty:
            all_animals_list.append(dairy_animals_df)
        if not meat_animals_df.empty:
            all_animals_list.append(meat_animals_df)
        
        if not all_animals_list:
            logger.warning(f"年份 {year} 没有动物流量数据，跳过")
            continue
        
        animals_df = pd.concat(all_animals_list, ignore_index=True)
        
        # 1d. Calculate stocks from animal flows.
        stock_df_year = calculator.calculate_stock_from_animals(animals_df, year)
        
        if not stock_df_year.empty:
            all_stock_results.append(stock_df_year)
            logger.info(f"  年份 {year}: 计算得到 {len(stock_df_year)} 条存栏记录")
    
    if not all_stock_results:
        logger.warning("没有计算得到任何存栏数据")
        return pd.DataFrame(columns=['country', 'iso3', 'year', 'commodity', 'stock_head'])
    
    # 2. Combine stocks for all years.
    combined_stock = pd.concat(all_stock_results, ignore_index=True)
    
    # 3. Convert to the input format required by build_feed_demand_from_stock.
    # Original: ['M49_Country_Code', 'Item', 'Item_Emis', 'year', 'stock', 'animal_type'].
    # Target: ['country', 'iso3', 'year', 'commodity', 'stock_head'].
    
    # Build the M49-to-country mapping.
    # M49???iso3
    def _norm_m49(val):
        if val is None or pd.isna(val):
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

    result_df = combined_stock.copy()
    result_df['M49_Country_Code'] = result_df['M49_Country_Code'].apply(_norm_m49)
    result_df['country'] = result_df['M49_Country_Code']
    if universe and hasattr(universe, 'country_by_m49') and universe.country_by_m49:
        result_df['country_name'] = result_df['country'].map(universe.country_by_m49)
    if universe and hasattr(universe, 'iso3_by_country') and universe.iso3_by_country:
        # Prefer standardized M49-to-ISO3 mapping; fall back to country_name.
        result_df['iso3'] = result_df['M49_Country_Code'].map(universe.iso3_by_country)
        if result_df['iso3'].isna().any() and 'country_name' in result_df.columns:
            result_df['iso3'] = result_df['iso3'].fillna(result_df['country_name'].map(universe.iso3_by_country))

    
    # Keep commodity in Item_Emis format.
    # build_feed_demand_from_stock expects Item_Emis names such as 'Cattle, dairy';
    # internally, comm_to_species maps them to species such as 'dairy_cattle'.
    # No conversion is needed here; use Item_Emis directly as commodity.
    result_df['commodity'] = result_df['Item_Emis']
    result_df['stock_head'] = result_df['stock']
    
    # Confirm the commodity format.
    unique_commodity = result_df['commodity'].unique()
    logger.info(f" 存栏数据commodity格式（Item_Emis）: {list(unique_commodity)[:10]}")
    
    # Trace commodity data using the USA as an example.
    us_converted = result_df[result_df['country'] == 'United States of America']
    if not us_converted.empty:
        logger.info("\n" + "=" * 80)
        logger.info(" [美国数据流] Step 3: 存栏数据格式确认")
        logger.info("=" * 80)
        us_sample = us_converted[['commodity', 'stock_head', 'year']].head(10)
        for _, row in us_sample.iterrows():
            logger.info(f"  {row['commodity']:25s} | {row['stock_head']:>12,.0f} head ({row['year']}年)")
    
    # Select valid data.
    result_df = result_df.dropna(subset=['country'])
    result_df = result_df[result_df['stock_head'] > 0]
    
    # Select required columns.
    output_cols = ['country', 'iso3', 'year', 'commodity', 'stock_head']
    result_df = result_df[output_cols].copy()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"存栏计算完成: 共 {len(result_df)} 条记录")
    logger.info(f"  - 年份范围: {result_df['year'].min()}-{result_df['year'].max()}")
    logger.info(f"  - 国家数: {result_df['country'].nunique()}")
    logger.info(f"  - 商品数: {result_df['commodity'].nunique()}")
    logger.info(f"  - 总存栏: {result_df['stock_head'].sum():,.0f} head")
    logger.info(f"{'='*60}")
    
    return result_df


def run_livestock_emissions(
    production_df: pd.DataFrame,
    years: List[int],
    gle_params_path: str,
    hist_production_path: str,
    hist_emissions_path: str,
    hist_manure_stock_path: str,
    dict_v3_path: str,
    scenario_params: Optional[Dict] = None,
    hist_cutoff_year: int = 2020
) -> Dict[str, Any]:
    """
    Main function for livestock emissions calculations.
    
    Args:
        production_df: Projected production DataFrame.
        years: List of calculation years.
        gle_params_path: Path to GLE parameters.
        hist_production_path: Historical production data path.
        hist_emissions_path: Historical emissions file, read directly for historical years.
        hist_manure_stock_path: Historical manure/stock data path.
        dict_v3_path: Path to dict_v3.
        scenario_params: Scenario parameters.
        hist_cutoff_year: Read CSV values through this year; calculate emissions only afterward.
        
    Returns:
        Dictionary containing emissions and parameters:
        {
            'emissions': {process_name: DataFrame, ...},  # Emissions results
            'parameters': {  # Livestock parameters
                'stock': DataFrame,
                'feed_requirement': DataFrame,
                'slaughter': DataFrame,
                'carcass_yield': DataFrame,
                'manure_ratio': DataFrame
            }
        }
    """
    calculator = _get_cached_livestock_calculator(
        gle_params_path=gle_params_path,
        hist_production_path=hist_production_path,
        hist_emissions_path=hist_emissions_path,
        hist_manure_stock_path=hist_manure_stock_path,
        dict_v3_path=dict_v3_path,
        scenario_params=scenario_params,
    )
    calculator.reset_runtime_outputs()
    
    emissions = calculator.run_full_calculation(
        production_df=production_df,
        years=years,
        scenario_params=scenario_params,
        hist_cutoff_year=hist_cutoff_year
    )

    counts = calculator._manure_ratio_fallback_counts
    logger.info(
        "[MANURE_RATIO_FALLBACK] region=%d global=%d default_0.5=%d",
        counts.get('region', 0),
        counts.get('global', 0),
        counts.get('default', 0),
    )
    
    # Extract livestock parameters from historical files and calculated results.
    parameters = calculator.extract_livestock_parameters(years, hist_cutoff_year)
    
    return {
        'emissions': emissions,
        'parameters': parameters
    }
