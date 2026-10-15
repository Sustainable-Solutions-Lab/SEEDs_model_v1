# -*- coding: utf-8 -*-
"""
S3.2 Feed demand builder
-----------------------
Derives livestock feed requirements (grass + crop) directly from
country-level livestock stocks and parameter tables stored under
input/Land/Feed_pasture/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
import logging
import numpy as np
import pandas as pd

from S1_0_schema import Universe
from S2_0_load_data import DataPaths, EmisItemMappings
from runtime_data_cache import read_excel_cached

# Configure the logger.
logger = logging.getLogger(__name__)


def _get_debug_level() -> int:
    raw = os.environ.get("NZF_DEBUG_LEVEL", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(float(raw)))
    except Exception:
        return 0


def _log_info(message: str, *, level: int = 1) -> None:
    if _get_debug_level() >= level:
        logger.info(message)


@dataclass
class FeedDemandOutputs:
    crop_feed_demand: pd.DataFrame
    grass_requirement: pd.DataFrame
    species_dm_detail: pd.DataFrame
    crop_feed_by_livestock: pd.DataFrame = field(default_factory=pd.DataFrame)
    grass_by_livestock: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_feed_demand_from_stock(*,
                                 stock_df: pd.DataFrame,
                                 universe: Universe,
                                 maps: EmisItemMappings,
                                 paths: DataPaths,
                                 years: List[int],
                                 conversion_multiplier: Optional[Dict[Tuple[str, str, int], float]] = None,
                                 feed_requirement_scheme: Optional[str] = None) -> FeedDemandOutputs:
    """
    Convert stock_head (by commodity/country/year) into:
      1) species-level DM requirements from Feed_need_per_head...xlsx
      2) grass vs crop DM split via Grass_feed_ratio...
      3) crop-specific feed demand by commodity (converted to grain using dm_conversion_coefficients)
      4) grass DM requirement + implied pasture area using Pasture_DM_yield_by_country.xlsx
    """
    # All DataFrames include M49_Country_Code.
    empty = FeedDemandOutputs(
        crop_feed_demand=pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','feed_t']),
        grass_requirement=pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','grass_tdm','grass_area_need_ha']),
        species_dm_detail=pd.DataFrame(columns=[
            'country','iso3','m49_code','year','commodity','species',
            'stock_head','kg_dm_per_head','feed_efficiency_multiplier',
            'dm_total_kg','grass_dm_kg','crop_dm_kg'
        ])
    )
    if stock_df is None or stock_df.empty:
        return empty
    conv_mult: Dict[Tuple[str, str, int], float] = {}
    for key, val in (conversion_multiplier or {}).items():
        try:
            country, commodity, year = key
            mult = float(val)
            if not np.isfinite(mult) or mult < 0.0:
                continue
            country_text = str(country).strip()
            commodity_text = str(commodity).strip()
            aliases = {country_text}
            m49_alias = _parse_m49(country_text)
            if m49_alias:
                aliases.add(m49_alias)
            for country_alias in aliases:
                conv_mult[(country_alias, commodity_text, int(year))] = mult
        except Exception:
            continue

    if not (paths.feed_need_xlsx and os.path.exists(paths.feed_need_xlsx)):
        logger.error(f"[S3_2 ERROR] Feed_need文件不存在或路径为None: {paths.feed_need_xlsx}")
        return empty
    if not (paths.grass_ratio_xlsx and os.path.exists(paths.grass_ratio_xlsx)):
        logger.error(f"[S3_2 ERROR]  Grass_ratio文件不存在或路径为None: {paths.grass_ratio_xlsx}")
        logger.error(f"[S3_2 ERROR] 这是导致草地需求缺失的根本原因！")
        return empty
    if not (paths.pasture_dm_yield_xlsx and os.path.exists(paths.pasture_dm_yield_xlsx)):
        logger.error(f"[S3_2 ERROR] Pasture_yield文件不存在或路径为None: {paths.pasture_dm_yield_xlsx}")
        return empty

    years = sorted(set(int(y) for y in years))
    _log_info(f"[S3_2 DEBUG] 请求的years: {years}")
    
    comm_to_species = {}
    for feed_item, comm in (maps.feed_item_to_comm or {}).items():
        if comm in comm_to_species and comm_to_species[comm] != feed_item:
            raise ValueError(f"Multiple feed systems for {comm}; an explicit allocation is required")
        comm_to_species[comm] = feed_item
    _log_info(f"[S3_2 DEBUG] comm_to_species映射: {len(comm_to_species)} 个commodity")
    if len(comm_to_species) > 0:
        _log_info(f"[S3_2 DEBUG] 映射样例: {list(comm_to_species.items())[:5]}", level=2)
    
    # comm_to_species maps Item_Emis keys to Item_Feed_Map species values.
    # Example: {'Cattle, dairy': 'dairy_cattle', 'Cattle, non-dairy': 'beef_cattle'}.
    # Incoming stock_df.commodity should already use Item_Feed_Map species names.
    # The mapping direction therefore needs checking and adjustment.
    
    if not comm_to_species:
        logger.error(f"[S3_2 ERROR]  comm_to_species映射为空！")
        return empty

    stock = stock_df.copy()
    _log_info(f"[S3_2 DEBUG] 输入存栏数据: {len(stock)} 行")
    if 'year' in stock.columns:
        stock_years = sorted(stock['year'].unique())
        _log_info(f"[S3_2 DEBUG] 存栏年份: {stock_years}")
    
    # Check incoming commodity values.
    if 'commodity' in stock.columns:
        unique_commodities = stock['commodity'].unique()
        _log_info(f"[S3_2 DEBUG] 传入的commodity值样例: {list(unique_commodities)[:10]}", level=2)
    
    stock['m49_code'] = _normalize_m49(stock['country'])
    
    # Map Item_Emis commodities such as 'Cattle, non-dairy' to Item_Feed_Map species such as beef_cattle.
    # Expected comm_to_species: {Item_Emis: Item_Feed_Map}.
    # Reversal of feed_item_to_comm may invert comm_to_species; correct if necessary.
    
    # Print comm_to_species and check dairy species.
    _log_info(f"[S3_2 DEBUG] comm_to_species映射数量: {len(comm_to_species)}")
    dairy_check = {k: v for k, v in comm_to_species.items() if 'dairy' in str(k).lower() and 'non-dairy' not in str(k).lower()}
    _log_info(f"[S3_2 DEBUG] comm_to_species中dairy品种: {dairy_check}", level=2)
    
    # Try direct mapping first.
    stock['species'] = stock['commodity'].map(comm_to_species)
    
    # Mostly NaN results suggest an inverted mapping; reverse it.
    unmapped_count = stock['species'].isna().sum()
    if unmapped_count > len(stock) * 0.5:  # If more than 50% are unmatched
        logger.warning(f"[S3_2 WARNING] comm_to_species映射失败率高({unmapped_count}/{len(stock)})，尝试反转映射...")
        # Reverse {commodity: feed_item} to {feed_item: commodity}.
        species_to_comm = {v: k for k, v in comm_to_species.items()}
        # Build the correct commodity-to-feed_item mapping.
        correct_mapping = {}
        for commodity in stock['commodity'].unique():
            if pd.isna(commodity):
                continue
            # Try matching within species_to_comm values.
            for feed_item, comm in species_to_comm.items():
                if comm == commodity:
                    correct_mapping[commodity] = feed_item
                    break
        _log_info(f"[S3_2 DEBUG] 构建的正确映射示例: {list(correct_mapping.items())[:5]}", level=2)
        stock['species'] = stock['commodity'].map(correct_mapping)
    
    # Supplement unmatched dairy species directly from dict_v3.
    still_unmapped = stock['species'].isna()
    if still_unmapped.any():
        unmapped_commodities = stock[still_unmapped]['commodity'].unique()
        dairy_unmapped = [c for c in unmapped_commodities if 'dairy' in str(c).lower() and 'non-dairy' not in str(c).lower()]
        
        if dairy_unmapped:
            logger.warning(f"[S3_2 DAIRY_FIX] 检测到{len(dairy_unmapped)}个dairy品种未映射，从dict_v3补充: {dairy_unmapped}")
            
            # Load dairy mappings directly from dict_v3.
            try:
                dict_v3_path = paths.dict_v3_path if hasattr(paths, 'dict_v3_path') else None
                if dict_v3_path and os.path.exists(dict_v3_path):
                    emis_df = read_excel_cached(dict_v3_path, sheet_name='Emis_item')
                    dairy_df = emis_df[['Item_Emis', 'Item_Feed_Map']].dropna(subset=['Item_Emis', 'Item_Feed_Map']).copy()
                    dairy_mask = dairy_df['Item_Emis'].astype(str).str.lower().str.contains('dairy')
                    dairy_mask &= ~dairy_df['Item_Emis'].astype(str).str.lower().str.contains('non-dairy')
                    dairy_df = dairy_df[dairy_mask]
                    dairy_mapping = dict(zip(dairy_df['Item_Emis'], dairy_df['Item_Feed_Map']))
                    
                    _log_info(f"[S3_2 DAIRY_FIX] 从dict_v3加载dairy映射: {dairy_mapping}", level=2)
                    
                    # Apply dairy mappings to unmatched rows.
                    for commodity in dairy_unmapped:
                        if commodity in dairy_mapping:
                            mask = (stock['commodity'] == commodity) & stock['species'].isna()
                            stock.loc[mask, 'species'] = dairy_mapping[commodity]
                            _log_info(f"[S3_2 DAIRY_FIX] 修复: {commodity} 转为 {dairy_mapping[commodity]} ({mask.sum()} 行)")
            except Exception as e:
                logger.error(f"[S3_2 DAIRY_FIX] 从dict_v3补充dairy映射失败: {e}")
    
    # Check for remaining unmapped rows.
    unmapped = stock['species'].isna()
    if unmapped.any():
        logger.warning(f"[S3_2 WARNING] {unmapped.sum()} 行commodity无法映射到species，将被过滤")
    
    stock['iso3'] = stock['iso3'].fillna(stock['country'].map(universe.iso3_by_country))
    
    _log_info(f"[S3_2 DEBUG] 映射后: m49_code缺失={stock['m49_code'].isna().sum()}, species缺失={stock['species'].isna().sum()}")
    
    # Trace species mappings for the USA.
    us_stock = stock[stock['country'] == 'United States of America']
    if not us_stock.empty:
        _log_info("\n" + "=" * 80, level=2)
        _log_info(" [美国数据流] Step 4: species映射完成", level=2)
        _log_info("=" * 80, level=2)
        _log_info(f"美国存栏数据: {len(us_stock)} 行", level=2)
        us_sample = us_stock[['commodity', 'species', 'stock_head', 'year']].head(10)
        for _, row in us_sample.iterrows():
            _log_info(f"  commodity={row['commodity']:15s} | species={row['species']:15s} | {row['stock_head']:>12,.0f} head ({row['year']}年)", level=2)
    stock = stock.dropna(subset=['m49_code', 'species'])
    _log_info(f"[S3_2 DEBUG] dropna后: {len(stock)} 行")
    
    # Check stock_head before filtering.
    _log_info("\n" + "=" * 80, level=2)
    _log_info(" [S3_2诊断] stock_head过滤前检查", level=2)
    _log_info("=" * 80, level=2)
    _log_info(f"stock_head列类型: {stock['stock_head'].dtype}", level=2)
    _log_info(f"stock_head非空行数: {stock['stock_head'].notna().sum()}/{len(stock)}", level=2)
    _log_info(f"stock_head>0行数（过滤前）: {(stock['stock_head'] > 0).sum()}/{len(stock)}", level=2)
    _log_info(f"stock_head总和: {stock['stock_head'].sum():,.0f}", level=2)
    if len(stock) > 0:
        _log_info(f"stock_head范围: {stock['stock_head'].min():.2e} ~ {stock['stock_head'].max():.2e}", level=2)
        _log_info(f"stock_head样例（前5行）: {stock['stock_head'].head().tolist()}", level=2)
    
    stock['stock_head'] = pd.to_numeric(stock['stock_head'], errors='coerce').fillna(0.0)
    
    # State after numeric conversion
    _log_info(f"numeric转换后stock_head>0行数: {(stock['stock_head'] > 0).sum()}/{len(stock)}", level=2)
    _log_info(f"numeric转换后stock_head总和: {stock['stock_head'].sum():,.0f}", level=2)
    
    stock = stock[stock['stock_head'] > 0]
    _log_info(f"[S3_2 DEBUG] 过滤stock_head>0后: {len(stock)} 行")
    _log_info("=" * 80 + "\n", level=2)
    
    if stock.empty:
        logger.error(f"[S3_2 ERROR]  存栏数据为空，提前返回！")
        return empty

    dm_per_head = _load_total_dm_per_head(paths.feed_need_xlsx, years, scheme=feed_requirement_scheme)
    crop_share = _load_crop_share(paths.feed_need_xlsx, years)
    dm_conversion = _load_dm_conversion(paths.feed_need_xlsx, years)
    grass_ratio = _load_grass_ratio(paths.grass_ratio_xlsx)
    pasture_yield = _load_pasture_yield(paths.pasture_dm_yield_xlsx)
    
    # Confirm parameter year coverage.
    if not dm_per_head.empty and 'year' in dm_per_head.columns:
        param_years = sorted(dm_per_head['year'].unique())
        _log_info(f"[S3_2 DEBUG] dm_per_head年份范围: {param_years[:3]}...{param_years[-3:]}, 共{len(param_years)}年")
        if 2080 in param_years:
            _log_info(f"[S3_2 DEBUG]  dm_per_head包含2080年数据（前向填充成功）")
        else:
            _log_info(f"[S3_2 DEBUG]  dm_per_head缺少2080年数据！")

    if dm_per_head.empty or crop_share.empty or dm_conversion.empty:
        logger.error(f"[S3_2 ERROR]  参数数据为空: dm_per_head={dm_per_head.empty}, crop_share={crop_share.empty}, dm_conversion={dm_conversion.empty}")
        return empty

    _log_info(f"[S3_2 DEBUG] merge前stock: {len(stock)} 行, dm_per_head: {len(dm_per_head)} 行")
    # Check species matching.
    stock_species = set(stock['species'].unique())
    dm_species = set(dm_per_head['species'].unique())
    _log_info(f"[S3_2 DEBUG] stock中的species ({len(stock_species)}个): {sorted(list(stock_species))[:10]}")
    _log_info(f"[S3_2 DEBUG] dm_per_head中的species ({len(dm_species)}个): {sorted(list(dm_species))[:10]}")
    overlap = stock_species & dm_species
    _log_info(f"[S3_2 DEBUG] 交集species: {len(overlap)} 个")
    if len(overlap) == 0:
        logger.error(f"[S3_2 ERROR]  stock和dm_per_head的species完全不匹配！")
        logger.error(f"[S3_2 ERROR] stock示例: {list(stock_species)[:5]}")
        logger.error(f"[S3_2 ERROR] dm_per_head示例: {list(dm_species)[:5]}")
    
    stock = stock.merge(
        dm_per_head,
        how='left',
        left_on=['species','m49_code','year'],
        right_on=['species','m49_code','year']
    )
    _log_info(f"[S3_2 DEBUG] merge后: {len(stock)} 行")
    
    stock['kg_dm_per_head'] = pd.to_numeric(stock['kg_dm_per_head'], errors='coerce')
    kg_dm_na = stock['kg_dm_per_head'].isna().sum()
    _log_info(f"[S3_2 DEBUG] kg_dm_per_head缺失: {kg_dm_na}/{len(stock)} 行")
    
    # Trace DM-per-head matching for the USA.
    us_stock = stock[stock['country'] == 'United States of America']
    if not us_stock.empty:
        _log_info("\n" + "=" * 80, level=2)
        _log_info(" [美国数据流] Step 5: DM per head参数匹配", level=2)
        _log_info("=" * 80, level=2)
        _log_info(f"美国存栏数据: {len(us_stock)} 行", level=2)
        us_sample = us_stock[['species', 'stock_head', 'kg_dm_per_head', 'year']].head(10)
        for _, row in us_sample.iterrows():
            dm_status = f"{row['kg_dm_per_head']:.1f}" if pd.notna(row['kg_dm_per_head']) else " NaN"
            _log_info(f"  {row['species']:15s} | {row['stock_head']:>12,.0f} head | DM={dm_status:>8s} kg/head ({row['year']}年)", level=2)
    
    stock = stock.dropna(subset=['kg_dm_per_head'])
    _log_info(f"[S3_2 DEBUG] dropna(kg_dm_per_head)后: {len(stock)} 行")
    
    if stock.empty:
        logger.error(f"[S3_2 ERROR]  merge dm_per_head后数据为空，提前返回！")
        logger.error(f"[S3_2 ERROR] 可能原因：存栏的species/m49_code/year组合在dm_per_head中找不到匹配")
        return empty
    # FeedEfficiency is a livestock dry-matter intensity multiplier.  Apply it
    # once to total DM before the grass/crop split.  The previous implementation
    # multiplied crop ``dm_fraction`` later, which inverted the direction
    # (a lower intensity increased grain demand) and did not match livestock
    # strategy keys.
    stock['feed_efficiency_multiplier'] = 1.0
    if conv_mult:
        multiplier_values = []
        for row in stock[['country', 'commodity', 'year', 'm49_code']].itertuples(index=False):
            country_aliases = [str(row.country).strip(), str(row.m49_code).strip()]
            value = 1.0
            for country_alias in country_aliases:
                lookup = conv_mult.get(
                    (country_alias, str(row.commodity).strip(), int(row.year))
                )
                if lookup is not None:
                    value = float(lookup)
                    break
            multiplier_values.append(max(0.0, value))
        stock['feed_efficiency_multiplier'] = np.asarray(multiplier_values, dtype=float)
    stock['dm_total_kg'] = (
        stock['stock_head']
        * stock['kg_dm_per_head']
        * stock['feed_efficiency_multiplier']
    )

    stock = stock.merge(
        grass_ratio,
        how='left',
        on=['species','m49_code']
    )
    stock['grass_ratio'] = stock['grass_ratio'].clip(lower=0.0, upper=1.0).fillna(0.0)
    stock['crop_ratio'] = stock['crop_ratio'].clip(lower=0.0, upper=1.0)
    stock['crop_ratio'] = stock['crop_ratio'].fillna(1.0 - stock['grass_ratio'])
    stock['crop_ratio'] = stock['crop_ratio'].clip(lower=0.0, upper=1.0)
    stock['grass_dm_kg'] = stock['dm_total_kg'] * stock['grass_ratio']
    stock['crop_dm_kg'] = stock['dm_total_kg'] * stock['crop_ratio']

    stock['livestock_commodity'] = stock['commodity']
    crop_dm_rows = stock[['country','iso3','m49_code','year','species','livestock_commodity','dm_total_kg','crop_ratio']].merge(
        crop_share,
        how='left',
        on=['species','m49_code','year']
    )
    crop_dm_rows['share'] = crop_dm_rows['share'].clip(lower=0.0)
    crop_dm_rows['share'] = crop_dm_rows['share'].fillna(0.0)
    crop_dm_rows['crop_dm_kg'] = crop_dm_rows['dm_total_kg'] * crop_dm_rows['crop_ratio'] * crop_dm_rows['share']
    crop_dm_rows = crop_dm_rows[crop_dm_rows['crop_dm_kg'] > 0]
    if crop_dm_rows.empty:
        crop_feed_demand = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','feed_t'])
    else:
        crop_dm_rows = crop_dm_rows.merge(
            dm_conversion,
            how='left',
            on=['m49_code','crop','year']
        )
        crop_dm_rows['dm_fraction'] = crop_dm_rows['dm_fraction'].replace(0, np.nan)
        crop_dm_rows = crop_dm_rows.dropna(subset=['dm_fraction'])
        crop_dm_rows['commodity'] = crop_dm_rows['crop'].map((maps.production_by_item or {}))
        crop_dm_rows['commodity'] = crop_dm_rows['commodity'].fillna(crop_dm_rows['crop'])
        crop_dm_rows['feed_commodity'] = crop_dm_rows['commodity']
        crop_dm_rows['grain_need_kg'] = crop_dm_rows['crop_dm_kg'] / crop_dm_rows['dm_fraction']
        crop_dm_rows = crop_dm_rows[crop_dm_rows['commodity'].isin(universe.commodities)]
        crop_dm_rows['feed_t'] = crop_dm_rows['grain_need_kg'] / 1000.0
        # Preserve M49_Country_Code by renaming m49_code to the standard column.
        if 'm49_code' in crop_dm_rows.columns:
            crop_dm_rows['M49_Country_Code'] = crop_dm_rows['m49_code']
        crop_feed_demand = crop_dm_rows.groupby(
            ['M49_Country_Code','country','iso3','year','commodity'],
            as_index=False
        )['feed_t'].sum()

    grass_req = stock.groupby(['country','iso3','m49_code','year'], as_index=False)['grass_dm_kg'].sum()
    _log_info(f"[S3_2 DEBUG] 草地DM需求聚合完成: {len(grass_req)}行, 年份范围: {grass_req['year'].min()}-{grass_req['year'].max()}")
    
    # Trace grassland DM demand for the USA.
    us_grass = grass_req[grass_req['country'] == 'United States of America']
    if not us_grass.empty:
        _log_info("\n" + "=" * 80, level=2)
        _log_info(" [美国数据流] Step 6: 草地DM需求计算", level=2)
        _log_info("=" * 80, level=2)
        for _, row in us_grass.iterrows():
            _log_info(f"  {row['year']}年: 草地DM需求 = {row['grass_dm_kg']:>15,.0f} kg", level=2)
    
    grass_req = grass_req.merge(pasture_yield, how='left', on='m49_code')
    
    # Check pasture_yield matching.
    missing_yield = grass_req['pasture_yield_kg_per_ha'].isna().sum()
    if missing_yield > 0:
        logger.warning(f"[S3_2 WARN]  {missing_yield}/{len(grass_req)}行缺失pasture_yield数据！")
        missing_countries = grass_req[grass_req['pasture_yield_kg_per_ha'].isna()]['country'].unique()
        logger.warning(f"[S3_2 WARN] 缺失yield的国家样例: {list(missing_countries)[:10]}")
    
    grass_req['grass_tdm'] = grass_req['grass_dm_kg'] / 1000.0
    grass_req['grass_area_need_ha'] = grass_req['grass_dm_kg'] / grass_req['pasture_yield_kg_per_ha'].replace(0, np.nan)
    
    # Check calculated areas.
    area_na_count = grass_req['grass_area_need_ha'].isna().sum()
    if area_na_count > 0:
        logger.warning(f"[S3_2 WARN]  {area_na_count}/{len(grass_req)}行的grass_area_need_ha为NaN（可能pasture_yield=0或NaN）")
    
    # Trace grassland area demand for the USA.
    us_grass_area = grass_req[grass_req['country'] == 'United States of America']
    if not us_grass_area.empty:
        _log_info("\n" + "=" * 80, level=2)
        _log_info(" [美国数据流] Step 7: 草地面积需求计算 (DM ÷ yield)", level=2)
        _log_info("=" * 80, level=2)
        for _, row in us_grass_area.iterrows():
            yield_val = row['pasture_yield_kg_per_ha']
            area_val = row['grass_area_need_ha']
            yield_str = f"{yield_val:,.0f}" if pd.notna(yield_val) else "NaN"
            area_str = f"{area_val:,.0f}" if pd.notna(area_val) else " NaN"
            _log_info(f"  {row['year']}年: 草地单产={yield_str:>10s} kg/ha | 面积需求={area_str:>15s} ha", level=2)
    
    # Preserve M49_Country_Code by renaming m49_code to the standard column.
    grass_req['M49_Country_Code'] = grass_req['m49_code']
    grass_requirement = grass_req[['M49_Country_Code','country','iso3','year','grass_tdm','grass_area_need_ha']].copy()
    
    _log_info(f"[S3_2 DEBUG]  grass_requirement生成完成: {len(grass_requirement)}行")
    for yr in [2020, 2080]:
        yr_data = grass_requirement[grass_requirement['year'] == yr]
        if not yr_data.empty:
            total_area = yr_data['grass_area_need_ha'].sum()
            valid_area = yr_data['grass_area_need_ha'].notna().sum()
            _log_info(f"[S3_2 DEBUG]   {yr}年: {len(yr_data)}行, 有效面积数据: {valid_area}行, 总面积: {total_area:,.0f} ha")

    species_dm_detail = stock[['country','iso3','m49_code','year','commodity','species',
                               'stock_head','kg_dm_per_head','feed_efficiency_multiplier',
                               'dm_total_kg','grass_dm_kg','crop_dm_kg']].copy()

    return FeedDemandOutputs(
        crop_feed_demand=crop_feed_demand,
        grass_requirement=grass_requirement,
        species_dm_detail=species_dm_detail,
        crop_feed_by_livestock=crop_dm_rows.copy(),
        grass_by_livestock=stock.merge(pasture_yield, how='left', on='m49_code'),
    )


def _parse_m49(val: Optional[str]) -> Optional[str]:
    if val is None or pd.isna(val):
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


def _normalize_m49(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(_parse_m49)

def _normalize_feed_requirement_scheme(scheme: Optional[str]) -> str:
    val = str(scheme or 'IPCC').strip().lower()
    if val == 'gleam':
        return 'gleam'
    return 'ipcc'

def _load_total_dm_per_head(xlsx_path: str, years: List[int], *, scheme: Optional[str] = None) -> pd.DataFrame:
    scheme_norm = _normalize_feed_requirement_scheme(scheme)
    if scheme_norm == 'gleam':
        df = read_excel_cached(xlsx_path, sheet_name='total_kgDM_per_head_GLEAM')
        df.columns = [str(c).strip() for c in df.columns]
        required = {'Species','M49_Country_Code'}
        if not required.issubset(set(df.columns)):
            raise KeyError("total_kgDM_per_head_GLEAM 表需包含新结构列: Species/M49_Country_Code/Yxxxx...")
        year_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit()]
        frames = []
        for col in year_cols:
            year = int(col[1:])
            if year not in years:
                continue
            tmp = df[['Species','M49_Country_Code', col]].copy()
            tmp = tmp.rename(columns={'Species':'species', col:'kg_dm_per_head'})
            tmp['year'] = year
            frames.append(tmp)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if out.empty:
            return out
        out['m49_code'] = _normalize_m49(out['M49_Country_Code'])
        out['year'] = pd.to_numeric(out['year'], errors='coerce').astype('Int64')
        out['kg_dm_per_head'] = pd.to_numeric(out['kg_dm_per_head'], errors='coerce')
        out = out[['species','m49_code','year','kg_dm_per_head']].dropna(subset=['m49_code','year','kg_dm_per_head'])
        out = _extend_years(out, ['species','m49_code'], 'kg_dm_per_head', years)
        return out

    df = read_excel_cached(xlsx_path, sheet_name='total_kgDM_per_head_IPCC')
    df.columns = [str(c).strip() for c in df.columns]
    required = {'Species','M49_Country_Code','year','total_kgDM_per_head'}
    if not required.issubset(set(df.columns)):
        raise KeyError("total_kgDM_per_head_IPCC 表需包含新结构列: Species/M49_Country_Code/year/total_kgDM_per_head")
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df = df[df['year'].isin(years)]
    df = df.rename(columns={'Species':'species','total_kgDM_per_head':'kg_dm_per_head'})
    out = df[['species','m49_code','year','kg_dm_per_head']].dropna(subset=['m49_code','year','kg_dm_per_head'])
    if out.empty:
        return out
    out = _extend_years(out, ['species','m49_code'], 'kg_dm_per_head', years)
    return out


def _load_crop_share(xlsx_path: str, years: List[int]) -> pd.DataFrame:
    df = read_excel_cached(xlsx_path, sheet_name='kgDM_per_head_crop_shares')
    df.columns = [str(c).strip() for c in df.columns]
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    df['crop'] = df['Crop'].astype(str).str.strip()
    value_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    frames = []
    for col in value_cols:
        year = int(col[1:])
        if year not in years:
            continue
        tmp = df[['Species','m49_code','crop', col]].copy()
        tmp = tmp.rename(columns={'Species':'species', col:'share'})
        tmp['year'] = year
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out = _extend_years(out, ['species','m49_code','crop'], 'share', years)
    return out


def _load_dm_conversion(xlsx_path: str, years: List[int]) -> pd.DataFrame:
    df = read_excel_cached(xlsx_path, sheet_name='dm_conversion_coefficients')
    df.columns = [str(c).strip() for c in df.columns]
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    df['crop'] = df['Crop'].astype(str).str.strip()
    value_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    frames = []
    for col in value_cols:
        year = int(col[1:])
        if year not in years:
            continue
        tmp = df[['m49_code','crop', col]].copy()
        tmp = tmp.rename(columns={col: 'dm_fraction'})
        tmp['year'] = year
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out = _extend_years(out, ['m49_code','crop'], 'dm_fraction', years)
    return out


def _load_grass_ratio(xlsx_path: str) -> pd.DataFrame:
    df = read_excel_cached(xlsx_path, sheet_name='country_level_weighted')
    df.columns = [str(c).strip() for c in df.columns]
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    df['species'] = df['Species'].astype(str).str.strip()
    df['grass_ratio'] = pd.to_numeric(df.get('Grass'), errors='coerce')
    df['crop_ratio'] = pd.to_numeric(df.get('Crop'), errors='coerce')
    return df[['species','m49_code','grass_ratio','crop_ratio']].dropna(subset=['m49_code','species'])


def _load_pasture_yield(xlsx_path: str) -> pd.DataFrame:
    df = read_excel_cached(xlsx_path, sheet_name='pasture_DM_yield')
    df.columns = [str(c).strip() for c in df.columns]
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    df['pasture_yield_kg_per_ha'] = pd.to_numeric(df.get('mean_AGB_kg_ha_weighted_by_area'), errors='coerce')
    return df[['m49_code','pasture_yield_kg_per_ha']].dropna(subset=['m49_code'])


def _extend_years(df: pd.DataFrame,
                  key_cols: List[str],
                  value_col: str,
                  years: List[int]) -> pd.DataFrame:
    """
    Extend data to all requested years.
     When grassland demand is calculated dynamically from optimized stocks, extend parameters to future years (2020-2080).
    Forward-fill the latest historical year, usually 2020, into all future years.
    
    Rationale:
    - DM per head is a technical parameter that is relatively stable in the short term.
    - Crop shares represent feed composition based on historical patterns.
    - Grass ratios represent husbandry practices assumed to continue.
    Represent future changes through scenarios rather than leaving parameters entirely missing.
    """
    if df.empty:
        return df
    pivot = df.pivot_table(index=key_cols, columns='year', values=value_col, aggfunc='last')
    
    # Create missing columns for every requested year.
    all_years = sorted(set(years))
    for y in all_years:
        if y not in pivot.columns:
            pivot[y] = np.nan
    
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.ffill(axis=1)  # Forward-fill across all years, including future years.
    pivot = pivot.reset_index()
    long_df = pivot.melt(id_vars=key_cols, var_name='year', value_name=value_col)
    long_df['year'] = long_df['year'].astype(int)
    long_df = long_df[long_df['year'].isin(all_years)]  # Retain all requested years.
    return long_df
