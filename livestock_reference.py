"""Traceable reference reconciliation for the existing animal-flow stock chain."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def reconcile_livestock_reference(production, observed_stock, *, paths, universe, base_year=2020):
    from gle_emissions_complete import _get_cached_livestock_calculator, _normalize_m49_for_comparison
    from config_paths import get_src_base, get_input_base
    calculator = _get_cached_livestock_calculator(
        gle_params_path=str(Path(get_src_base())/'GLE_parameters.xlsx'),
        hist_production_path=paths.production_faostat_csv,
        hist_emissions_path=str(Path(get_input_base())/'Emission/Emissions_livestock_dairy_split.csv'),
        hist_manure_stock_path=paths.livestock_patterns_csv, dict_v3_path=paths.dict_v3_path)
    # Read unique product mappings explicitly; repeated process rows are not
    # additional products or additional herds.
    mapping=pd.read_excel(paths.dict_v3_path,sheet_name='Emis_item',
                          usecols=['Item_Emis','Item_Production_Map']).dropna().drop_duplicates()
    counts=mapping.groupby('Item_Emis').Item_Production_Map.nunique()
    if (counts>1).any():
        raise ValueError('Ambiguous livestock production mapping: '+str(counts[counts>1].to_dict()))
    mapping=mapping.set_index('Item_Emis').Item_Production_Map.to_dict()
    base=production.loc[production.year.eq(base_year)].copy()
    base['Commodity']=base.commodity.map(mapping).fillna(base.commodity)
    base['M49_Country_Code']=base.country.map(lambda c: _normalize_m49_for_comparison(universe.m49_by_country.get(c,c)))
    flows=pd.concat([calculator.calculate_milk_animals(base,base_year),
                     calculator.calculate_meat_animals(base,base_year)],ignore_index=True)
    if flows.empty:
        raise ValueError('No baseline animal flows for B04 reference calibration')
    flow_columns=['milk_animals','slaughtered','laying','producing_animals']
    numbers=flows.reindex(columns=flow_columns).apply(pd.to_numeric,errors='coerce').fillna(0.)
    flows['animal_flow']=numbers.where(numbers.gt(0)).bfill(axis=1).iloc[:,0].fillna(0.)
    flows['M49_Country_Code']=flows.M49_Country_Code.map(_normalize_m49_for_comparison)
    keys=['M49_Country_Code','Item_Emis']
    flow=flows.groupby(keys,as_index=False).animal_flow.sum()
    old=calculator.calculate_stock_from_animals(flows,base_year).groupby(keys,as_index=False).stock.sum().rename(columns={'stock':'stock_before'})
    obs=observed_stock.loc[observed_stock.year.eq(base_year)].copy()
    obs['M49_Country_Code']=obs.country.map(lambda c:_normalize_m49_for_comparison(universe.m49_by_country.get(c,c)))
    obs=obs.rename(columns={'commodity':'Item_Emis'}).groupby(keys,as_index=False).stock_head.sum()
    prod=base.rename(columns={'commodity':'Item_Emis'}).groupby(keys,as_index=False).production_t.sum()
    audit=obs.merge(flow,on=keys,how='outer').merge(old,on=keys,how='outer').merge(prod,on=keys,how='left')
    audit['observed_stock_head']=audit.stock_head
    # Missing observations are distinct from an observed zero. Reuse the
    # pre-existing animal/stock converter only for the former; do not invent a
    # regional feed coefficient or overwrite an observed herd/feed account.
    estimated=audit.stock_head.isna()&audit.production_t.gt(0)&audit.stock_before.gt(0)
    audit.loc[estimated,'stock_head']=audit.loc[estimated,'stock_before']
    audit['stock_reference_source']=np.where(estimated,'existing_animal_stock_converter_missing_observation','observed_stock')
    audit['theta_before']=audit.animal_flow/audit.stock_before.replace(0,np.nan)
    audit['theta_reference']=audit.animal_flow/audit.stock_head.replace(0,np.nan)
    audit['yield_per_stock']=audit.production_t/audit.stock_head.replace(0,np.nan)
    audit['yield_per_animal_reference']=audit.production_t/audit.animal_flow.replace(0,np.nan)
    valid=audit.animal_flow.gt(0)&audit.stock_head.gt(0)&audit.production_t.gt(0)
    audit['status']=np.where(valid,'reconciled_preserve_production_stock_feed','unidentified_reference')
    audit['stock_reproduced']=audit.animal_flow/audit.theta_reference
    background=(~audit.production_t.gt(0))&audit.stock_head.gt(0)
    audit.loc[background,'status']='background_herd_preserved_without_output_ratio'
    audit.loc[background,'stock_reproduced']=audit.loc[background,'stock_head']
    audit['relative_stock_residual']=(audit.stock_reproduced-audit.stock_head)/audit.stock_head.replace(0,np.nan)
    ratios={(str(r.M49_Country_Code),str(r.Item_Emis)):float(r.theta_reference)
            for r in audit[valid].itertuples(index=False)}
    return ratios,audit
