"""Country/crop seed input coefficients from unscaled reference-year use."""
from __future__ import annotations
from typing import Any
import math
import pandas as pd


def load_seed_reference_components(paths, universe) -> pd.DataFrame:
    """Read unscaled FBS Seed directly through the dictionary demand mapping.

    The legacy component allocator contains FAO product labels as targets;
    applying those labels as model keys loses Cassava/Fruits/etc. This focused
    loader changes only seed calibration, not the other demand components.
    """
    from S2_0_load_data import (_read_fbs_table, _filter_select_rows, _lc,
        _faostat_wide_to_long, _find_col, _maybe_find_col, _attach_country_from_m49)
    import re
    def normalized(value):
        return re.sub(r'[^a-z0-9]', '', str(value).lower())
    dictionary=pd.read_excel(paths.dict_v3_path,sheet_name='Emis_item')
    dictionary=dictionary.loc[dictionary.Item_Cat2.eq('Crop'),['Item_Emis','Item_Demand_Map']].dropna().drop_duplicates()
    lookup={normalized(row.Item_Demand_Map):row.Item_Emis for row in dictionary.itertuples(index=False)}
    raw=_filter_select_rows(_read_fbs_table(paths.fbs_csv))
    frame=_lc(_faostat_wide_to_long(raw))
    columns={key:_find_col(frame,[key]) for key in ['Area','Year','Item','Element','Value']}
    unit=_maybe_find_col(frame,['Unit'])
    selected=list(columns.values())+([unit] if unit else [])
    work=frame[selected].copy().rename(columns={v:k.lower() for k,v in columns.items()})
    work=work.rename(columns={'item':'item_raw'})
    work=_attach_country_from_m49(frame,work,universe,context='B03 unscaled FBS seeds')
    work['commodity']=work.item_raw.map(lambda value:lookup.get(normalized(value)))
    work=work[work.commodity.notna() & work.country.isin(universe.countries)].copy()
    work['value']=pd.to_numeric(work.value,errors='coerce')
    if unit:
        work['value']*=work[unit].astype(str).str.contains('1000',case=False,na=False).map({True:1000.,False:1.})
    keys=['country','commodity','year']
    present=work[keys].drop_duplicates()
    seeds=work[work.element.astype(str).str.strip().str.lower().eq('seed')].groupby(keys,as_index=False).value.sum(min_count=1).rename(columns={'value':'seed_t'})
    out=present.merge(seeds,on=keys,how='left',indicator=True)
    out['seed_source']=out['_merge'].map({'both':'FBS_reported_seed','left_only':'FBS_present_account_no_seed_element','right_only':'FBS_reported_seed'}).astype(str)
    out.loc[out['_merge'].eq('left_only'),'seed_t']=0.
    out['raw_fbs_seed_t']=out.seed_t
    # The existing amount sheet is an explicit second reference source. Keep
    # its provenance, including reported zeros; NaN is never implicitly zero.
    from pathlib import Path
    composition=Path(paths.fbs_csv).parent/'Demand_composition.xlsx'
    if composition.exists():
        amounts=pd.read_excel(composition,sheet_name='amount')
        amounts=amounts[amounts.Element.astype(str).str.lower().eq('seed')].copy()
        amounts['commodity']=amounts.Item.map(lambda x:lookup.get(normalized(x)))
        def country_key(value):
            try: return "'"+str(int(float(str(value).lstrip("'")))).zfill(3)
            except (ValueError,TypeError): return str(value)
        amounts['country']=amounts.M49_Country_Code.map(country_key)
        amounts['reference_seed_t']=pd.to_numeric(amounts.Y2020,errors='coerce')
        amounts['reference_seed_t']*=amounts.Unit.astype(str).str.contains('1000').map({True:1000.,False:1.})
        secondary=amounts.groupby(['country','commodity']).reference_seed_t.sum(min_count=1).to_dict()
        mask=out.year.eq(2020)&out.seed_t.isna()
        for index,row in out.loc[mask].iterrows():
            value=secondary.get((row.country,row.commodity))
            if value is not None and math.isfinite(value) and value>=0:
                out.loc[index,'seed_t']=value
                out.loc[index,'seed_source']='Demand_composition_amount_Y2020_FBS_missing'
    return out.drop(columns='_merge')


def calibrate_seed_coefficients(nodes: list[Any], components: pd.DataFrame, *, base_year: int,
                                crop_commodities: set[str], fallback_seed: dict | None = None) -> pd.DataFrame:
    base = {(str(n.country), str(n.commodity)): n for n in nodes if int(n.year)==base_year}
    raw = components[components.year.eq(base_year)].groupby(['country','commodity']).seed_t.sum(min_count=1).to_dict() if (
        isinstance(components,pd.DataFrame) and not components.empty and 'seed_t' in components
    ) else {}
    rows=[]
    for (country,commodity),node in base.items():
        if commodity not in crop_commodities:
            continue
        seed=raw.get((country,commodity))
        source='FBS_unscaled_seed_t'
        if 'seed_source' in components:
            matches=components.loc[components.year.eq(base_year)&components.country.eq(country)&components.commodity.eq(commodity),'seed_source']
            if len(matches): source=str(matches.iloc[0])
        if seed is None or not math.isfinite(float(seed)):
            seed=(fallback_seed or {}).get((country,commodity))
            source='unscaled_total_use_times_seed_share' if seed is not None else 'missing_seed_account'
        q=float(node.Q0)
        seed=float(seed) if seed is not None else math.nan
        if not math.isfinite(seed) or seed < 0:
            coef=math.nan; status='missing_seed_account'
        elif q > 0:
            coef=seed/q; status='calibrated'
        elif seed == 0:
            coef=math.nan; status='zero_reference_production'
        else:
            coef=math.nan; status='positive_seed_zero_production'
        rows.append(dict(country=country,commodity=commodity,base_year=base_year,production_t=q,
                         seed_t=seed,seed_coefficient=coef,source=source,status=status))
    return pd.DataFrame(rows)


def assign_seed_coefficients(nodes: list[Any], audit: pd.DataFrame) -> None:
    """Assign only identified coefficients; missing new production fails closed.

    Zero production and zero seed has no identifiable ratio. It is marked as
    uncalibrated (rather than given zero intensity) and future production is
    disallowed until an explicit coefficient is provided.
    """
    if audit.empty:
        return
    lookup=audit.set_index(['country','commodity']).to_dict('index')
    for n in nodes:
        row=lookup.get((str(n.country),str(n.commodity)))
        if row is None: continue
        n.meta=n.meta or {}
        coefficient=row['seed_coefficient']
        n.meta['seed_coefficient']=float(coefficient) if math.isfinite(coefficient) else 0.0
        n.meta['seed_reference_t']=float(row['seed_t']) if math.isfinite(row['seed_t']) else 0.0
        n.meta['seed_coefficient_unidentified']=not math.isfinite(coefficient)
