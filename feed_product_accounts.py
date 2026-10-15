"""Product-resolved feed intensities, with explicit source and unit records."""
from __future__ import annotations
from typing import Any
import math
import pandas as pd


def intensity_multiplier(intake_multiplier: float, productivity_multiplier: float,
                         *, basis: str = 'per_head') -> float:
    if not math.isfinite(intake_multiplier) or intake_multiplier < 0:
        raise ValueError('Feed multiplier must be finite and nonnegative')
    if basis == 'per_output':
        return intake_multiplier
    if basis != 'per_head' or not math.isfinite(productivity_multiplier) or productivity_multiplier <= 0:
        raise ValueError('Per-head feed conversion requires positive productivity')
    return intake_multiplier / productivity_multiplier


def parameter_value(mapping: dict | None, country: str, commodity: str, year: int) -> float:
    aliases=[country, country.lstrip("'")]
    for c in aliases + ['All','all']:
        for j in [commodity,'All','all']:
            value=(mapping or {}).get((c,j,year))
            if value is not None:
                return float(value)
    return 1.0


def stock_feed_multipliers(scenario: dict, countries, commodities, years, *,
                          base_year: int = 2020, feed_efficiency: float = 1.) -> dict:
    """Translate output-based efficiency back to intake for stock postprocessing.

    Future stocks already include productivity. Thus an output-based feed
    multiplier must be multiplied by productivity before applying it per head.
    Independent background herds have no product denominator.
    """
    if feed_efficiency <= 0 or not math.isfinite(feed_efficiency):
        raise ValueError('Feed efficiency must be positive and finite')
    basis = scenario.get('feed_multiplier_basis', 'per_head')
    result = {}
    for country in countries:
        for commodity in commodities:
            for year in years:
                if int(year) <= base_year:
                    continue
                key = (str(country), str(commodity), int(year))
                intake = parameter_value(scenario.get('dm_conversion_multiplier'), *key)
                productivity = parameter_value(scenario.get('yield_multiplier'), *key)
                intensity_multiplier(intake, productivity, basis=basis)
                if basis == 'per_output':
                    intake = (1. if key[:2] in scenario.get('background_stock_reference', {})
                              else intake * productivity)
                result[key] = intake / feed_efficiency
    return result


def build_product_coefficients(crop_detail: pd.DataFrame, grass_detail: pd.DataFrame,
                               production: pd.DataFrame, *, base_year: int, years: list[int],
                               intake_multiplier: dict | None = None, yield_multiplier: dict | None = None,
                               multiplier_basis: str = 'per_head', region_by_country: dict | None = None,
                               feed_efficiency: float = 1.0, background: dict | None = None) -> tuple[dict, dict, pd.DataFrame]:
    """Return crop coefficients (r,feed,livestock,t), pasture (r,livestock,t).

    Regional aggregation weights each product by its own reference production,
    never by total livestock output. Missing base production is an explicit
    data issue in the ledger and is not silently assigned a zero coefficient.
    """
    if crop_detail.empty or grass_detail.empty:
        raise ValueError('B04 requires product-resolved feed and grass source rows')
    if feed_efficiency <= 0 or not math.isfinite(feed_efficiency):
        raise ValueError('Feed efficiency must be positive and finite')
    prod=production.loc[production.year.eq(base_year)].groupby(['country','commodity']).production_t.sum().to_dict()
    crop=crop_detail[crop_detail.year.eq(base_year)].copy()
    crop=crop.groupby(['country','livestock_commodity','feed_commodity'],as_index=False).agg(feed_t=('feed_t','sum'),crop_dm_kg=('crop_dm_kg','sum'))
    grass=grass_detail[grass_detail.year.eq(base_year)].copy()
    grass=grass.groupby(['country','livestock_commodity'],as_index=False).agg(
        dm_total_kg=('dm_total_kg','sum'),grass_dm_kg=('grass_dm_kg','sum'),stock_head=('stock_head','sum'),
        pasture_yield_kg_per_ha=('pasture_yield_kg_per_ha','first'))
    grass_lookup=grass.set_index(['country','livestock_commodity']).to_dict('index')
    crop_dm_totals=crop.groupby(['country','livestock_commodity']).crop_dm_kg.sum().to_dict()
    rows=[]
    for rec in crop.itertuples(index=False):
        key=(rec.country,rec.livestock_commodity)
        q=float(prod.get(key,0.))
        g=grass_lookup.get(key,{})
        rows.append(dict(country=rec.country,livestock_commodity=rec.livestock_commodity,
                         feed_commodity=rec.feed_commodity,production_t=q,feed_t=rec.feed_t,
                         crop_dm_t=rec.crop_dm_kg/1000.,total_dm_t=g.get('dm_total_kg',0.)/1000.,
                         total_dm_coefficient=g.get('dm_total_kg',0.)/1000./q if q>0 else math.nan,
                         crop_dm_ratio=crop_dm_totals.get(key,0.)/g['dm_total_kg'] if g.get('dm_total_kg',0.)>0 else 0.,
                         grass_dm_ratio=g.get('grass_dm_kg',0.)/g['dm_total_kg'] if g.get('dm_total_kg',0.)>0 else 0.,
                         crop_dm_share=rec.crop_dm_kg/crop_dm_totals[key] if crop_dm_totals.get(key,0.)>0 else 0.,
                         crop_dry_matter_fraction=rec.crop_dm_kg/(rec.feed_t*1000.) if rec.feed_t>0 else math.nan,
                         crop_coefficient=rec.feed_t/q if q>0 else math.nan,
                         status='calibrated' if q>0 else 'positive_feed_zero_reference_production',
                         source='2020_product_stock_and_feed_parameters'))
    audit=pd.DataFrame(rows)
    coefficients={};pasture={}
    background = background if background is not None else {}
    background.update(crop={},pasture={},stock={},dm={},reference_totals=grass.copy())
    background['reference_totals']['production_t']=[float(prod.get((r.country,r.livestock_commodity),0.)) for r in grass.itertuples(index=False)]
    background['reference_totals']['background_herd']=background['reference_totals'].production_t.le(0)
    future=[int(y) for y in years if int(y)>base_year]
    # User-authorized policy: zero-output positive herds are a separate fixed
    # background activity, never a fictitious finite feed/output coefficient.
    for row in audit.itertuples(index=False):
        region=(region_by_country or {}).get(row.country,row.country)
        key=(row.country,row.livestock_commodity)
        coef=row.crop_coefficient
        if not math.isfinite(coef):
            audit.loc[(audit.country==row.country)&(audit.livestock_commodity==row.livestock_commodity)&(audit.feed_commodity==row.feed_commodity),['status','source']]=['background_herd','observed_stock_feed_without_reference_output']
            for t in future:
                mult=parameter_value(intake_multiplier,*key,t) if multiplier_basis=='per_head' else 1.
                rkey=(region,row.feed_commodity,t)
                background['crop'][rkey]=background['crop'].get(rkey,0.)+row.feed_t*mult/feed_efficiency
            continue
        for t in future:
            multiplier=intensity_multiplier(parameter_value(intake_multiplier,*key,t),
                parameter_value(yield_multiplier,*key,t),basis=multiplier_basis)
            rkey=(region,row.feed_commodity,row.livestock_commodity,t)
            if region_by_country:
                denominator=sum(q for (country,livestock),q in prod.items() if livestock==row.livestock_commodity and region_by_country.get(country,country)==region)
                weight=row.production_t/denominator if denominator>0 else 0.
            else:
                weight=1.
            coefficients[rkey]=coefficients.get(rkey,0.)+float(coef)*multiplier*weight/feed_efficiency
    for row in grass.itertuples(index=False):
        key=(row.country,row.livestock_commodity)
        q=float(prod.get(key,0.))
        if row.grass_dm_kg > 0 and (not math.isfinite(row.pasture_yield_kg_per_ha) or row.pasture_yield_kg_per_ha<=0):
            raise ValueError(f'Missing pasture productivity for {key}')
        if q<=0:
            background['stock'][key]=float(row.stock_head)
            region=(region_by_country or {}).get(row.country,row.country)
            for t in [base_year]+future:
                mult=1. if t==base_year or multiplier_basis=='per_output' else parameter_value(intake_multiplier,*key,t)
                rkey=(region,t)
                area=row.grass_dm_kg/row.pasture_yield_kg_per_ha if row.grass_dm_kg>0 else 0.
                background['pasture'][rkey]=background['pasture'].get(rkey,0.)+area*mult/feed_efficiency
                background['dm'][(row.country,row.livestock_commodity,t)]=row.dm_total_kg/1000.*mult/feed_efficiency
            continue
        base=row.grass_dm_kg/row.pasture_yield_kg_per_ha/q if row.grass_dm_kg>0 else 0.
        region=(region_by_country or {}).get(row.country,row.country)
        for t in [base_year]+future:
            multiplier=1. if t==base_year else intensity_multiplier(parameter_value(intake_multiplier,*key,t),parameter_value(yield_multiplier,*key,t),basis=multiplier_basis)
            if region_by_country:
                denominator=sum(v for (c,l),v in prod.items() if l==row.livestock_commodity and region_by_country.get(c,c)==region)
                weight=q/denominator if denominator>0 else 0.
            else:weight=1.
            rkey=(region,row.livestock_commodity,t)
            pasture[rkey]=pasture.get(rkey,0.)+base*multiplier*weight/feed_efficiency
    return coefficients,pasture,audit


def product_feed_expression(coefficients: dict, supply: dict, region: str, feed_crop: str, year: int):
    import gurobipy as gp
    terms=[float(coef)*supply[(region,livestock,year)] for (r,c,livestock,t),coef in coefficients.items()
           if r==region and c==feed_crop and t==year and (region,livestock,year) in supply]
    return gp.quicksum(terms) if terms else gp.LinExpr(0.)


def bounded_feed_credit(model, gross_feed, potential: float, *, name: str):
    """Optional realized coproduct use, capped once after livestock aggregation."""
    if not math.isfinite(potential) or potential < 0:
        raise ValueError('Coproduct potential must be finite and nonnegative')
    credit=model.addVar(lb=0.,ub=potential,name=f'feed_credit[{name}]')
    model.addConstr(credit<=gross_feed,name=f'feed_credit_gross_cap[{name}]')
    return credit
