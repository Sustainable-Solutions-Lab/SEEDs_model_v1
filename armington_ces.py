"""Calibrated two-origin CES accounts and sequential-LP integration.

Quantities in the physical ledger are tonnes. ``composite`` is a CES quantity
index calibrated to physical absorption only at reference prices. Source
prices and the CES price use the same commodity price normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CESAccount:
    domestic: float
    imports: float
    domestic_price: float
    import_price: float
    sigma: float

    def __post_init__(self) -> None:
        values = (self.domestic, self.imports, self.domestic_price, self.import_price, self.sigma)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('CES calibration requires finite quantities, prices and elasticity')
        if min(self.domestic, self.imports) < 0 or self.domestic + self.imports <= 0:
            raise ValueError('CES reference sources must be nonnegative with positive absorption')
        if min(self.domestic_price, self.import_price, self.sigma) <= 0:
            raise ValueError('CES prices and elasticity must be strictly positive')

    @property
    def quantity(self) -> float:
        return self.domestic + self.imports

    @property
    def price(self) -> float:
        return (self.domestic * self.domestic_price + self.imports * self.import_price) / self.quantity

    def hicks(self, domestic_price: float, import_price: float) -> tuple[float, float, float]:
        """Return unit cost and physical source quantities per composite unit.

        Calibrated-share form includes the sigma=1 Cobb–Douglas limit.
        A reference source with zero flow remains a structural zero; no
        artificial epsilon trade flow or silently imputed elasticity is used.
        """
        if self.domestic == 0 and import_price > 0:
            return import_price, 0., 1.
        if self.imports == 0 and domestic_price > 0:
            return domestic_price, 1., 0.
        if min(domestic_price, import_price) <= 0:
            raise ValueError('CES evaluation requires positive source prices')
        sd = self.domestic * self.domestic_price / (self.quantity * self.price)
        si = 1.0 - sd
        ld = math.log(domestic_price / self.domestic_price)
        li = math.log(import_price / self.import_price)
        power = 1.0 - self.sigma
        if abs(power) < 1e-8:
            lp = sd * ld + si * li
        else:
            terms = [math.log(s) + power * p for s, p in ((sd, ld), (si, li)) if s > 0]
            peak = max(terms)
            lp = (peak + math.log(sum(math.exp(v - peak) for v in terms))) / power
        cost = self.price * math.exp(lp)
        hd = (self.domestic / self.quantity) * math.exp(self.sigma * (lp - ld))
        hi = (self.imports / self.quantity) * math.exp(self.sigma * (lp - li))
        return cost, hd, hi

    def aggregate(self, domestic: float, imports: float) -> float:
        if min(domestic, imports) < 0:
            raise ValueError('CES source quantities cannot be negative')
        if self.domestic == 0:
            return imports
        if self.imports == 0:
            return domestic
        sd = self.domestic * self.domestic_price / (self.quantity * self.price)
        if self.sigma == 1.0:
            if min(domestic, imports) == 0:
                return 0.0
            return self.quantity * math.exp(sd * math.log(domestic / self.domestic) + (1-sd) * math.log(imports / self.imports))
        rho = (self.sigma - 1.0) / self.sigma
        if min(domestic, imports) == 0 and rho < 0:
            return 0.0
        return self.quantity * (sd * (domestic / self.domestic) ** rho + (1-sd) * (imports / self.imports) ** rho) ** (1/rho)


def balance_reference_accounts(frame: pd.DataFrame) -> pd.DataFrame:
    """Produce a separate, deterministic physical reference ledger.

    Production is preserved. First remove the *minimum* common import/export
    amount needed to keep exports below production. This is an accounting
    reconciliation, not an observation of re-exports. Clip the remaining
    impossible exports to production. Within each commodity reduce only the
    larger global trade margin proportionally to the smaller one. Absorption
    then follows Q+I-X. Every alteration is retained alongside original values.
    No raw input is written and no missing price/elasticity is filled here.
    """
    out = frame.copy()
    cols = ['production', 'imports', 'exports']
    if out.duplicated(['region','commodity']).any():
        raise ValueError('Reference accounts must have unique region/commodity keys')
    if not np.isfinite(out[cols].to_numpy(dtype=float)).all() or (out[cols] < 0).any().any():
        raise ValueError('Reference production and trade must be finite and nonnegative')
    for col in cols:
        out[f'original_{col}'] = out[col]
    common = np.minimum(out.imports, np.maximum(out.exports - out.production, 0.0))
    out['common_flow_removed'] = common
    out['imports'] -= common
    out['exports'] = np.minimum(out.exports - common, out.production)
    for _, group in out.groupby('commodity', sort=False):
        total_i, total_x = float(group.imports.sum()), float(group.exports.sum())
        balanced = min(total_i, total_x)
        if total_i > 0:
            out.loc[group.index, 'imports'] *= balanced / total_i
        if total_x > 0:
            out.loc[group.index, 'exports'] *= balanced / total_x
    out['domestic_sales'] = out.production - out.exports
    out['absorption'] = out.domestic_sales + out.imports
    out['original_absorption'] = out.original_production + out.original_imports - out.original_exports
    for col in ['imports', 'exports', 'absorption']:
        out[f'adjustment_{col}'] = out[col] - out[f'original_{col}']
    out['calibration_rule'] = 'preserve_production_minimum_common_flow_then_downscale_larger_global_trade_margin'
    return out


def attach_ces_sources(model: Any, *, idx: dict, qs: dict, qd: dict, net: dict,
                       pd_vars: dict, pm_vars: dict, trade_net: dict, trade_volume: dict,
                       sigma_map: dict, base_year: int, quantity_scale: float,
                       price_reference: dict, relative_price: bool, bioenergy: dict,
                       output_dir: str | None, external_balance: bool = False,
                       entry_region_map: dict | None = None) -> dict:
    """Create physical source balances and CES local approximation rows."""
    import gurobipy as gp
    from pathlib import Path
    rows = []
    pairs = sorted({(r,j) for r,j,_ in net})
    for r,j in pairs:
        key = next(k for k in net if k[:2] == (r,j))
        row = idx.get((r,j,base_year), idx[key])
        volume = float(trade_volume.get((r,j), 0.0))
        ni = float(trade_net.get((r,j), 0.0))
        rows.append(dict(region=r, commodity=j, production=max(0., float(row['Q0'])),
                         imports=max(0., (volume+ni)/2), exports=max(0., (volume-ni)/2),
                         reference_price=float(row['P0'])))
    ledger = balance_reference_accounts(pd.DataFrame(rows))
    # Existing model input uses a common commodity world-price benchmark.
    # Some zero-production nodes still contain the generic Node default P0=1;
    # use the same observed commodity benchmark for entry accounts as well.
    ledger['original_reference_price']=ledger.reference_price
    ledger['reference_price']=ledger.groupby('commodity').reference_price.transform('median')
    ledger['reference_price_source']='common_commodity_benchmark_median_of_model_reference_prices'
    if output_dir:
        destination = Path(output_dir) / 'Diagnostics'
        destination.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(destination / 'armington_reference_accounts.csv', index=False)
    lookup = ledger.set_index(['region','commodity']).to_dict('index')
    entry_parameters = regional_import_entry_parameters(ledger, entry_region_map or {})
    entry_lookup = entry_parameters.set_index(['region','commodity']).to_dict('index') if not entry_parameters.empty else {}
    if output_dir:
        entry_parameters.to_csv(Path(output_dir)/'Diagnostics'/'armington_import_entry_parameters.csv',index=False)
    state: dict[str, Any] = {'records': {}, 'ledger': ledger, 'history': [], 'converged': False,
                            'linearization_slacks': []}
    for key in net:
        r,j,t = key
        row = lookup[r,j]
        sigma = sigma_map.get((r,j), sigma_map.get(j))
        if sigma is None or not math.isfinite(float(sigma)) or float(sigma) <= 0:
            raise ValueError(f'CES elasticity missing or invalid: {(r,j)}')
        pref = float(price_reference.get(j,1.0)) if relative_price else 1.0
        p0 = row['reference_price'] / pref
        d0, i0 = row['domestic_sales'], row['imports']
        reference_status = 'calibrated_positive_absorption'
        entry=entry_lookup.get((r,j))
        if entry and entry['status']=='regional_import_entry':
            # A future entry prior, not a revision of observed 2020 trade.
            absorption=max(d0+i0,1.)
            i0=absorption*entry['import_share'];d0=absorption-i0
            reference_status='zero_import_regional_entry_prior'
        if d0+i0 <= 0:
            # At zero absorption the mix is unidentified. Explicit entry policy:
            # an observed producer starts domestic-only; a nonproducer starts
            # import-only. Unit normalization here is not an invented trade flow
            # in the preserved calibration ledger.
            d0, i0 = (1.,0.) if row['production'] > 0 else (0.,1.)
            reference_status = 'zero_absorption_domestic_entry' if d0 else 'zero_absorption_import_entry'
        account = CESAccount(d0, i0, p0, p0, float(sigma))
        dp, mp = pd_vars[key], pm_vars[j,t]
        if d0 > 0:
            dp.LB = max(float(dp.LB), p0 * 1e-5)
        if i0 > 0:
            mp.LB = max(float(mp.LB), p0 * 1e-5)
        domestic = model.addVar(lb=0., name=f'Qdom[{r},{j},{t}]')
        imp = model.addVar(lb=0., name=f'Imports[{r},{j},{t}]')
        exp = model.addVar(lb=0., name=f'Exports[{r},{j},{t}]')
        composite = model.addVar(lb=0., name=f'CES_quantity[{r},{j},{t}]')
        price = model.addVar(lb=p0*1e-5, name=f'CES_price[{r},{j},{t}]')
        physical = qd[key] + float(bioenergy.get(key,0.))/quantity_scale
        model.addConstr(domestic + exp == qs[key], name=f'ces_production_split[{r},{j},{t}]')
        model.addConstr(domestic + imp == physical, name=f'ces_physical_use[{r},{j},{t}]')
        if not external_balance:
            model.addConstr(imp-exp == net[key], name=f'ces_net_trade[{r},{j},{t}]')
        cost, hd, hi = account.hicks(p0,p0)
        cprice = model.addConstr(price == hd*dp+hi*mp, name=f'ces_price_index[{r},{j},{t}]')
        # Feasibility restoration is allowed only for intermediate LPs. The
        # exact nonlinear residual (including source demands) must vanish at
        # acceptance; these are not economic trade-deviation variables.
        sp = model.addVar(lb=0., name=f'ces_iteration_pos[{r},{j},{t}]')
        sn = model.addVar(lb=0., name=f'ces_iteration_neg[{r},{j},{t}]')
        state['linearization_slacks'].extend([sp, sn])
        cimport = model.addConstr(imp == hi/(hd+hi)*physical+sp-sn, name=f'ces_source_demand[{r},{j},{t}]')
        cquantity = model.addConstr(composite == physical/(hd+hi), name=f'ces_quantity_bridge[{r},{j},{t}]')
        state['records'][key] = dict(account=account, reference_status=reference_status,
             domestic=domestic, imports=imp, exports=exp,
             composite=composite, price=price, pd=dp, pm=mp, physical=physical, qd=qd[key],
             bio=float(bioenergy.get(key,0.))/quantity_scale, cprice=cprice, cimport=cimport,
             cquantity=cquantity, iterate=(p0,p0,max((d0+i0)/quantity_scale,1e-9)))
    model.update()
    return state


def regional_import_entry_parameters(ledger: pd.DataFrame, region_map: dict) -> pd.DataFrame:
    """User-authorized zero-import entry priors from the same market region.

    Pool only same-commodity accounts with both reference sources positive.
    No same-region donor means no imputation; original trade is never changed.
    """
    columns=['region','commodity','market_region','import_share','donor_count','donors','status']
    if not region_map:
        return pd.DataFrame(columns=columns)
    table=ledger.copy()
    table['market_region']=table.region.map(region_map)
    donors=table[table.domestic_sales.gt(0)&table.imports.gt(0)&table.market_region.notna()]
    rows=[]
    for row in table[table.imports.eq(0)&table.production.gt(0)].itertuples(index=False):
        eligible=donors[donors.commodity.eq(row.commodity)&donors.market_region.eq(row.market_region)]
        share=float(eligible.imports.sum()/eligible.absorption.sum()) if len(eligible) else math.nan
        rows.append(dict(region=row.region,commodity=row.commodity,market_region=row.market_region,
            import_share=share,donor_count=len(eligible),donors=';'.join(eligible.region.astype(str)),
            status='regional_import_entry' if 0<share<1 else 'zero_import_no_same_region_donor'))
    return pd.DataFrame(rows,columns=columns)


def update_ces_approximation(model: Any, state: dict, *, damping: float = 1.0) -> float:
    """Linearize exact CES equations around the current positive-price iterate.

    Return the maximum scaled nonlinear residual before changing any rows.
    Tonne ledgers remain hard equalities at every inner LP solve.
    """
    error = 0.0
    values = []
    worst = []
    for key, rec in state['records'].items():
        dp, mp = rec['pd'].X, rec['pm'].X
        physical = rec['qd'].X + rec['bio']
        cost, hd, hi = rec['account'].hicks(dp,mp)
        h = hd+hi
        share = hi/h
        vals = (rec['price'].X-cost, rec['imports'].X-share*physical,
                rec['composite'].X-physical/h)
        err = max(abs(vals[0])/max(cost,1e-8), abs(vals[1])/max(physical,1e-6),
                  abs(vals[2])/max(physical/h,1e-6))
        error = max(error,err)
        if err > 1e-7:
            worst.append(dict(key=list(key),residual=err,pd=dp,pm=mp,physical=physical,
                              price_error=vals[0],imports_error=vals[1],quantity_error=vals[2]))
        values.append((key, dp,mp,physical,cost,hd,hi))
    state['history'].append(dict(iteration=len(state['history']), max_relative_residual=error,
                                worst=sorted(worst,key=lambda row:row['residual'],reverse=True)[:5]))
    if error <= 1e-7:
        state['converged'] = True
        return error
    for key, dp,mp,physical,_,_,_ in values:
        rec = state['records'][key]
        old = rec['iterate']
        damping=state.get('damping_override',min(damping,0.3))
        dp,mp,physical = tuple(damping*a+(1-damping)*b for a,b in zip((dp,mp,physical),old))
        account = rec['account']
        cost,hd,hi = account.hicks(dp,mp)
        h=hd+hi; share=hi/h
        # Shephard's lemma gives exact unit-cost derivatives. The physical
        # import share derivative follows the conditional source-demand ratio.
        dshare_dp=account.sigma*share*(1-share)/max(dp,1e-12)
        dshare_mp=-account.sigma*share*(1-share)/max(mp,1e-12)
        rows=[(rec['cprice'],rec['price'],0.,hd,hi,cost-hd*dp-hi*mp),
              (rec['cimport'],rec['imports'],share,physical*dshare_dp,physical*dshare_mp,
               -physical*(dshare_dp*dp+dshare_mp*mp)),
              # Composite is an accounting index, absent from physical demand
              # equations and the objective. A frozen positive bridge avoids a
              # Taylor step giving a negative index for a large price step.
              (rec['cquantity'],rec['composite'],1/h,0.,0.,0.)]
        for con,var,qcoef,pcoef,mcoef,intercept in rows:
            model.chgCoeff(con,rec['qd'],-qcoef)
            model.chgCoeff(con,rec['pd'],-pcoef)
            model.chgCoeff(con,rec['pm'],-mcoef)
            con.RHS=intercept+qcoef*rec['bio']
        rec['iterate']=(dp,mp,physical)
    # Keep successive price steps within the neighborhood where the source
    # equation is informative. These bounds are iteration safeguards only.
    bounds=state.setdefault('price_bounds',{})
    for rec in state['records'].values():
        # World price is shared and determined by the existing price closure;
        # separate country records must not impose inconsistent bounds on it.
        for name,center in [('pd',rec['iterate'][0])]:
            var=rec[name]
            if var.index not in bounds:
                bounds[var.index]=(var,float(var.LB),float(var.UB))
            _,lower,upper=bounds[var.index]
            factor=state.get('trust_factor',2.)
            var.LB=max(lower,center/factor)
            var.UB=min(upper,max(center*factor,lower))
    model.update()
    return error


def prepare_quantity_projection(model: Any, cache: dict) -> bool:
    """Eliminate uncoupled CES prices in the exact-nutrition allocation branch.

    With fixed physical food demand and no price-elastic supply, the domestic
    price is an inverse conditional-source-demand function. World prices are
    commodity numeraires, not averages of those inverse prices. The projection
    is valid ONLY when no remaining equation/objective uses a source or consumer
    price. Other demand branches continue to solve the coupled CES equations.
    """
    state = cache.get('armington_ces', {})
    if state.get('method') == 'analytic_price_elimination':
        return True
    if not cache.get('nutrition_supply_driven') or not state.get('records'):
        return False
    model.update()
    remove = {rec[name].index for rec in state['records'].values()
              for name in ('cprice', 'cimport', 'cquantity')}
    for _, j, t in state['records']:
        con = model.getConstrByName(f'world_price[{j},{t}]')
        if con is not None:
            remove.add(con.index)
    # Audit price independence before modifying the model, rather than silently
    # replacing a price-responsive demand equation with an ex-post calculation.
    matrix = model.getA().tocsc()
    price_vars = {rec[name].index: rec[name] for rec in state['records'].values()
                  for name in ('pd', 'pm', 'price', 'composite')}
    for index, var in price_vars.items():
        dependencies = set(matrix.indices[matrix.indptr[index]:matrix.indptr[index+1]]) - remove
        if dependencies or abs(var.Obj) > 0:
            raise ValueError(f'CES price elimination is invalid for {var.VarName}: '
                             f'{len(dependencies)} coupled equations, objective={var.Obj}')
    constraints = model.getConstrs()
    model.remove([constraints[index] for index in sorted(remove)])
    # A small interior domain prevents infinite inverse prices when both
    # reference sources are positive. It does not create a missing source.
    # This parameter is disclosed and sensitivity-checked with the run evidence.
    interior = float(state.get('interior_share', 1e-7))
    for key, rec in state['records'].items():
        account = rec['account']
        physical = rec['qd'] + rec['bio']
        if account.domestic == 0:
            model.addConstr(rec['domestic'] == 0, name=f'ces_import_only{key}')
        elif account.imports == 0:
            model.addConstr(rec['imports'] == 0, name=f'ces_domestic_only{key}')
        else:
            model.addConstr(rec['imports'] >= interior*physical, name=f'ces_import_interior{key}')
            model.addConstr(rec['domestic'] >= interior*physical, name=f'ces_domestic_interior{key}')
        # These are eliminated auxiliary coordinates. Reported values are the
        # exact inverse solution, never these placeholder LP coordinates.
        for name in ('pd', 'pm', 'price'):
            rec[name].LB = account.price
            rec[name].UB = account.price
        rec['composite'].LB = 0
        rec['composite'].UB = 0
    for var in state['linearization_slacks']:
        var.UB = 0
    state.update(method='analytic_price_elimination', interior_share=interior,
                 price_closure='commodity_world_benchmark_numeraire', history=[])
    model.update()
    return True


def recover_projected_ces(state: dict) -> float:
    """Recover and independently verify exact CES coordinates of an LP optimum."""
    error = 0.0
    for key, rec in state['records'].items():
        a = rec['account']
        d, i = max(0., rec['domestic'].X), max(0., rec['imports'].X)
        physical = max(0., rec['qd'].X + rec['bio'])
        pd, pm = a.domestic_price, a.import_price
        if min(a.domestic, a.imports) > 0 and physical > 1e-6:
            if min(d, i) <= 0:
                raise ValueError(f'CES interior lost to solver tolerance: {key}, D={d}, I={i}')
            log_relative = (math.log(i/d) - math.log(a.imports/a.domestic))/a.sigma
            pd = a.domestic_price*math.exp(log_relative)
        cost, hd, hi = a.hicks(pd, pm)
        composite = a.aggregate(d, i)
        residual = max(abs(d-composite*hd), abs(i-composite*hi), abs(d+i-physical))/max(physical, 1.)
        expenditure = pd*d + pm*i
        residual = max(residual, abs(cost*composite-expenditure)/max(expenditure, 1.))
        error = max(error, residual)
        rec['recovered'] = dict(pd=pd, pm=pm, price=cost, composite=composite,
                                relative_residual=residual)
    state['history'].append(dict(method=state['method'], max_relative_residual=error,
                                  interior_share=state['interior_share']))
    state['converged'] = error <= 1e-7
    if not state['converged']:
        raise RuntimeError(f'Projected CES verification failed: residual={error}')
    return error


def ces_value(record: dict, field: str) -> float:
    if field in record.get('recovered', {}):
        return float(record['recovered'][field])
    return float(record[field].X)
