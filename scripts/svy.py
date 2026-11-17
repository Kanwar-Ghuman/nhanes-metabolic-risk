"""
Design-based estimation for NHANES (stratified, clustered, weighted).

Variances use Taylor-series linearization with PSUs treated as sampled with
replacement within strata -- the same estimator as R's survey package
(svydesign(ids=~psu, strata=~strata, weights=~w, nest=TRUE)).

Subpopulations are handled by zeroing weights outside the domain rather than
dropping rows, so every PSU still contributes to the variance. Dropping rows
before building the design understates standard errors.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


class Design:
    def __init__(self, data, weight, strata="SDMVSTRA", psu="SDMVPSU"):
        self.data = data
        self.w = data[weight].fillna(0).to_numpy(float)
        self.strata = data[strata].to_numpy()
        self.psu = data[psu].to_numpy()
        clusters = pd.MultiIndex.from_arrays([self.strata, self.psu])
        self._cluster, uniq = pd.factorize(clusters)
        cl_stratum = pd.factorize(uniq.get_level_values(0))[0]
        self._by_stratum = [np.flatnonzero(cl_stratum == h) for h in np.unique(cl_stratum)]

    def _weights(self, domain):
        if domain is None:
            return self.w
        return np.where(np.asarray(domain, bool), self.w, 0.0)

    def degf(self, domain=None):
        """Design degrees of freedom: #PSUs - #strata among units in the domain."""
        keep = self._weights(domain) > 0
        n_psu = len(set(zip(self.strata[keep], self.psu[keep])))
        return n_psu - len(set(self.strata[keep]))

    def vcov(self, infl):
        """Covariance of estimated totals given per-unit influence values (n x p)."""
        infl = np.asarray(infl, float)
        if infl.ndim == 1:
            infl = infl[:, None]
        p = infl.shape[1]
        totals = np.zeros((self._cluster.max() + 1, p))
        np.add.at(totals, self._cluster, infl)
        out = np.zeros((p, p))
        for idx in self._by_stratum:
            n_h = len(idx)
            if n_h < 2:
                continue
            d = totals[idx] - totals[idx].mean(axis=0)
            out += n_h / (n_h - 1) * d.T @ d
        return out

    def _ratio_infl(self, y, domain):
        y = np.asarray(y, float)
        dom = ~np.isnan(y)
        if domain is not None:
            dom = dom & np.asarray(domain, bool)
        w = self._weights(dom)
        y0 = np.nan_to_num(y)
        total_w = w.sum()
        est = (w * y0).sum() / total_w
        return est, w * (y0 - est) / total_w, dom

    def mean(self, y, domain=None):
        """Weighted mean (or proportion, for a 0/1 variable) with linearized SE."""
        est, z, _ = self._ratio_infl(y, domain)
        return est, float(np.sqrt(self.vcov(z)[0, 0]))

    def prop_ci(self, y, domain=None, level=0.95):
        """Proportion with a logit-scale CI (same idea as svyciprop(method='logit'))."""
        est, se = self.mean(y, domain)
        dom = ~np.isnan(np.asarray(y, float))
        if domain is not None:
            dom = dom & np.asarray(domain, bool)
        t = stats.t.ppf(0.5 + level / 2, self.degf(dom))
        logit, se_logit = np.log(est / (1 - est)), se / (est * (1 - est))
        lo, hi = logit - t * se_logit, logit + t * se_logit
        return est, se, 1 / (1 + np.exp(-lo)), 1 / (1 + np.exp(-hi))

    def quantile(self, y, q, domain=None):
        y = np.asarray(y, float)
        dom = ~np.isnan(y) if domain is None else (np.asarray(domain, bool) & ~np.isnan(y))
        order = np.argsort(y[dom])
        ys, ws = y[dom][order], self.w[dom][order]
        cdf = np.cumsum(ws) / ws.sum()
        return float(np.interp(q, cdf, ys))

    def diff_test(self, y, group, domain=None):
        """Design-based t-test for a difference in means between group==1 and group==0."""
        group = np.asarray(group, float)
        base = np.ones(len(group), bool) if domain is None else np.asarray(domain, bool)
        m1, z1, d1 = self._ratio_infl(y, base & (group == 1))
        m0, z0, d0 = self._ratio_infl(y, base & (group == 0))
        se = np.sqrt(self.vcov(z1 - z0)[0, 0])
        df = self.degf(d1 | d0)
        t = (m1 - m0) / se
        return 2 * stats.t.sf(abs(t), df)

    def chisq_test(self, cat, group, domain=None):
        """
        Adjusted Wald F-test that a categorical variable's distribution is the same
        in group==1 and group==0 (design-based analogue of a chi-square test).
        """
        cat = pd.Series(cat).astype("object")
        group = np.asarray(group, float)
        base = np.ones(len(group), bool) if domain is None else np.asarray(domain, bool)
        base = base & cat.notna().to_numpy()
        levels = list(pd.unique(cat[base]))
        diffs, infl = [], []
        for lv in levels[:-1]:
            ind = (cat == lv).astype(float).to_numpy()
            m1, z1, _ = self._ratio_infl(ind, base & (group == 1))
            m0, z0, _ = self._ratio_infl(ind, base & (group == 0))
            diffs.append(m1 - m0)
            infl.append(z1 - z0)
        d = np.array(diffs)
        V = self.vcov(np.column_stack(infl))
        q = len(d)
        df = self.degf(base & ~np.isnan(group))
        wald = float(d @ np.linalg.pinv(V) @ d)
        f = wald * (df - q + 1) / (df * q)
        return stats.f.sf(f, q, df - q + 1)

    def logit(self, y, X, domain=None, level=0.95):
        """
        Survey-weighted logistic regression. Point estimates are the weighted MLE;
        the covariance is the linearized sandwich A^-1 B A^-1 with B computed from
        PSU-level score totals. CIs and p-values use t with design df.
        """
        y = np.asarray(y, float)
        Xv = X.to_numpy(float)
        ok = ~np.isnan(y) & ~np.isnan(Xv).any(axis=1)
        if domain is not None:
            ok = ok & np.asarray(domain, bool)
        w = self._weights(ok)
        y0 = np.where(ok, y, 0.0)
        X0 = np.where(ok[:, None], Xv, 0.0)

        fit = sm.GLM(y0[ok], X0[ok], family=sm.families.Binomial(), var_weights=w[ok]).fit()
        beta = fit.params
        mu = 1 / (1 + np.exp(-X0 @ beta))
        bread = np.linalg.inv((X0 * (w * mu * (1 - mu))[:, None]).T @ X0)
        scores = X0 * (w * (y0 - mu))[:, None]
        cov = bread @ self.vcov(scores) @ bread

        se = np.sqrt(np.diag(cov))
        df = self.degf(ok)
        t = stats.t.ppf(0.5 + level / 2, df)
        res = pd.DataFrame({
            "term": X.columns,
            "beta": beta,
            "se": se,
            "OR": np.exp(beta),
            "lower": np.exp(beta - t * se),
            "upper": np.exp(beta + t * se),
            "p": 2 * stats.t.sf(np.abs(beta / se), df),
        })
        res.attrs.update(n=int(ok.sum()), df=df, pop=float(w.sum()))
        return res


def design_matrix(df, variables, levels):
    """Treatment-coded dummies named like R's model.matrix (e.g. 'bmi_cat30-34.9')."""
    cols = {"(Intercept)": np.ones(len(df))}
    for var in variables:
        values = df[var].astype("object")
        for lv in levels[var][1:]:
            col = (values == lv).astype(float)
            col[values.isna()] = np.nan
            cols[f"{var}{lv}"] = col.to_numpy()
    return pd.DataFrame(cols, index=df.index)
