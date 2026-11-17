"""
Step 4: weighted descriptive statistics, diabetes prevalence, and the main
survey-weighted logistic regression, plus sensitivity analyses.

Main model (MEC weights, complete cases):
    diabetes ~ physical activity + BMI category + age group + sex + income-to-poverty + race/ethnicity
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from nhanes import LABELS, LEVELS, MODEL_VARS, RESULTS, W_FAST, W_MEC, load_analytic
from svy import Design, design_matrix

CONTINUOUS = {
    "RIDAGEYR": "Age (years)",
    "BMXBMI": "BMI (kg/m²)",
    "BMXWAIST": "Waist circumference (cm)",
    "LBXGH": "HbA1c (%)",
    "sbp": "Systolic BP (mmHg)",
    "mvpa_min_wk": "MVPA (moderate-equivalent min/week)",
    "sed_hr": "Sedentary time (h/day)",
    "DR1TKCAL": "Energy intake (kcal/day)",
    "DR1TSUGR": "Total sugars (g/day)",
    "DR1TFIBE": "Dietary fiber (g/day)",
}


def descriptives(df, des, analytic):
    rows = []
    for col, label in CONTINUOUS.items():
        for status, dom in [("All", analytic), ("No diabetes", analytic & (df["diabetes"] == 0)),
                            ("Diabetes", analytic & (df["diabetes"] == 1))]:
            dom = dom.to_numpy()
            x = df[col].to_numpy(float)
            mean, se = des.mean(x, dom)
            q25, q50, q75 = (des.quantile(x, q, dom) for q in (0.25, 0.5, 0.75))
            rows.append({"variable": label, "group": status, "n": int((dom & ~np.isnan(x)).sum()),
                         "missing": int((dom & np.isnan(x)).sum()), "mean": mean, "se": se,
                         "p25": q25, "median": q50, "p75": q75})
    return pd.DataFrame(rows)


def prevalence(df, des, analytic):
    rows = []
    outcomes = {"Total diabetes": "diabetes", "Undiagnosed": "dm_undiagnosed"}
    for name, col in outcomes.items():
        est, se, lo, hi = des.prop_ci(df[col], analytic.to_numpy())
        rows.append({"outcome": name, "variable": "Overall", "level": "All adults",
                     "n": int(analytic.sum()), "prev": est, "lower": lo, "upper": hi})
    for var in MODEL_VARS:
        for lv in LEVELS[var]:
            dom = (analytic & (df[var] == lv)).to_numpy()
            est, se, lo, hi = des.prop_ci(df["diabetes"], dom)
            rows.append({"outcome": "Total diabetes", "variable": LABELS[var], "level": lv,
                         "n": int(dom.sum()), "prev": est, "lower": lo, "upper": hi})
    return pd.DataFrame(rows)


def fit(df, weight, domain, levels=LEVELS, outcome="diabetes"):
    X = design_matrix(df, MODEL_VARS, levels)
    return Design(df, weight).logit(df[outcome], X, domain)


def unweighted(df, domain):
    X = design_matrix(df, MODEL_VARS, LEVELS)
    keep = domain & X.notna().all(axis=1) & df["diabetes"].notna()
    res = sm.GLM(df.loc[keep, "diabetes"], X[keep], family=sm.families.Binomial()).fit()
    ci = np.exp(res.conf_int())
    return pd.DataFrame({"term": X.columns, "beta": res.params.values, "se": res.bse.values,
                         "OR": np.exp(res.params.values), "lower": ci[0].values, "upper": ci[1].values,
                         "p": res.pvalues.values})


def main():
    df = load_analytic()
    des = Design(df, W_MEC)
    analytic = df["analytic"] == 1

    desc = descriptives(df, des, analytic)
    desc.to_csv(RESULTS / "descriptives.csv", index=False, float_format="%.3f")

    prev = prevalence(df, des, analytic)
    fasting = df["fasting_sample"] == 1
    undx_fpg = np.where(df["diabetes_fpg"].notna(), ((df["diabetes_fpg"] == 1) & (df["DIQ010"] != 1)), np.nan)
    des_fast = Design(df, W_FAST)
    for name, y in [("Total diabetes incl. FPG", df["diabetes_fpg"]), ("Undiagnosed incl. FPG", undx_fpg)]:
        est, _, lo, hi = des_fast.prop_ci(y, fasting.to_numpy())
        prev.loc[len(prev)] = {"outcome": name, "variable": "Overall", "level": "Fasting subsample",
                               "n": int(fasting.sum()), "prev": est, "lower": lo, "upper": hi}
    prev.to_csv(RESULTS / "prevalence.csv", index=False, float_format="%.4f")

    main_fit = fit(df, W_MEC, analytic)
    main_fit.to_csv(RESULTS / "or_python.csv", index=False, float_format="%.12g")

    # Sensitivity analyses
    levels_miss = {**LEVELS, "pir_cat": LEVELS["pir_cat"] + ["Missing"]}
    df_miss = df.assign(pir_cat=df["pir_cat"].astype("object").fillna("Missing"))

    models = {
        "Main (survey-weighted)": main_fit,
        "Unweighted (ignores design)": unweighted(df, analytic),
        "Income missing as category": fit(df_miss, W_MEC, analytic, levels_miss),
        "Diabetes incl. FPG >=126 (fasting subsample)": fit(df, W_FAST, fasting, outcome="diabetes_fpg"),
    }
    sens = pd.concat([m.assign(model=name) for name, m in models.items()])
    sens = sens[sens["term"] != "(Intercept)"]
    sens[["model", "term", "OR", "lower", "upper", "p"]].to_csv(
        RESULTS / "sensitivity.csv", index=False, float_format="%.4f")

    for _, r in prev[prev["variable"] == "Overall"].iterrows():
        print(f"{r.outcome:<26} {100 * r.prev:5.1f}% ({100 * r.lower:.1f}-{100 * r.upper:.1f})  n={r.n:,}")
    print(f"\nMain model: n = {main_fit.attrs['n']:,}, design df = {main_fit.attrs['df']}")
    show = main_fit[main_fit["term"] != "(Intercept)"]
    for _, r in show.iterrows():
        print(f"  {r.term:<22} OR {r.OR:5.2f} ({r.lower:.2f}-{r.upper:.2f})  p={r.p:.3g}")

    print("\nPhysical activity across sensitivity analyses:")
    pa = sens[sens["term"].str.startswith("pa_cat")]
    print(pa.pivot(index="model", columns="term", values="OR").round(2).to_string())


if __name__ == "__main__":
    main()
