"""
Step 3: Table 1 -- participant characteristics by diabetes status.

Counts are unweighted; percentages, means and SEs are survey-weighted (MEC exam
weights) and represent the U.S. non-institutionalized adult population.
p-values are design-based: Wald t-test for continuous variables, adjusted Wald
F-test for categorical variables.
"""
import numpy as np
import pandas as pd

from nhanes import LABELS, LEVELS, PA_LABELS, RESULTS, W_MEC, load_analytic
from svy import Design


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build(df):
    des = Design(df, W_MEC)
    analytic = (df["analytic"] == 1).to_numpy()
    dm = df["diabetes"].to_numpy()
    groups = {
        "Overall": analytic,
        "No diabetes": analytic & (dm == 0),
        "Diabetes": analytic & (dm == 1),
    }
    rows = []

    def header(label, p=""):
        rows.append({"Characteristic": label, **{g: "" for g in groups}, "p": p, "level": False})

    rows.append({"Characteristic": "Participants, n", **{g: f"{dom.sum():,}" for g, dom in groups.items()},
                 "p": "", "level": False})
    rows.append({"Characteristic": "Weighted population, millions",
                 **{g: f"{des.w[dom].sum() / 1e6:.1f}" for g, dom in groups.items()}, "p": "", "level": False})

    def continuous(label, col, digits=1):
        vals = {}
        for g, dom in groups.items():
            est, se = des.mean(df[col], dom)
            vals[g] = f"{est:.{digits}f} ({se:.{digits}f})"
        p = des.diff_test(df[col], dm, analytic)
        rows.append({"Characteristic": f"{label}, mean (SE)", **vals, "p": fmt_p(p), "level": False})

    def categorical(label, col, levels, names=None):
        header(f"{label}, n (%)", fmt_p(des.chisq_test(df[col], dm, analytic)))
        known = df[col].notna()
        for lv in levels:
            is_lv = (df[col] == lv).to_numpy()
            ind = np.where(known, is_lv, np.nan)
            vals = {}
            for g, dom in groups.items():
                pct, _ = des.mean(ind, dom)
                vals[g] = f"{(is_lv & dom).sum():,} ({100 * pct:.1f})"
            rows.append({"Characteristic": (names or {}).get(lv, lv), **vals, "p": "", "level": True})
        n_miss = {g: int((~known.to_numpy() & dom).sum()) for g, dom in groups.items()}
        if any(n_miss.values()):
            rows.append({"Characteristic": "Missing", **{g: f"{n:,}" for g, n in n_miss.items()},
                         "p": "", "level": True})

    continuous("Age, years", "RIDAGEYR")
    categorical(LABELS["age_cat"], "age_cat", LEVELS["age_cat"])
    categorical(LABELS["sex"], "sex", ["Female", "Male"])
    categorical(LABELS["race"], "race", ["NH White", "NH Black", "Hispanic", "NH Asian", "Other/Multi"],
                {"NH White": "Non-Hispanic White", "NH Black": "Non-Hispanic Black",
                 "NH Asian": "Non-Hispanic Asian", "Other/Multi": "Other / multiracial"})
    categorical(LABELS["pir_cat"], "pir_cat", ["<1.3", "1.3-3.49", ">=3.5"])
    continuous("BMI, kg/m²", "BMXBMI")
    categorical(LABELS["bmi_cat"], "bmi_cat", LEVELS["bmi_cat"])
    continuous("Waist circumference, cm", "BMXWAIST")
    categorical("Physical activity", "pa_cat", ["Active", "Insufficient", "Inactive"], PA_LABELS)
    continuous("Sedentary time, h/day", "sed_hr")
    continuous("Systolic BP, mmHg", "sbp")
    continuous("HbA1c, %", "LBXGH", 2)

    est, se = des.mean(df["dm_undiagnosed"], groups["Diabetes"])
    rows.append({"Characteristic": "Undiagnosed (HbA1c ≥6.5%, no prior diagnosis), % (SE)",
                 "Overall": "", "No diabetes": "", "Diabetes": f"{100 * est:.1f} ({100 * se:.1f})",
                 "p": "", "level": False})
    return pd.DataFrame(rows)


def to_markdown(table):
    cols = ["Characteristic", "Overall", "No diabetes", "Diabetes", "p"]
    lines = ["| " + " | ".join(cols[:-1] + ["p-value"]) + " |", "|:---|---:|---:|---:|---:|"]
    for _, r in table.iterrows():
        name = f"&nbsp;&nbsp;&nbsp;{r['Characteristic']}" if r["level"] else r["Characteristic"]
        lines.append("| " + " | ".join([name] + [str(r[c]) for c in cols[1:]]) + " |")
    return "\n".join(lines) + "\n"


def main():
    df = load_analytic()
    table = build(df)
    table.drop(columns="level").to_csv(RESULTS / "table1.csv", index=False)
    note = ("\nCounts are unweighted; percentages, means and standard errors are weighted with MEC exam weights "
            "(WTMECPRP) and account for strata and PSUs. Percentages exclude missing values.\n")
    (RESULTS / "table1.md").write_text(to_markdown(table) + note)
    print(table.drop(columns="level").to_string(index=False))


if __name__ == "__main__":
    main()
