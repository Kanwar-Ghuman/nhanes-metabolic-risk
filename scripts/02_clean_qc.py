"""
Step 2: data quality control and derived variables.

Cleaning rules are applied in a fixed order and every step is logged with
counts before and after. Outputs:
    data/processed/nhanes_analytic.csv   all participants, cleaned + derived variables, sample flags
    results/qc_report.md
"""
from datetime import date

import numpy as np
import pandas as pd

from nhanes import ANALYTIC_CSV, COMPONENTS, MERGED_CSV, RAW, RESULTS, W_FAST

YES_NO = ["DIQ010", "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]
DAYS = ["PAQ610", "PAQ625", "PAQ640", "PAQ655", "PAQ670"]
MINUTES = ["PAD615", "PAD630", "PAD645", "PAD660", "PAD675", "PAD680"]
SBP = ["BPXOSY1", "BPXOSY2", "BPXOSY3"]
DBP = ["BPXODI1", "BPXODI2", "BPXODI3"]
DIET = ["DR1TKCAL", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TSFAT", "DR1TSODI"]

PLAUSIBLE = {
    "BMXHT": (100, 230),
    "BMXWT": (20, 300),
    "BMXBMI": (12, 100),
    "BMXWAIST": (40, 200),
    "LBXGH": (3, 20),
    "LBXGLU": (40, 600),
    "INDFMPIR": (0, 5),
    **{c: (60, 270) for c in SBP},
    **{c: (30, 150) for c in DBP},
    **{c: (1, 7) for c in DAYS},
    **{c: (0, 960) for c in MINUTES[:-1]},
    "PAD680": (0, 1440),
}

# Moderate-equivalent minutes: vigorous minutes count double (2018 PA Guidelines)
PA_DOMAINS = {
    "vig_work": ("PAQ605", "PAQ610", "PAD615", 2),
    "mod_work": ("PAQ620", "PAQ625", "PAD630", 1),
    "transport": ("PAQ635", "PAQ640", "PAD645", 1),
    "vig_rec": ("PAQ650", "PAQ655", "PAD660", 2),
    "mod_rec": ("PAQ665", "PAQ670", "PAD675", 1),
}

KEY_VARS = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST", "sbp", "dbp", "LBXGH",
            "mvpa_min_wk", "sed_hr", "DR1TKCAL"]


class QCLog:
    def __init__(self):
        self.rows = {}

    def add(self, section, **row):
        self.rows.setdefault(section, []).append(row)

    def frame(self, section):
        return pd.DataFrame(self.rows.get(section, []))


def set_missing(df, cols, mask, log, step, rule, group=None):
    if group:
        has_data = df[cols].notna().any(axis=1)
        before, n = int(has_data.sum()), int((mask & has_data).sum())
        df.loc[mask, cols] = np.nan
        log.add("cleaning", step=step, variable=group, rule=rule,
                non_missing_before=before, set_to_missing=n, non_missing_after=before - n)
        return
    for col in cols:
        before = int(df[col].notna().sum())
        hit = mask(df[col]) if callable(mask) else mask
        hit = hit & df[col].notna()
        n = int(hit.sum())
        df.loc[hit, col] = np.nan
        log.add("cleaning", step=step, variable=col, rule=rule,
                non_missing_before=before, set_to_missing=n, non_missing_after=before - n)


def file_checks(merged, log):
    demo_ids = set(merged["SEQN"])
    for name, cols in COMPONENTS.items():
        raw = pd.read_sas(RAW / f"{name}.xpt", format="xport", encoding="latin-1")
        ids = raw["SEQN"].astype(int)
        log.add("files", file=name, rows=len(raw), unique_SEQN=ids.nunique(),
                duplicate_SEQN=int(ids.duplicated().sum()),
                not_in_DEMO=int((~ids.isin(demo_ids)).sum()),
                fully_empty_rows=int(raw[cols].isna().all(axis=1).sum()))

    log.add("duplicates", check="Duplicate SEQN in merged file", count=int(merged["SEQN"].duplicated().sum()))
    dup = merged.drop(columns="SEQN").duplicated(keep=False)
    log.add("duplicates", check="Records identical on every column except SEQN", count=int(dup.sum()))
    log.add("duplicates", check="...of which adults (age >= 20)",
            count=int((dup & (merged["RIDAGEYR"] >= 20)).sum()))


def clean(df, log):
    num = df.select_dtypes("number").columns.drop("SEQN")
    tiny = (df[num].abs() < 1e-10) & (df[num] != 0)
    log.add("artifact_detail", values_recoded=int(tiny.sum().sum()),
            columns=", ".join(tiny.columns[tiny.any()]))
    df[num] = df[num].mask(tiny, 0.0)

    set_missing(df, YES_NO, lambda s: s.isin([7, 9]), log, 1, "Refused / don't know (7, 9)")
    set_missing(df, DAYS, lambda s: s.isin([77, 99]), log, 1, "Refused / don't know (77, 99)")
    set_missing(df, MINUTES, lambda s: s.isin([7777, 9999]), log, 1, "Refused / don't know (7777, 9999)")

    adult = df["RIDAGEYR"] >= 20
    for col, (lo, hi) in PLAUSIBLE.items():
        set_missing(df, [col], lambda s, lo=lo, hi=hi: adult & ((s < lo) | (s > hi)), log, 2,
                    f"Outside plausible adult range [{lo}, {hi}]")

    for s, d in zip(SBP, DBP):
        bad = df[d] >= df[s]
        set_missing(df, [s, d], bad, log, 3, "Diastolic >= systolic in same reading")

    daily = sum(df[mins].where(df[yn] == 1, 0) for yn, _, mins, _ in PA_DOMAINS.values())
    pa_vars = [c for v in PA_DOMAINS.values() for c in v[:3]]
    set_missing(df, pa_vars, daily > 960, log, 4, "Activity across domains > 960 min/day (IPAQ rule)",
                group=f"{len(pa_vars)} PA items (participants)")

    unreliable = df["DR1DRSTZ"] != 1
    set_missing(df, DIET, unreliable, log, 5, "Day-1 recall not complete/reliable (DR1DRSTZ != 1)",
                group="Diet totals (participants)")

    kcal, male = df["DR1TKCAL"], df["RIAGENDR"] == 1
    implausible = (male & ((kcal < 800) | (kcal > 4200))) | (~male & ((kcal < 500) | (kcal > 3500)))
    set_missing(df, DIET, implausible, log, 6, "Implausible energy intake (Willett: M 800-4200, F 500-3500 kcal)",
                group="Diet totals (participants)")
    return df


def derive(df):
    df["sex"] = df["RIAGENDR"].map({1: "Male", 2: "Female"})
    df["age_cat"] = pd.cut(df["RIDAGEYR"], [19, 39, 59, np.inf], labels=["20-39", "40-59", "60+"]).astype("object")
    df["race"] = df["RIDRETH3"].map({1: "Hispanic", 2: "Hispanic", 3: "NH White", 4: "NH Black",
                                     6: "NH Asian", 7: "Other/Multi"})
    df["pir_cat"] = pd.cut(df["INDFMPIR"], [-np.inf, 1.3, 3.5, np.inf], right=False,
                           labels=["<1.3", "1.3-3.49", ">=3.5"]).astype("object")
    df["bmi_cat"] = pd.cut(df["BMXBMI"], [-np.inf, 25, 30, 35, np.inf], right=False,
                           labels=["<25", "25-29.9", "30-34.9", ">=35"]).astype("object")

    df["sbp"] = df[SBP].mean(axis=1)
    df["dbp"] = df[DBP].mean(axis=1)

    for name, (yn, days, mins, mult) in PA_DOMAINS.items():
        active = df[days] * df[mins] * mult
        df[f"pa_{name}"] = np.where(df[yn] == 2, 0.0, np.where(df[yn] == 1, active, np.nan))
    pa_cols = [f"pa_{k}" for k in PA_DOMAINS]
    df["mvpa_min_wk"] = df[pa_cols].sum(axis=1, min_count=len(pa_cols))
    df.loc[df[pa_cols].isna().any(axis=1), "mvpa_min_wk"] = np.nan
    m = df["mvpa_min_wk"]
    df["pa_cat"] = np.select([m == 0, m < 150, m >= 150], ["Inactive", "Insufficient", "Active"], None)
    df["sed_hr"] = df["PAD680"] / 60

    # Borderline (DIQ010 == 3) counts as no diagnosis, as in CDC surveillance definitions
    told = df["DIQ010"] == 1
    high_a1c = df["LBXGH"] >= 6.5
    positive = told | high_a1c
    negative = df["DIQ010"].isin([2, 3]) & (df["LBXGH"] < 6.5)
    df["diabetes"] = np.select([positive, negative], [1.0, 0.0], np.nan)
    df["dm_diagnosed"] = np.where(df["diabetes"] == 1, told.astype(float), np.nan)
    df["dm_undiagnosed"] = np.where(df["diabetes"].notna(), (positive & ~told).astype(float), np.nan)

    high_fpg = df["LBXGLU"] >= 126
    positive_fpg = positive | high_fpg
    negative_fpg = negative & (df["LBXGLU"] < 126)
    df["diabetes_fpg"] = np.select([positive_fpg, negative_fpg], [1.0, 0.0], np.nan)
    return df


def sample_flow(df, log):
    steps = [
        ("All 2017-Mar 2020 participants", pd.Series(True, index=df.index)),
        ("Examined at the mobile exam center", df["RIDSTATR"] == 2),
        ("Age >= 20 years", df["RIDAGEYR"] >= 20),
        ("Not pregnant", df["RIDEXPRG"] != 1),
        ("Diabetes status determinable (analytic sample)", df["diabetes"].notna()),
        ("Physical activity available", df["pa_cat"].notna()),
        ("BMI available", df["bmi_cat"].notna()),
        ("Income-to-poverty ratio available (model sample)", df["pir_cat"].notna()),
    ]
    keep = pd.Series(True, index=df.index)
    for i, (label, cond) in enumerate(steps):
        before = int(keep.sum())
        keep &= cond.fillna(False)
        log.add("flow", step=i, criterion=label, excluded=before - int(keep.sum()) if i else 0,
                remaining=int(keep.sum()))
        if label.startswith("Diabetes"):
            df["analytic"] = keep.astype(int)
    df["model_complete"] = keep.astype(int)
    df["fasting_sample"] = ((df["analytic"] == 1) & (df[W_FAST] > 0) & df["diabetes_fpg"].notna()).astype(int)
    log.add("flow", step="-", criterion="Sensitivity: analytic + fasting subsample with valid FPG",
            excluded="", remaining=int(df["fasting_sample"].sum()))
    return df


def outlier_screen(df, log):
    sub = df[df["analytic"] == 1]
    for col in KEY_VARS:
        x = sub[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        log.add("outliers", variable=col, n=len(x), min=round(x.min(), 1), q1=round(q1, 1),
                median=round(x.median(), 1), q3=round(q3, 1), max=round(x.max(), 1),
                below_fence=int((x < lo).sum()), above_fence=int((x > hi).sum()))


def missingness(df, log):
    sub = df[df["analytic"] == 1]
    for col in KEY_VARS + ["pa_cat", "pir_cat", "bmi_cat", "diabetes_fpg"]:
        n_miss = int(sub[col].isna().sum())
        log.add("missing", variable=col, missing=n_miss, pct=round(100 * n_miss / len(sub), 1))


def md_table(frame):
    if frame.empty:
        return "_none_\n"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def write_report(log, path):
    art = log.frame("artifact_detail").iloc[0]
    cleaning = log.frame("cleaning")
    changed = cleaning[cleaning["set_to_missing"] > 0]

    text = f"""# Data quality control report

NHANES 2017–March 2020 pre-pandemic files. Generated {date.today():%Y-%m-%d} by `scripts/02_clean_qc.py`.

## 1. Source files

{md_table(log.frame("files"))}
`fully_empty_rows` = rows where every analysis variable from that file is missing (e.g. an exam component that was not completed).

## 2. Duplicates

{md_table(log.frame("duplicates"))}
The non-unique records are pairs of children in the same household with the same age, sex and household-level
weights (almost certainly twins who were interviewed but not examined). They are distinct participants and are kept.

## 3. Value cleaning

Rules are applied in order. Only rules that changed at least one value are shown; the full log is in `results/qc_cleaning_log.csv`.

Before any rule is applied, {art['values_recoded']:,} near-zero values are recoded to exactly 0 (pandas reads SAS zeros as 5.4e-79). Affected: {art['columns'] or 'none'}.

{md_table(changed)}
## 4. Outlier screen (analytic sample)

Values beyond Q1 − 3·IQR or Q3 + 3·IQR. These are **flagged, not removed**: after the plausibility rules above they are
extreme but physiologically possible, and dropping them would bias prevalence toward the healthy middle. Models use
categorized exposures, which limits their influence.

{md_table(log.frame("outliers"))}
## 5. Participant flow

{md_table(log.frame("flow"))}
## 6. Missingness in the analytic sample

{md_table(log.frame("missing"))}
"""
    path.write_text(text)


def main():
    merged = pd.read_csv(MERGED_CSV)
    log = QCLog()

    file_checks(merged, log)
    df = clean(merged.copy(), log)
    df = derive(df)
    df = sample_flow(df, log)
    outlier_screen(df, log)
    missingness(df, log)

    df.to_csv(ANALYTIC_CSV, index=False)
    log.frame("cleaning").to_csv(RESULTS / "qc_cleaning_log.csv", index=False)
    write_report(log, RESULTS / "qc_report.md")

    print(log.frame("flow").to_string(index=False))
    print(f"\nValues set to missing: {int(log.frame('cleaning')['set_to_missing'].sum()):,}")
    print("Wrote results/qc_report.md and data/processed/nhanes_analytic.csv")


if __name__ == "__main__":
    main()
