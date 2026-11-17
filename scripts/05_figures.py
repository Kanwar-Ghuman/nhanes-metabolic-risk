"""
Step 6: figures for the descriptive and regression results.

    figures/fig1_distributions.png   weighted distributions by diabetes status
    figures/fig2_prevalence.png      weighted prevalence (95% CI) by subgroup
    figures/fig3_forest.png          adjusted odds ratios from the main model
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nhanes import COLORS, FIGURES, LABELS, LEVELS, MODEL_VARS, PA_LABELS, RESULTS, W_MEC, load_analytic, set_style

LEVEL_NAMES = {
    **PA_LABELS,
    "NH White": "Non-Hispanic White", "NH Black": "Non-Hispanic Black",
    "NH Asian": "Non-Hispanic Asian", "Other/Multi": "Other / multiracial",
    "<25": "<25", "25-29.9": "25–29.9", "30-34.9": "30–34.9", ">=35": "≥35",
    "20-39": "20–39", "40-59": "40–59", "60+": "≥60",
    ">=3.5": "≥3.5", "1.3-3.49": "1.3–3.49", "<1.3": "<1.3",
}

PANELS = [
    ("BMXBMI", "BMI (kg/m²)", np.arange(15, 61, 1.5)),
    ("BMXWAIST", "Waist circumference (cm)", np.arange(60, 161, 3)),
    ("LBXGH", "HbA1c (%)", np.arange(4, 12.1, 0.2)),
    ("sbp", "Systolic blood pressure (mmHg)", np.arange(80, 201, 4)),
    ("mvpa_min_wk", "MVPA (min/week, capped at 3,000)", np.arange(0, 3001, 150)),
    ("sed_hr", "Sedentary time (h/day)", np.arange(-1, 18, 2)),
]


def distributions(df):
    sub = df[df["analytic"] == 1]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (col, label, bins) in zip(axes.flat, PANELS):
        for status, name, color in [(0, "No diabetes", COLORS["no"]), (1, "Diabetes", COLORS["yes"])]:
            g = sub[(sub["diabetes"] == status) & sub[col].notna()]
            x = g[col].clip(bins[0], bins[-1] - 1e-9)
            ax.hist(x, bins=bins, weights=g[W_MEC] / g[W_MEC].sum() * 100, histtype="stepfilled",
                    alpha=0.35, color=color, label=name)
            ax.hist(x, bins=bins, weights=g[W_MEC] / g[W_MEC].sum() * 100, histtype="step",
                    lw=1.2, color=color)
        ax.set_xlabel(label)
        ax.set_ylabel("Weighted %")
    axes[0, 2].axvline(6.5, color="black", lw=0.8, ls="--")
    axes[0, 2].text(6.6, axes[0, 2].get_ylim()[1] * 0.9, "6.5% cutoff", fontsize=8)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Distribution of key variables by diabetes status (U.S. adults, NHANES 2017–March 2020)",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_distributions.png")
    plt.close(fig)


def _grouped_rows(frame, var_col, level_col):
    """Lay out rows top-to-bottom with a header line before each variable group."""
    ys, labels, headers, y = [], [], [], 0
    for var in MODEL_VARS:
        rows = frame[frame[var_col] == var]
        headers.append((y, LABELS[var]))
        y -= 1
        for _, r in rows.iterrows():
            ys.append((y, r))
            labels.append((y, LEVEL_NAMES.get(r[level_col], r[level_col])))
            y -= 1
        y -= 0.4
    return ys, labels, headers, y


def _draw_row_labels(ax, labels, headers, x=-0.42):
    tr = ax.get_yaxis_transform()
    for y, text in headers:
        ax.text(x, y, text, fontweight="bold", va="center", ha="left", fontsize=9, transform=tr)
    for y, text in labels:
        ax.text(x + 0.03, y, text, va="center", ha="left", fontsize=8.5, transform=tr)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def prevalence(prev):
    overall = prev.iloc[0]
    groups = prev[prev["variable"] != "Overall"].copy()
    key = {v: k for k, v in LABELS.items()}
    groups["var"] = groups["variable"].map(key)

    ys, labels, headers, ymin = _grouped_rows(groups, "var", "level")
    fig, ax = plt.subplots(figsize=(7, 8.2))
    ax.axvline(overall.prev * 100, color=COLORS["grey"], ls="--", lw=0.9)
    ax.text(overall.prev * 100 + 0.4, 0.4, f"All adults {overall.prev * 100:.1f}%", fontsize=8, color="#555")
    for y, r in ys:
        ax.plot([r.lower * 100, r.upper * 100], [y, y], color=COLORS["yes"], lw=1.6)
        ax.plot(r.prev * 100, y, "o", color=COLORS["yes"], ms=5)
        ax.text(r.upper * 100 + 0.6, y, f"{r.prev * 100:.1f}", va="center", fontsize=7.5, color="#444")
    _draw_row_labels(ax, labels, headers)
    ax.set_ylim(ymin, 1)
    ax.set_xlim(0, None)
    ax.set_xlabel("Weighted prevalence of diabetes, % (95% CI)")
    ax.set_title("Diabetes prevalence by subgroup", loc="left")
    fig.savefig(FIGURES / "fig2_prevalence.png")
    plt.close(fig)


def forest(ors):
    ors = ors.set_index("term")
    rows = []
    for var in MODEL_VARS:
        for i, lv in enumerate(LEVELS[var]):
            term = f"{var}{lv}"
            if i == 0:
                rows.append({"var": var, "level": lv, "OR": 1.0, "lower": np.nan, "upper": np.nan, "ref": True})
            else:
                r = ors.loc[term]
                rows.append({"var": var, "level": lv, "OR": r.OR, "lower": r.lower, "upper": r.upper, "ref": False})
    frame = pd.DataFrame(rows)

    ys, labels, headers, ymin = _grouped_rows(frame, "var", "level")
    fig, ax = plt.subplots(figsize=(8, 8.2))
    ax.axvline(1, color="black", lw=0.8)
    for y, r in ys:
        if r.ref:
            ax.plot(1, y, "D", color=COLORS["grey"], ms=4)
            text = "Reference"
        else:
            ax.plot([r.lower, r.upper], [y, y], color=COLORS["no"], lw=1.6)
            ax.plot(r.OR, y, "s", color=COLORS["no"], ms=5)
            text = f"{r.OR:.2f} ({r.lower:.2f}–{r.upper:.2f})"
        ax.text(1.02, y, text, va="center", fontsize=8, transform=ax.get_yaxis_transform())
    ax.text(1.02, 0.6, "OR (95% CI)", fontweight="bold", fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
    ax.minorticks_off()
    _draw_row_labels(ax, labels, headers, x=-0.45)
    ax.set_ylim(ymin, 1.2)
    ax.set_xlabel("Adjusted odds ratio for diabetes (log scale)")
    ax.set_title("Adjusted odds ratios for diabetes (survey-weighted logistic regression)", loc="left")
    fig.savefig(FIGURES / "fig3_forest.png")
    plt.close(fig)


def main():
    set_style()
    df = load_analytic()
    distributions(df)
    prevalence(pd.read_csv(RESULTS / "prevalence.csv"))
    forest(pd.read_csv(RESULTS / "or_python.csv"))
    print("Wrote figures/fig1_distributions.png, fig2_prevalence.png, fig3_forest.png")


if __name__ == "__main__":
    main()
