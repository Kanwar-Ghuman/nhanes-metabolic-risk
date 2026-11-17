"""
Step 7: can non-laboratory measures identify adults with diabetes, and does
performance hold across demographic groups?

Logistic regression vs. random forest on a 70/30 stratified split. HbA1c and
glucose are excluded because they define the outcome. Models are trained
unweighted; test-set AUC is reported both unweighted and survey-weighted.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nhanes import COLORS, FIGURES, RESULTS, W_MEC, load_analytic, set_style

SEED = 2020
N_BOOT = 1000
TARGET_SENSITIVITY = 0.80

NUMERIC = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST", "sbp", "dbp",
           "mvpa_min_wk", "sed_hr", "DR1TKCAL", "DR1TSUGR", "DR1TFIBE"]
CATEGORICAL = ["sex", "race"]
FEATURE_NAMES = {
    "RIDAGEYR": "Age", "INDFMPIR": "Income-to-poverty ratio", "BMXBMI": "BMI",
    "BMXWAIST": "Waist circumference", "sbp": "Systolic BP", "dbp": "Diastolic BP",
    "mvpa_min_wk": "MVPA", "sed_hr": "Sedentary time", "DR1TKCAL": "Energy intake",
    "DR1TSUGR": "Total sugars", "DR1TFIBE": "Fiber", "sex": "Sex", "race": "Race/ethnicity",
}
FAIRNESS_GROUPS = {
    "Sex": ("sex", ["Male", "Female"]),
    "Race/ethnicity": ("race", ["NH White", "NH Black", "Hispanic", "NH Asian", "Other/Multi"]),
    "Income-to-poverty ratio": ("pir_cat", [">=3.5", "1.3-3.49", "<1.3", "Missing"]),
    "Age (years)": ("age_cat", ["20-39", "40-59", "60+"]),
}


def preprocessor(scale):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])


def build_models(X, y):
    lr = Pipeline([("prep", preprocessor(scale=True)),
                   ("clf", LogisticRegression(max_iter=2000))])
    rf = Pipeline([("prep", preprocessor(scale=False)),
                   ("clf", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1))])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    grid = {"clf__min_samples_leaf": [5, 10, 20, 40], "clf__max_features": ["sqrt", 0.5]}
    search = GridSearchCV(rf, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y)
    print(f"RF tuning: best {search.best_params_}, CV AUC {search.best_score_:.3f}")
    lr_cv = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    rf_cv = cross_val_predict(search.best_estimator_, X, y, cv=cv, method="predict_proba")[:, 1]
    print(f"LR CV AUC {roc_auc_score(y, lr_cv):.3f}")
    models = {"Logistic regression": lr.fit(X, y), "Random forest": search.best_estimator_}
    # Threshold chosen on out-of-fold training predictions, never on the test set
    thresholds = {name: np.quantile(p[y == 1], 1 - TARGET_SENSITIVITY)
                  for name, p in [("Logistic regression", lr_cv), ("Random forest", rf_cv)]}
    return models, thresholds


def boot_idx(y, rng, n=N_BOOT):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def performance(y, w, preds, rng):
    rows = []
    idx = list(boot_idx(y, rng))
    boots = {name: np.array([roc_auc_score(y[i], p[i]) for i in idx]) for name, p in preds.items()}
    for name, p in preds.items():
        lo, hi = np.percentile(boots[name], [2.5, 97.5])
        rows.append({"model": name, "auc": roc_auc_score(y, p), "auc_lower": lo, "auc_upper": hi,
                     "auc_weighted": roc_auc_score(y, p, sample_weight=w), "brier": brier_score_loss(y, p)})
    diff = boots["Random forest"] - boots["Logistic regression"]
    point = roc_auc_score(y, preds["Random forest"]) - roc_auc_score(y, preds["Logistic regression"])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return pd.DataFrame(rows), (point, lo, hi)


def fairness(test, y, preds, thresholds, rng):
    rows = []
    for label, (col, levels) in FAIRNESS_GROUPS.items():
        values = test[col].astype("object").fillna("Missing").to_numpy()
        for lv in levels:
            m = values == lv
            yg = y[m]
            for name, p in preds.items():
                pg = p[m]
                flag = pg >= thresholds[name]
                aucs = [roc_auc_score(yg[i], pg[i]) for i in boot_idx(yg, rng, 500)]
                rows.append({
                    "group": label, "level": lv, "model": name, "n": int(m.sum()), "cases": int(yg.sum()),
                    "auc": roc_auc_score(yg, pg),
                    "auc_lower": np.percentile(aucs, 2.5), "auc_upper": np.percentile(aucs, 97.5),
                    "sensitivity": flag[yg == 1].mean(), "fpr": flag[yg == 0].mean(),
                    "ppv": yg[flag].mean() if flag.any() else np.nan,
                    "observed": yg.mean(), "predicted": pg.mean(),
                })
    return pd.DataFrame(rows)


def plot_roc_calibration(y, preds, perf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    colors = {"Logistic regression": COLORS["lr"], "Random forest": COLORS["rf"]}
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        r = perf.set_index("model").loc[name]
        ax1.plot(fpr, tpr, color=colors[name], lw=1.8,
                 label=f"{name}  AUC {r.auc:.3f} ({r.auc_lower:.3f}–{r.auc_upper:.3f})")
        bins = pd.qcut(p, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin").mean()
        ax2.plot(cal["p"], cal["y"], "o-", color=colors[name], ms=4, label=name)
    ax1.plot([0, 1], [0, 1], color=COLORS["grey"], ls=":", lw=1)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (held-out test set)")
    ax1.legend(loc="lower right", fontsize=8)
    lim = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([0, lim], [0, lim], color=COLORS["grey"], ls=":", lw=1)
    ax2.set(xlabel="Mean predicted risk (decile)", ylabel="Observed prevalence", title="Calibration")
    ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_roc_calibration.png")
    plt.close(fig)


def plot_fairness(fair):
    names = {">=3.5": "≥3.5", "1.3-3.49": "1.3–3.49", "NH White": "NH White", "NH Black": "NH Black",
             "NH Asian": "NH Asian", "Other/Multi": "Other/multi", "20-39": "20–39", "40-59": "40–59", "60+": "≥60"}
    rows, y, headers = [], 0, []
    for label, (_, levels) in FAIRNESS_GROUPS.items():
        headers.append((y, label))
        y -= 1
        for lv in levels:
            rows.append((y, label, lv))
            y -= 1
        y -= 0.4

    fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)
    offsets = {"Logistic regression": 0.15, "Random forest": -0.15}
    colors = {"Logistic regression": COLORS["lr"], "Random forest": COLORS["rf"]}
    f = fair.set_index(["group", "level", "model"])
    for ypos, label, lv in rows:
        for name in offsets:
            r = f.loc[(label, lv, name)]
            yy = ypos + offsets[name]
            axes[0].plot([r.auc_lower, r.auc_upper], [yy, yy], color=colors[name], lw=1.4)
            axes[0].plot(r.auc, yy, "o", color=colors[name], ms=4, label=name)
            axes[1].plot(r.sensitivity, yy, "o", color=colors[name], ms=4)
            axes[1].plot(r.fpr, yy, "x", color=colors[name], ms=5)
            axes[2].plot(r.predicted, yy, "o", color=colors[name], ms=4)
        axes[2].plot(f.loc[(label, lv, "Random forest")].observed, ypos, "|", color="black", ms=11, mew=2)

    tr = axes[0].get_yaxis_transform()
    for ypos, text in headers:
        axes[0].text(-0.42, ypos, text, fontweight="bold", va="center", ha="left", fontsize=9, transform=tr)
    for ypos, _, lv in rows:
        axes[0].text(-0.39, ypos, names.get(lv, lv), va="center", ha="left", fontsize=8.5, transform=tr)
    for ax in axes:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(y, 0.8)

    axes[0].set(xlabel="AUC (95% bootstrap CI)", title="Discrimination")
    axes[1].set(xlabel="Rate", title=f"Sensitivity (●) and FPR (×)\nthreshold set for {TARGET_SENSITIVITY:.0%} sensitivity")
    axes[2].set(xlabel="Diabetes prevalence", title="Calibration-in-the-large\nobserved (|) vs. mean predicted (●)")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Model performance across demographic groups (held-out test set)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "fig5_fairness.png")
    plt.close(fig)


def plot_importance(imp):
    order = imp.groupby("feature")["importance"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ypos = np.arange(len(order))
    for name, off, color in [("Logistic regression", 0.18, COLORS["lr"]), ("Random forest", -0.18, COLORS["rf"])]:
        s = imp[imp["model"] == name].set_index("feature").loc[order]
        ax.barh(ypos + off, s["importance"], height=0.34, xerr=s["sd"], color=color, label=name,
                error_kw={"lw": 0.8, "ecolor": "#555"})
    ax.set_yticks(ypos, [FEATURE_NAMES[f] for f in order])
    ax.set_xlabel("Drop in test AUC when feature is permuted")
    ax.set_title("Permutation importance", loc="left")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES / "fig6_importance.png")
    plt.close(fig)


def main():
    set_style()
    rng = np.random.default_rng(SEED)
    df = load_analytic()
    data = df[df["analytic"] == 1].copy()
    data[CATEGORICAL] = data[CATEGORICAL].astype(str)
    features = NUMERIC + CATEGORICAL

    train, test = train_test_split(data, test_size=0.3, stratify=data["diabetes"], random_state=SEED)
    y_train, y_test = train["diabetes"].astype(int).to_numpy(), test["diabetes"].astype(int).to_numpy()
    print(f"Train n={len(train):,} ({y_train.mean():.1%} diabetes), test n={len(test):,}")

    models, thresholds = build_models(train[features], y_train)
    preds = {name: m.predict_proba(test[features])[:, 1] for name, m in models.items()}

    perf, (d, lo, hi) = performance(y_test, test[W_MEC].to_numpy(), preds, rng)
    perf["threshold"] = perf["model"].map(thresholds)
    perf.to_csv(RESULTS / "ml_performance.csv", index=False, float_format="%.4f")
    print(perf.round(3).to_string(index=False))
    print(f"AUC difference (RF - LR): {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")

    fair = fairness(test, y_test, preds, thresholds, rng)
    fair.to_csv(RESULTS / "ml_fairness.csv", index=False, float_format="%.4f")

    imp_rows = []
    for name, m in models.items():
        res = permutation_importance(m, test[features], y_test, scoring="roc_auc",
                                     n_repeats=20, random_state=SEED, n_jobs=-1)
        for f, mu, sd in zip(features, res.importances_mean, res.importances_std):
            imp_rows.append({"model": name, "feature": f, "importance": mu, "sd": sd})
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(RESULTS / "ml_importance.csv", index=False, float_format="%.4f")

    plot_roc_calibration(y_test, preds, perf)
    plot_fairness(fair)
    plot_importance(imp)

    summary = fair.pivot_table(index=["group", "level"], columns="model", values=["auc", "sensitivity"], sort=False)
    print("\n" + summary.round(3).to_string())


if __name__ == "__main__":
    main()
