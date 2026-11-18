"""Shared paths, variable definitions and plot style for the NHANES 2017–March 2020 analysis."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _d in (RAW, PROCESSED, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

MERGED_CSV = PROCESSED / "nhanes_merged.csv"
ANALYTIC_CSV = PROCESSED / "nhanes_analytic.csv"

# Pre-pandemic files (P_ prefix). Columns not listed are dropped at merge time.
COMPONENTS = {
    "P_DEMO": ["RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "RIDEXPRG", "INDFMPIR",
               "WTINTPRP", "WTMECPRP", "SDMVPSU", "SDMVSTRA"],
    "P_BMX": ["BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"],
    "P_BPXO": ["BPXOSY1", "BPXODI1", "BPXOSY2", "BPXODI2", "BPXOSY3", "BPXODI3"],
    "P_GHB": ["LBXGH"],
    "P_GLU": ["WTSAFPRP", "LBXGLU"],
    "P_DIQ": ["DIQ010"],
    "P_PAQ": ["PAQ605", "PAQ610", "PAD615", "PAQ620", "PAQ625", "PAD630", "PAQ635",
              "PAQ640", "PAD645", "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670",
              "PAD675", "PAD680"],
    "P_DR1TOT": ["WTDRD1PP", "DR1DRSTZ", "DR1TKCAL", "DR1TCARB", "DR1TSUGR", "DR1TFIBE",
                 "DR1TSFAT", "DR1TSODI"],
}

STRATA, PSU = "SDMVSTRA", "SDMVPSU"
W_MEC, W_FAST, W_DIET = "WTMECPRP", "WTSAFPRP", "WTDRD1PP"

# First level of each list is the regression reference category.
LEVELS = {
    "pa_cat": ["Active", "Insufficient", "Inactive"],
    "bmi_cat": ["<25", "25-29.9", "30-34.9", ">=35"],
    "age_cat": ["20-39", "40-59", "60+"],
    "sex": ["Male", "Female"],
    "pir_cat": [">=3.5", "1.3-3.49", "<1.3"],
    "race": ["NH White", "NH Black", "Hispanic", "NH Asian", "Other/Multi"],
}

LABELS = {
    "pa_cat": "Physical activity",
    "bmi_cat": "BMI (kg/m²)",
    "age_cat": "Age (years)",
    "sex": "Sex",
    "pir_cat": "Income-to-poverty ratio",
    "race": "Race/ethnicity",
}

PA_LABELS = {
    "Active": "Active (≥150 min/wk)",
    "Insufficient": "Insufficient (1–149)",
    "Inactive": "Inactive (0)",
}

MODEL_VARS = ["pa_cat", "bmi_cat", "age_cat", "sex", "pir_cat", "race"]

COLORS = {"no": "#4C72B0", "yes": "#C44E52", "lr": "#4C72B0", "rf": "#DD8452", "nn": "#55A868", "grey": "#8C8C8C"}


def load_analytic():
    df = pd.read_csv(ANALYTIC_CSV)
    for var, levels in LEVELS.items():
        df[var] = pd.Categorical(df[var], categories=levels)
    return df


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
    })
