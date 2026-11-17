# Data quality control report

NHANES 2017–March 2020 pre-pandemic files. Generated 2026-11-17 by `scripts/02_clean_qc.py`.

## 1. Source files

| file | rows | unique_SEQN | duplicate_SEQN | not_in_DEMO | fully_empty_rows |
|---|---|---|---|---|---|
| P_DEMO | 15,560 | 15,560 | 0 | 0 | 0 |
| P_BMX | 14,300 | 14,300 | 0 | 0 | 203 |
| P_BPXO | 11,656 | 11,656 | 0 | 0 | 1,303 |
| P_GHB | 10,409 | 10,409 | 0 | 0 | 672 |
| P_GLU | 5,090 | 5,090 | 0 | 0 | 0 |
| P_DIQ | 14,986 | 14,986 | 0 | 0 | 0 |
| P_PAQ | 9,693 | 9,693 | 0 | 0 | 0 |
| P_DR1TOT | 14,300 | 14,300 | 0 | 0 | 0 |

`fully_empty_rows` = rows where every analysis variable from that file is missing (e.g. an exam component that was not completed).

## 2. Duplicates

| check | count |
|---|---|
| Duplicate SEQN in merged file | 0 |
| Records identical on every column except SEQN | 12 |
| ...of which adults (age >= 20) | 0 |

The non-unique records are pairs of children in the same household with the same age, sex and household-level
weights (almost certainly twins who were interviewed but not examined). They are distinct participants and are kept.

## 3. Value cleaning

Rules are applied in order. Only rules that changed at least one value are shown; the full log is in `results/qc_cleaning_log.csv`.

Before any rule is applied, 4,405 near-zero values are recoded to exactly 0 (pandas reads SAS zeros as 5.4e-79). Affected: RIDAGEYR, INDFMPIR, WTMECPRP, WTSAFPRP, PAD680, WTDRD1PP, DR1TKCAL, DR1TCARB, DR1TSUGR, DR1TFIBE, DR1TSFAT, DR1TSODI.

| step | variable | rule | non_missing_before | set_to_missing | non_missing_after |
|---|---|---|---|---|---|
| 1 | DIQ010 | Refused / don't know (7, 9) | 14,986 | 8 | 14,978 |
| 1 | PAQ605 | Refused / don't know (7, 9) | 9,693 | 7 | 9,686 |
| 1 | PAQ620 | Refused / don't know (7, 9) | 9,693 | 7 | 9,686 |
| 1 | PAQ635 | Refused / don't know (7, 9) | 9,693 | 2 | 9,691 |
| 1 | PAQ665 | Refused / don't know (7, 9) | 9,693 | 2 | 9,691 |
| 1 | PAQ610 | Refused / don't know (77, 99) | 2,418 | 3 | 2,415 |
| 1 | PAQ625 | Refused / don't know (77, 99) | 4,187 | 10 | 4,177 |
| 1 | PAQ640 | Refused / don't know (77, 99) | 2,253 | 6 | 2,247 |
| 1 | PAQ655 | Refused / don't know (77, 99) | 2,422 | 1 | 2,421 |
| 1 | PAQ670 | Refused / don't know (77, 99) | 3,904 | 3 | 3,901 |
| 1 | PAD615 | Refused / don't know (7777, 9999) | 2,402 | 8 | 2,394 |
| 1 | PAD630 | Refused / don't know (7777, 9999) | 4,164 | 17 | 4,147 |
| 1 | PAD645 | Refused / don't know (7777, 9999) | 2,241 | 5 | 2,236 |
| 1 | PAD660 | Refused / don't know (7777, 9999) | 2,417 | 2 | 2,415 |
| 1 | PAD675 | Refused / don't know (7777, 9999) | 3,893 | 1 | 3,892 |
| 1 | PAD680 | Refused / don't know (7777, 9999) | 9,676 | 66 | 9,610 |
| 2 | LBXGH | Outside plausible adult range [3, 20] | 9,737 | 1 | 9,736 |
| 2 | BPXODI1 | Outside plausible adult range [30, 150] | 10,352 | 1 | 10,351 |
| 4 | 15 PA items (participants) | Activity across domains > 960 min/day (IPAQ rule) | 9,693 | 148 | 9,545 |
| 6 | Diet totals (participants) | Implausible energy intake (Willett: M 800-4200, F 500-3500 kcal) | 12,392 | 901 | 11,491 |

## 4. Outlier screen (analytic sample)

Values beyond Q1 − 3·IQR or Q3 + 3·IQR. These are **flagged, not removed**: after the plausibility rules above they are
extreme but physiologically possible, and dropping them would bias prevalence toward the healthy middle. Models use
categorized exposures, which limits their influence.

| variable | n | min | q1 | median | q3 | max | below_fence | above_fence |
|---|---|---|---|---|---|---|---|---|
| RIDAGEYR | 8,063 | 20.0 | 37.0 | 53.0 | 65.0 | 80.0 | 0 | 0 |
| INDFMPIR | 6,962 | 0.0 | 1.2 | 2.2 | 4.2 | 5.0 | 0 | 0 |
| BMXBMI | 7,925 | 14.2 | 24.9 | 28.8 | 33.8 | 92.3 | 0 | 31 |
| BMXWAIST | 7,669 | 57.9 | 89.2 | 99.6 | 111.5 | 187.5 | 0 | 1 |
| sbp | 7,269 | 76.3 | 111.0 | 122.3 | 135.7 | 218.7 | 0 | 6 |
| dbp | 7,269 | 41.3 | 66.7 | 74.0 | 82.0 | 137.7 | 0 | 3 |
| LBXGH | 7,999 | 3.6 | 5.3 | 5.6 | 6.0 | 16.2 | 0 | 364 |
| mvpa_min_wk | 7,877 | 0.0 | 0.0 | 330.0 | 1220.0 | 11820.0 | 0 | 345 |
| sed_hr | 7,995 | 0.0 | 3.0 | 5.0 | 8.0 | 22.0 | 0 | 0 |
| DR1TKCAL | 6,836 | 503.0 | 1431.8 | 1927.5 | 2526.0 | 4194.0 | 0 | 0 |

## 5. Participant flow

| step | criterion | excluded | remaining |
|---|---|---|---|
| 0 | All 2017-Mar 2020 participants | 0 | 15,560 |
| 1 | Examined at the mobile exam center | 1,260 | 14,300 |
| 2 | Age >= 20 years | 5,756 | 8,544 |
| 3 | Not pregnant | 87 | 8,457 |
| 4 | Diabetes status determinable (analytic sample) | 394 | 8,063 |
| 5 | Physical activity available | 186 | 7,877 |
| 6 | BMI available | 137 | 7,740 |
| 7 | Income-to-poverty ratio available (model sample) | 1,029 | 6,711 |
| - | Sensitivity: analytic + fasting subsample with valid FPG |  | 3,757 |

## 6. Missingness in the analytic sample

| variable | missing | pct |
|---|---|---|
| RIDAGEYR | 0 | 0.0 |
| INDFMPIR | 1,101 | 13.7 |
| BMXBMI | 138 | 1.7 |
| BMXWAIST | 394 | 4.9 |
| sbp | 794 | 9.8 |
| dbp | 794 | 9.8 |
| LBXGH | 64 | 0.8 |
| mvpa_min_wk | 186 | 2.3 |
| sed_hr | 68 | 0.8 |
| DR1TKCAL | 1,227 | 15.2 |
| pa_cat | 186 | 2.3 |
| pir_cat | 1,101 | 13.7 |
| bmi_cat | 138 | 1.7 |
| diabetes_fpg | 3,287 | 40.8 |

