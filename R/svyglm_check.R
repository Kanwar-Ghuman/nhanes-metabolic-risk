# Step 5: refit the main model with the survey package and compare with the
# Python implementation (scripts/04_analysis.py -> results/or_python.csv).
#
#   Rscript R/svyglm_check.R

suppressPackageStartupMessages(library(survey))

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(script)) normalizePath(file.path(dirname(script), "..")) else getwd()

nh <- read.csv(file.path(root, "data/processed/nhanes_analytic.csv"))

lv <- list(
  pa_cat  = c("Active", "Insufficient", "Inactive"),
  bmi_cat = c("<25", "25-29.9", "30-34.9", ">=35"),
  age_cat = c("20-39", "40-59", "60+"),
  sex     = c("Male", "Female"),
  pir_cat = c(">=3.5", "1.3-3.49", "<1.3"),
  race    = c("NH White", "NH Black", "Hispanic", "NH Asian", "Other/Multi")
)
for (v in names(lv)) nh[[v]] <- factor(nh[[v]], levels = lv[[v]])

# Build the design on everyone with an exam weight, then subset -- never subset first.
mec <- subset(nh, !is.na(WTMECPRP) & WTMECPRP > 0)
des <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMECPRP, nest = TRUE, data = mec)
adults <- subset(des, analytic == 1)

fit <- svyglm(diabetes ~ pa_cat + bmi_cat + age_cat + sex + pir_cat + race,
              design = adults, family = quasibinomial())

ddf <- degf(fit$survey.design)
b <- coef(fit)
se <- SE(fit)
tq <- qt(0.975, ddf)

r_res <- data.frame(term = names(b), beta = b, se = se, OR = exp(b),
                    lower = exp(b - tq * se), upper = exp(b + tq * se),
                    p = 2 * pt(-abs(b / se), ddf), row.names = NULL)
write.csv(r_res, file.path(root, "results/or_r.csv"), row.names = FALSE)

py <- read.csv(file.path(root, "results/or_python.csv"))
py <- py[match(r_res$term, py$term), ]
stopifnot(identical(py$term, r_res$term))

cmp <- data.frame(term = r_res$term,
                  OR_python = py$OR, OR_R = r_res$OR,
                  se_python = py$se, se_R = r_res$se,
                  abs_diff_beta = abs(py$beta - r_res$beta),
                  abs_diff_se = abs(py$se - r_res$se))
write.csv(cmp, file.path(root, "results/python_vs_r.csv"), row.names = FALSE)

cat(sprintf("n = %d, design df = %d\n\n", nobs(fit), ddf))
print(format(cmp[, 1:5], digits = 4), row.names = FALSE)
cat(sprintf("\nMax |difference| in log-odds:  %.2e\n", max(cmp$abs_diff_beta)))
cat(sprintf("Max |difference| in SE:        %.2e\n", max(cmp$abs_diff_se)))

# Spot-check the descriptive estimates behind Table 1 and the prevalence figure
prev_py <- read.csv(file.path(root, "results/prevalence.csv"))
prev_r <- svyciprop(~I(diabetes == 1), adults, method = "logit")
bmi_r <- svyby(~BMXBMI, ~diabetes, adults, svymean, na.rm = TRUE)
cat(sprintf("\nDiabetes prevalence  R: %.4f   Python: %.4f\n", coef(prev_r), prev_py$prev[1]))
cat(sprintf("Mean BMI (no DM / DM)  R: %.2f / %.2f\n", bmi_r$BMXBMI[1], bmi_r$BMXBMI[2]))

tol <- 1e-6
if (max(cmp$abs_diff_beta) < tol && max(cmp$abs_diff_se) < tol) {
  cat("\nPython and R estimates agree to within", tol, "\n")
} else {
  warning("Python and R estimates differ by more than ", tol)
}
