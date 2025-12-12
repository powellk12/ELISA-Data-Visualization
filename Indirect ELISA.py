import pandas as pd
import matplotlib.pyplot as plt
import numpy as np     
import os
from scipy.optimize import curve_fit
print(os.getcwd())

# Load data
df = pd.read_csv("Indirect ELISA Results - Sheet1.csv")
print(df)

# Row 0 contains ng/mL values + "coating buffer control" in last column
conc_row = df.iloc[0, 2:]  # skip the first two columns ("row" label & description)

# Convert to numeric; non-numeric (like 'coating buffer control') -> NaN
concs = pd.to_numeric(conc_row, errors="coerce")

# Keep only columns that have a numeric concentration
valid_cols = concs.dropna().index          # column names like '1','2',..., '11'
concs = concs.dropna().values.astype(float)  # numpy array of concentrations

# -------------------------
# 3. Extract triplicates for 100 ng/mL and 10 ng/mL
# -------------------------
# Rows:
#   1: A (100 ng/mL)
#   2: B (100 ng/mL)
#   3: C (100 ng/mL)
#   4: D (10 ng/mL)
#   5: E (10 ng/mL)
#   6: F (10 ng/mL)
#
# We'll ignore row G (unknown) and H (neg control) for the curve.

# High antibody: rows A–C (indices 1–3)
high_rows = df.iloc[1:4]   # rows 1,2,3
# Low antibody: rows D–F (indices 4–6)
low_rows  = df.iloc[4:7]   # rows 4,5,6

# Grab only the columns with numeric concentrations
high_vals = high_rows[valid_cols].apply(pd.to_numeric, errors="coerce")
low_vals  = low_rows[valid_cols].apply(pd.to_numeric, errors="coerce")

# Now high_vals and low_vals are 3 x N matrices (3 replicates per concentration)

# -------------------------
# 4. Background (negative control) calculation
# -------------------------
# Row H (index 8) is coating buffer control across columns.
# Column '12' is also coating buffer control across rows.
# We'll take ALL of those numeric values as background replicates.

# Row H (index 8), all OD columns (2..end)
neg_row_vals = pd.to_numeric(df.iloc[8, 2:], errors="coerce")

# Column 12 (header "12"), rows A–H (indices 1..8)
if "12" in df.columns:
    neg_col_vals = pd.to_numeric(df["12"].iloc[1:9], errors="coerce")
    background_all = pd.concat([neg_row_vals, neg_col_vals], ignore_index=True)
else:
    background_all = neg_row_vals.copy()

background_all = background_all.dropna()
background_mean = background_all.mean()

print(f"Background mean OD (controls): {background_mean:.3f}")

# -------------------------
# 5. Compute mean and SEM for each concentration, minus background
# -------------------------

def mean_and_sem(mat):
    """
    mat: DataFrame with rows = replicates (3), cols = concentrations
    Returns:
        mean (1D np.array), sem (1D np.array)
    """
    mean = mat.mean(axis=0).values
    sem = mat.sem(axis=0).values   # pandas sem uses std/sqrt(n)
    return mean, sem

high_mean_raw, high_sem_raw = mean_and_sem(high_vals)
low_mean_raw,  low_sem_raw  = mean_and_sem(low_vals)

# Subtract global background from means
high_mean = high_mean_raw - background_mean
low_mean  = low_mean_raw  - background_mean

##Checking that these arrays line up
print("Concs:", concs)
print("High mean:", high_mean)
print("Low mean:", low_mean)

# -------------------------
# 6. Define 4-parameter logistic (4PL) model and fit
# -------------------------

def four_pl(x, a, b, c, d):
    """
    4-parameter logistic function:
    y = d + (a - d) / (1 + (x / c)**b)
    """
    x = np.asarray(x, dtype=float)
    return d + (a - d) / (1.0 + (x / c)**b)

def fit_4pl(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove NaNs / infs
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]

    # Sort by x (concentration)
    order = np.argsort(x_fit)
    x_fit = x_fit[order]
    y_fit = y_fit[order]

    # Initial guesses
    a0 = np.max(y_fit) if y_fit.size else 1.0
    d0 = np.min(y_fit) if y_fit.size else 0.0
    b0 = 1.0
    # ensure positive c0 (EC50)
    c0 = np.median(x_fit) if np.median(x_fit) > 0 else (np.mean(np.abs(x_fit)) + 1e-6)

    p0 = [a0, b0, c0, d0]

    # Bounds: require c (EC50) >= 0
    lower = [-np.inf, -np.inf, 0.0, -np.inf]
    upper = [np.inf, np.inf, np.inf, np.inf]

    try:
        popt, pcov = curve_fit(four_pl, x_fit, y_fit, p0=p0, bounds=(lower, upper), maxfev=10000)
    except Exception:
        # fallback to initial guesses if fit fails
        popt = np.array(p0, dtype=float)
        pcov = np.full((4, 4), np.nan)

    return popt, pcov

# Fit 4PL for each antibody concentration
popt_high, _ = fit_4pl(concs, high_mean)
popt_low,  _ = fit_4pl(concs, low_mean)

# Generate smooth curve data over the concentration range
x_smooth = np.logspace(np.log10(np.min(concs)), np.log10(np.max(concs)), 200)
y_high_fit = four_pl(x_smooth, *popt_high)
y_low_fit  = four_pl(x_smooth, *popt_low)

# -------------------------
# 7. Plot: standard curves with SEM and 4PL fits
# -------------------------

fig, ax = plt.subplots(figsize=(8, 6))

# Scatter + SEM error bars
ax.errorbar(
    concs,
    high_mean,
    yerr=high_sem_raw,   # SEM is not background-subtracted in magnitude; acceptable
    fmt="o",
    label="100 ng/mL Ab",
)
ax.errorbar(
    concs,
    low_mean,
    yerr=low_sem_raw,
    fmt="s",
    label="10 ng/mL Ab",
)

# 4PL fits
ax.plot(x_smooth, y_high_fit, linestyle="-", label="4PL fit (100 ng/mL)")
ax.plot(x_smooth, y_low_fit, linestyle="--", label="4PL fit (10 ng/mL)")

# Log-scale x-axis
ax.set_xscale("log")

ax.set_xlabel("SARS Spike antigen (ng/mL)")
ax.set_ylabel("OD450 (background-subtracted)")
ax.set_title("ELISA Standard Curve – SARS Spike (with 4PL fit)")

ax.legend()
ax.grid(True, which="both", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

## Checking which triplicates had more than 10% standard error - looking for consistency with pipetting
# Percent SEM (SEM divided by raw mean, before bg subtraction)
high_pct_sem = (high_sem_raw / high_mean_raw) * 100
low_pct_sem  = (low_sem_raw  / low_mean_raw) * 100

sem_table = pd.DataFrame({
    "Concentration (ng/mL)": concs,
    "High Ab Mean": high_mean_raw,
    "High Ab SEM": high_sem_raw,
    "High Ab SEM %": high_pct_sem,
    "Low Ab Mean": low_mean_raw,
    "Low Ab SEM": low_sem_raw,
    "Low Ab SEM %": low_pct_sem,
})

print("\n================ SEM TABLE ================")
print(sem_table.to_string(index=False))

# Flag SEM > 10%
print("\n========= SEM > 10% FLAG (High Ab) =========")
print(sem_table[sem_table["High Ab SEM %"] > 10][
    ["Concentration (ng/mL)", "High Ab SEM %"]
])

print("\n========= SEM > 10% FLAG (Low Ab) =========")
print(sem_table[sem_table["Low Ab SEM %"] > 10][
    ["Concentration (ng/mL)", "Low Ab SEM %"]
])