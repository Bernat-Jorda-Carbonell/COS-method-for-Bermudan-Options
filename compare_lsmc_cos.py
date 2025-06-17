# compare_lsmc_cos.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gamma as sp_gamma
from cos_bermudan import price_bermudan_cos
from lsmc import price_american_lsmc_cgmy

# ----------------------------
# 1) USER‐SPECIFIED PARAMETERS
# ----------------------------

# Model parameters for CGMY (use exactly the same as in run_experiments.py)
S0     = 100.0
K      = 100.0
r      = 0.05
sigma  = 0.2   # not used by CGMY LSMC, but kept for consistency
C      = 1.0
G      = 5.0
Mjump  = 5.0
Y      = 1.5

T      = 1.0
M_steps = 10   # number of exercise dates (same as --M in run_experiments)
basis_deg = 3  # same as --basis_deg

# COS reference: choose a large N for “ground truth”
N_ref = 4096

# Monte Carlo sample sizes and multiple seeds to average over
M_sims_list = [500, 1000, 2000, 5000, 10000]
seeds       = [12345, 23456, 34567]  # e.g., three independent replicates per M_sims

# Directory to save comparison plots
plots_dir = os.path.join("plots", "plots_comparision")
os.makedirs(plots_dir, exist_ok=True)

# ----------------------------
# 2) COMPUTE CGMY COS REFERENCE
# ----------------------------
# We need the truncation interval [a, b] for COS:
x0 = np.log(S0 / K)
# Compute cumulants for CGMY to get a,b (as done in run_experiments.py):

c2 = T * C * sp_gamma(2 - Y) * (Mjump**(Y - 2) + G**(Y - 2))
c4 = T * C * sp_gamma(4 - Y) * (Mjump**(Y - 4) + G**(Y - 4))
L = 8.0  # same as default in run_experiments.py
half_width = L * np.sqrt(c2 + np.sqrt(c4))
a = x0 - half_width
b = x0 + half_width

# Define the CGMY characteristic function wrapper:
from cos_models import cgmy_charfn
charfn = lambda u, dt, rr: cgmy_charfn(u, dt, rr, C, G, Mjump, Y)

# Now compute the COS‐based Bermudan price once
print("Computing COS reference (this may take a moment)…")
V_COS, _ = price_bermudan_cos(
    S0=S0,
    K=K,
    r=r,
    T=T,
    M=M_steps,
    N=N_ref,
    a=a,
    b=b,
    charfn=charfn,
    option_type="put"  # or "call", whichever you want to compare
)
print(f"  COS reference price (N={N_ref}): {V_COS:.6f}\n")

# ----------------------------
# 3) RUN LSMC MULTIPLE TIMES & COLLECT ERRORS
# ----------------------------
records = []

for M_sims in M_sims_list:
    for seed in seeds:
        # Price via CGMY LSMC
        price_lsmc = price_american_lsmc_cgmy(
            S0=S0,
            K=K,
            r=r,
            C=C,
            G=G,
            M_jump=Mjump,
            Y=Y,
            T=T,
            M_steps=M_steps,
            M_sims=M_sims,
            basis_degree=basis_deg,
            seed=seed
        )
        abs_err = abs(price_lsmc - V_COS)
        rel_err = abs_err / V_COS * 100.0
        records.append({
            "M_sims": M_sims,
            "seed": seed,
            "lsmc_price": price_lsmc,
            "abs_error": abs_err,
            "rel_error_pct": rel_err
        })
        print(f"[M_sims={M_sims}, seed={seed}] LSMC price = {price_lsmc:.6f}, abs_err = {abs_err:.6f}, rel_err = {rel_err:.2f}%")

df_all = pd.DataFrame(records)

# (Optional) Save merged results
df_all.to_csv(os.path.join("csv", "all_lsmc.csv"), index=False)
print("\nMerged all LSMC runs into 'csv/ all_lsmc.csv'")

# ----------------------------
# 4) AGGREGATE & PLOT
# ----------------------------
# Group by M_sims to get mean and std of errors
grouped = df_all.groupby("M_sims")
df_stats = pd.DataFrame({
    "M_sims":        grouped["M_sims"].first(),
    "mean_abs_err":  grouped["abs_error"].mean(),
    "std_abs_err":   grouped["abs_error"].std(),
    "mean_rel_err":  grouped["rel_error_pct"].mean(),
    "std_rel_err":   grouped["rel_error_pct"].std()
}).reset_index(drop=True)

# Plot mean absolute error +/- std vs. M_sims (log–log)
plt.figure(figsize=(6, 4))
plt.errorbar(
    df_stats["M_sims"],
    df_stats["mean_abs_err"],
    yerr=df_stats["std_abs_err"],
    fmt="o-",
    capsize=4,
    color="C1",
    label="Mean +/- Std Abs Error"
)
plt.xscale("log", base=10)
plt.yscale("log", base=10)
plt.xlabel("Number of Monte Carlo paths (M_sims)")
plt.ylabel("Mean absolute error vs. COS(ref)")
plt.title(f"CGMY LSMC Convergence (Absolute Error, deg={basis_deg})")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
abs_err_plot = os.path.join(plots_dir, "lsmc_mean_abs_error_vs_Msims.png")
plt.savefig(abs_err_plot, dpi=150)
plt.close()

# Plot mean relative error +/- std vs. M_sims (log–log)
plt.figure(figsize=(6, 4))
plt.errorbar(
    df_stats["M_sims"],
    df_stats["mean_rel_err"],
    yerr=df_stats["std_rel_err"],
    fmt="o-",
    capsize=4,
    color="C2",
    label="Mean +/- Std Rel Error (%)"
)
plt.xscale("log", base=10)
plt.yscale("log", base=10)
plt.xlabel("Number of Monte Carlo paths (M_sims)")
plt.ylabel("Mean relative error vs. COS(ref) [%]")
plt.title(f"CGMY LSMC Convergence (Relative Error, deg={basis_deg})")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
rel_err_plot = os.path.join(plots_dir, "lsmc_mean_rel_error_vs_Msims.png")
plt.savefig(rel_err_plot, dpi=150)
plt.close()

print(f"\nSaved comparison plots in '{plots_dir}':")
print(f"  • {abs_err_plot}")
print(f"  • {rel_err_plot}")

