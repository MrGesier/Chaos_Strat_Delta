import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

N_NOTIONAL = 1_000_000      # USD equivalent per sleeve
T_DAYS = 30
ALPHA = 0.99                # 99% VaR / CVaR

# Funding (annualised)
MU_F = 0.15                 # 15% mean annual funding (basis)
SIGMA_F = 0.10              # 10% annual funding volatility

# Gap risk (large move you don't hedge perfectly)
LAMBDA_GAP = 0.5            # 0.5 events per year
LOSS_GAP = -0.05            # -5% of notional if a gap event hits

# Venue / default risk (Deribit extreme event)
LAMBDA_DEF = 0.02           # 2% per year severe venue event
LOSS_DEF = -0.50            # -50% of notional if it happens

N_SCENARIOS = 100_000
rng = np.random.default_rng(42)


# ============================================================
# ENGINE
# ============================================================

def simulate_basis_pnl(
    N=N_NOTIONAL,
    T=T_DAYS,
    mu_f=MU_F,
    sigma_f=SIGMA_F,
    lambda_gap=LAMBDA_GAP,
    loss_gap=LOSS_GAP,
    lambda_def=LAMBDA_DEF,
    loss_def=LOSS_DEF,
    n_scenarios=N_SCENARIOS,
):
    """
    Monte Carlo P&L for a delta-hedged basis trade on Deribit.

    Components:
      - funding P&L (what you want to earn)
      - gap P&L (one jump over horizon, Poisson)
      - venue/default P&L (Deribit event, Poisson)
    """
    # --- 1) Funding component ---
    mu_f_daily = mu_f / 365.0
    sigma_f_daily = sigma_f / np.sqrt(365.0)

    funding_daily = rng.normal(
        loc=mu_f_daily,
        scale=sigma_f_daily,
        size=(n_scenarios, T)
    )
    pnl_funding = N * funding_daily.sum(axis=1)

    # --- 2) Gap risk ---
    p_gap = lambda_gap * T / 365.0
    gap_event = rng.random(n_scenarios) < p_gap
    pnl_gap = np.where(gap_event, N * loss_gap, 0.0)

    # --- 3) Venue / default risk ---
    p_def = lambda_def * T / 365.0
    def_event = rng.random(n_scenarios) < p_def
    pnl_def = np.where(def_event, N * loss_def, 0.0)

    # --- Total P&L ---
    pnl_total = pnl_funding + pnl_gap + pnl_def
    return pnl_total, pnl_funding, pnl_gap, pnl_def


def compute_var_cvar(pnl, alpha=ALPHA):
    """
    Compute VaR and CVaR on the loss distribution = -PnL.
    """
    losses = -pnl
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = tail.mean() if len(tail) > 0 else var
    return var, cvar


# ============================================================
# PLOTS
# ============================================================

def plot_basis_risk(
    pnl_total,
    pnl_funding,
    pnl_gap,
    pnl_def,
    var,
    cvar,
    N=N_NOTIONAL,
    T=T_DAYS,
):
    """
    Create a 3-panel figure:
      1) P&L distribution with VaR/CVaR
      2) Component P&L (funding / gap / default), full vs tail
      3) Risk/return vs notional scaling
    """
    # Convert to % of notional for nicer axes
    pnl_pct = pnl_total / N
    var_pct = var / N
    cvar_pct = cvar / N

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    # ---- Panel 1: P&L distribution ----
    ax1.hist(pnl_pct, bins=80, alpha=0.7, edgecolor="black")
    ax1.axvline(-var_pct, color="red", linestyle="--",
                label=f"VaR {int(ALPHA*100)}: {-var_pct:.2%}")
    ax1.axvline(-cvar_pct, color="darkred", linestyle="--",
                label=f"CVaR {int(ALPHA*100)}: {-cvar_pct:.2%}")
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_title("30d basis P&L distribution")
    ax1.set_xlabel("PnL / Notional")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---- Panel 2: Component contributions ----
    tail_mask = -pnl_total >= var
    # Full-sample averages
    mean_funding = pnl_funding.mean() / N
    mean_gap = pnl_gap.mean() / N
    mean_def = pnl_def.mean() / N
    # Tail-only averages
    mean_funding_tail = pnl_funding[tail_mask].mean() / N if tail_mask.any() else 0.0
    mean_gap_tail = pnl_gap[tail_mask].mean() / N if tail_mask.any() else 0.0
    mean_def_tail = pnl_def[tail_mask].mean() / N if tail_mask.any() else 0.0

    labels = ["Funding", "Gap", "Venue"]
    full_vals = [mean_funding, mean_gap, mean_def]
    tail_vals = [mean_funding_tail, mean_gap_tail, mean_def_tail]

    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width/2, full_vals, width, label="Average (all scenarios)")
    ax2.bar(x + width/2, tail_vals, width, label=f"Average (tail ≥ VaR {int(ALPHA*100)}%)")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("PnL / Notional")
    ax2.set_title("Component contributions\n(full sample vs tail)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ---- Panel 3: Risk/return vs notional ----
    # We exploit linear scaling: PnL ∝ notional
    scales = np.linspace(0.25, 2.0, 15)  # 0.25x to 2.0x notional
    exp_annuals = []
    cvars_pct = []

    base_exp_30d = pnl_pct.mean()
    base_cvar_pct = cvar_pct

    for s in scales:
        # PnL scales linearly, so returns (as %) stay the same,
        # but absolute CVaR in $ scales with s.
        exp_30d_scaled = base_exp_30d * s
        exp_annual = (1 + exp_30d_scaled) ** (365 / T) - 1
        exp_annuals.append(exp_annual)

        cvar_scaled_pct = base_cvar_pct * s
        cvars_pct.append(-cvar_scaled_pct)

    ax3.plot(cvars_pct, exp_annuals, marker="o")
    # Mark current notional at scale=1
    current_cvar = -base_cvar_pct
    current_exp_annual = (1 + base_exp_30d) ** (365 / T) - 1
    ax3.scatter([current_cvar], [current_exp_annual],
                color="red", s=80, marker="*",
                label="Current notional (1x)")
    ax3.set_xlabel("−CVaR 30d / Notional")
    ax3.set_ylabel("Expected annual return")
    ax3.set_title("Risk/return vs notional scaling")
    ax3.legend()
    ax3.grid(alpha=0.3)

    fig.suptitle("Deribit basis sleeve — funding + gap + venue risk", fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DERIBIT BASIS SLEEVE — RISK ENGINE")
    print("=" * 70)
    print("\nAssumptions:")
    print(f"  • Notional: ${N_NOTIONAL:,.0f}, horizon: {T_DAYS} days.")
    print(f"  • Funding: μ = {MU_F:.0%} annual, σ = {SIGMA_F:.0%} annual.")
    print(f"  • Gap event: λ_gap = {LAMBDA_GAP:.2f}/year, loss = {LOSS_GAP:.0%} of notional.")
    print(f"  • Venue event: λ_def = {LAMBDA_DEF:.2f}/year, loss = {LOSS_DEF:.0%} of notional.")
    print(f"  • Monte Carlo: {N_SCENARIOS:,} scenarios, VaR/CVaR level = {int(ALPHA*100)}%.")
    print("-" * 70)

    # 1) Simulate
    pnl_total, pnl_funding, pnl_gap, pnl_def = simulate_basis_pnl()

    # 2) Risk metrics
    var99, cvar99 = compute_var_cvar(pnl_total, alpha=ALPHA)

    exp_30d = pnl_total.mean() / N_NOTIONAL
    ann_equiv = (1 + exp_30d) ** (365 / T_DAYS) - 1

    print(f"Expected 30d PnL: {exp_30d:.2%} of notional "
          f"(~{ann_equiv:.1%} annualised)")
    print(f"99% VaR (30d): {var99 / N_NOTIONAL:.2%} of notional")
    print(f"99% CVaR (30d): {cvar99 / N_NOTIONAL:.2%} of notional")

    # Quick decomposition of average losses
    print("\nAverage component PnL (all scenarios):")
    print(f"  Funding: {pnl_funding.mean() / N_NOTIONAL:.2%}")
    print(f"  Gap:     {pnl_gap.mean() / N_NOTIONAL:.2%}")
    print(f"  Venue:   {pnl_def.mean() / N_NOTIONAL:.2%}")

    # 3) Plots (always shown)
    plot_basis_risk(pnl_total, pnl_funding, pnl_gap, pnl_def, var99, cvar99)
