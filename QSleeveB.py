import numpy as np

# -----------------------------
# Parameters
# -----------------------------
N_NOTIONAL = 1_000_000      # USD equivalent
T_DAYS = 30
ALPHA = 0.99                # 99% VaR/CVaR

# Funding (annualised)
MU_F = 0.15                 # 15% mean annual funding (basis)
SIGMA_F = 0.10              # 10% annual funding vol

# Gap risk
LAMBDA_GAP = 0.5            # 0.5 events per year
LOSS_GAP = -0.05            # -5% of notional if gap event

# Venue / default risk
LAMBDA_DEF = 0.02           # 2% per year severe venue event
LOSS_DEF = -0.50            # -50% of notional if event

N_SCENARIOS = 100_000
rng = np.random.default_rng(42)


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
    # --- 1) Funding component ---
    # Daily funding ~ N(mu_f/365, sigma_f/sqrt(365))
    mu_f_daily = mu_f / 365.0
    sigma_f_daily = sigma_f / np.sqrt(365.0)

    funding_daily = rng.normal(mu_f_daily, sigma_f_daily,
                               size=(n_scenarios, T))
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
    # Losses = -PnL
    losses = -pnl
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = tail.mean() if len(tail) > 0 else var
    return var, cvar


if __name__ == "__main__":
    pnl, pnl_funding, pnl_gap, pnl_def = simulate_basis_pnl()
    var99, cvar99 = compute_var_cvar(pnl, alpha=ALPHA)

    exp_30d = pnl.mean() / N_NOTIONAL
    ann_equiv = (1 + exp_30d) ** (365 / T_DAYS) - 1

    print(f"Expected 30d PnL: {exp_30d:.2%} of notional "
          f"(~{ann_equiv:.1%} annualised)")
    print(f"99% VaR (30d): {var99 / N_NOTIONAL:.2%} of notional")
    print(f"99% CVaR (30d): {cvar99 / N_NOTIONAL:.2%} of notional")

    # Quick decomposition: expected loss from gap & default
    print(f"Avg gap PnL: {pnl_gap.mean() / N_NOTIONAL:.2%} of notional")
    print(f"Avg default PnL: {pnl_def.mean() / N_NOTIONAL:.2%} of notional")
