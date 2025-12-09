"""
CUSD Liquidity Programme — Incentive Flywheel & P&L Analysis

Figures:
A) Bribes -> TVL response curve
B) Net P&L vs circulating CUSD supply
C) Net LP APR vs TVL (flywheel efficiency region)
D) P&L waterfall (breakdown of components)
E) Heatmap: (CUSD supply, TVL) -> Net P&L
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =========================================================
# PARAMETERS (EASY TO TUNE)
# =========================================================

# Incentives / bribes
BRIBE_BUDGET = 1_000_000      # $ per year
EMISSION_LEVERAGE = 2.0       # emissions = leverage * bribes (veCurve/CVX effect)
LP_SHARE = 0.60               # our share of LP tokens in the flagship pool

# Curve pool economics
FEE_RATE = 0.0004             # 4 bps
VOLUME_PER_TVL = 36.0         # turnover per year (e.g. 3x/month => 36x/year)

# T-bill backing
T_BILL_YIELD = 0.05           # 5% on off-chain reserves

# Scale assumptions
TARGET_INCENTIVE_APR = 0.10   # target emissions APR for stable LPs
SEED_TVL = 12_000_000         # TVL we seed ourselves (USDT + CUSD)
TARGET_TVL = (BRIBE_BUDGET * EMISSION_LEVERAGE) / TARGET_INCENTIVE_APR  # ~20M

# Supply / TVL relationships for some plots
SUPPLY_MIN = 5_000_000
SUPPLY_MAX = 80_000_000
TVL_MIN = 5_000_000
TVL_MAX = 60_000_000

# Max fraction of supply that can realistically sit in one pool
# (mis en large pour ne PAS couper le cas 10M supply / 20M TVL)
MAX_TVL_SUPPLY_RATIO = 10.0   

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def emissions_from_bribes(bribes: float, leverage: float = EMISSION_LEVERAGE) -> float:
    """Total emissions paid to LPs per year given bribe budget."""
    return bribes * leverage

def incentive_apr(emissions: float, tvl: float) -> float:
    """Emissions APR (as decimal) given pool TVL."""
    if tvl <= 0:
        return 0.0
    return emissions / tvl

def fee_apr(volume_per_tvl: float = VOLUME_PER_TVL,
            fee_rate: float = FEE_RATE) -> float:
    """Fee APR (as decimal) for LPs, independent of TVL under constant V/TVL."""
    return volume_per_tvl * fee_rate

def our_fee_income(tvl: float,
                   volume_per_tvl: float = VOLUME_PER_TVL,
                   fee_rate: float = FEE_RATE,
                   lp_share: float = LP_SHARE) -> float:
    """Annual fee income (USD) for our LP share."""
    total_fees = tvl * volume_per_tvl * fee_rate
    return total_fees * lp_share

def reserve_yield(cusd_supply: float,
                  y: float = T_BILL_YIELD) -> float:
    """Annual T-bill carry on circulating CUSD (USD)."""
    return cusd_supply * y

def net_incentive_pnl(tvl: float,
                      bribes: float = BRIBE_BUDGET,
                      leverage: float = EMISSION_LEVERAGE,
                      lp_share: float = LP_SHARE,
                      tvl_cap: float = TARGET_TVL) -> float:
    """
    Net P&L from the incentive loop for us.
    """
    emissions = leverage * bribes
    utilisation = min(1.0, tvl / tvl_cap)
    our_emissions = emissions * lp_share * utilisation
    return our_emissions - bribes

def net_program_pnl(cusd_supply: float,
                    tvl: float,
                    overhead_haircut: float = 0.0) -> float:
    """
    Global P&L of the liquidity programme at given supply & TVL (USD / year).
    """
    # Enforce TVL <= MAX_TVL_SUPPLY_RATIO * supply
    tvl_eff = min(tvl, MAX_TVL_SUPPLY_RATIO * cusd_supply)

    y_res = reserve_yield(cusd_supply)
    y_fees = our_fee_income(tvl_eff)
    y_incentive = net_incentive_pnl(tvl_eff)
    return y_res + y_fees + y_incentive - overhead_haircut

# =========================================================
# FIGURE A — Bribes vs TVL response (simple model)
# =========================================================

def plot_bribes_vs_tvl():
    bribes_grid = np.linspace(200_000, 2_000_000, 10)
    target_aprs = [0.08, 0.10, 0.12]

    plt.figure(figsize=(8, 5))
    for apr in target_aprs:
        emissions = EMISSION_LEVERAGE * bribes_grid
        tvl = emissions / apr
        plt.plot(bribes_grid / 1e6, tvl / 1e6, marker="o", label=f"Target APR = {apr*100:.0f}%")

    emissions_star = emissions_from_bribes(BRIBE_BUDGET)
    tvl_star = emissions_star / TARGET_INCENTIVE_APR
    plt.scatter(BRIBE_BUDGET / 1e6, tvl_star / 1e6, c="red", s=80, zorder=5,
                label=f"Chosen: 1M bribe → {tvl_star/1e6:.1f}M TVL @ 10% APR")

    plt.xlabel("Annual bribes (M USD)")
    plt.ylabel("Equilibrium TVL (M USD)")
    plt.title("Figure A — Bribes → Equilibrium TVL (Curve stable pool)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

# =========================================================
# FIGURE B — Net P&L vs CUSD supply
# =========================================================

def plot_pnl_vs_supply():
    supplies = np.linspace(SUPPLY_MIN, SUPPLY_MAX, 80)
    pnls = []
    tvls = []

    # Simple assumption: TVL is min(TARGET_TVL, 0.6 * supply)
    for S in supplies:
        tvl = min(TARGET_TVL, 0.6 * S)
        tvls.append(tvl)
        pnls.append(net_program_pnl(S, tvl))

    pnls = np.array(pnls)
    tvls = np.array(tvls)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(supplies / 1e6, pnls / 1e6, label="Net P&L", color="tab:blue")
    ax1.axhline(0, color="black", linewidth=1, linestyle="--")
    ax1.set_xlabel("Circulating CUSD supply (M USD)")
    ax1.set_ylabel("Net annual P&L (M USD)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.set_title("Figure B — Net P&L vs CUSD supply (fixed bribe budget)")
    ax1.grid(alpha=0.3)

    breakeven_indices = np.where(np.sign(pnls[:-1]) != np.sign(pnls[1:]))[0]
    if len(breakeven_indices) > 0:
        idx = breakeven_indices[0]
        S_star = supplies[idx]
        P_star = pnls[idx]
        ax1.scatter(S_star / 1e6, P_star / 1e6, c="red", s=80,
                    label=f"Break-even ~{S_star/1e6:.1f}M CUSD")
        ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(supplies / 1e6, tvls / 1e6, color="tab:orange", linestyle=":",
             label="Implied TVL")
    ax2.set_ylabel("Implied TVL (M USD)", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()

# =========================================================
# FIGURE C — Net LP APR vs TVL
# =========================================================

def plot_net_apr_vs_tvl():
    tvls = np.linspace(TVL_MIN, TVL_MAX, 100)
    emissions = emissions_from_bribes(BRIBE_BUDGET)

    inc_aprs = np.array([incentive_apr(emissions, T) for T in tvls])
    fee_aprs = np.full_like(inc_aprs, fee_apr())
    net_aprs = inc_aprs + fee_aprs

    plt.figure(figsize=(8, 5))
    plt.plot(tvls / 1e6, inc_aprs * 100, label="Incentive APR", color="tab:blue")
    plt.plot(tvls / 1e6, fee_aprs * 100, label="Fee APR", color="tab:orange")
    plt.plot(tvls / 1e6, net_aprs * 100, label="Net APR", color="tab:green", linewidth=2)

    plt.axvspan(20, 30, color="grey", alpha=0.1, label="Target TVL 20–30M")

    plt.xlabel("Pool TVL (M USD)")
    plt.ylabel("APR (%)")
    plt.title("Figure C — LP APR vs TVL for fixed bribe budget")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

# =========================================================
# FIGURE D — P&L waterfall for a reference point
# =========================================================

def plot_pnl_waterfall(reference_supply=10_000_000, reference_tvl=20_000_000):
    """
    Waterfall at one reference point (e.g. 10M supply, 20M TVL).
    """
    S = reference_supply
    T = reference_tvl

    T_eff = min(T, MAX_TVL_SUPPLY_RATIO * S)

    y_res = reserve_yield(S)
    y_fees = our_fee_income(T_eff)
    y_incentive = net_incentive_pnl(T_eff)
    y_net = y_res + y_fees + y_incentive

    components = ["Reserve yield", "Fees", "Net incentives", "Net P&L"]
    values = [y_res, y_fees, y_incentive, y_net]

    fig, ax = plt.subplots(figsize=(7, 4))

    xs = np.arange(len(components))
    colors = ["#4CAF50", "#2196F3", "#9C27B0", "#FF9800"]

    for i, v in enumerate(values):
        ax.bar(i, v / 1e6, color=colors[i])
        ax.text(i, (v / 1e6) * 0.5, f"{v/1e6:.2f}",
                ha="center", va="center", color="white", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(components, rotation=20)
    ax.set_ylabel("USD (M)")
    ax.set_title(f"Figure D — P&L waterfall @ Supply={S/1e6:.0f}M, TVL={T_eff/1e6:.0f}M")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

# =========================================================
# FIGURE E — Heatmap: supply vs TVL -> Net P&L
# =========================================================

def plot_pnl_heatmap():
    supplies = np.linspace(SUPPLY_MIN, SUPPLY_MAX, 40)
    tvls = np.linspace(TVL_MIN, TVL_MAX, 40)

    S_grid, T_grid = np.meshgrid(supplies, tvls)
    pnl_grid = np.zeros_like(S_grid, dtype=float)

    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            S = S_grid[i, j]
            T = T_grid[i, j]
            pnl_grid[i, j] = net_program_pnl(S, T)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(S_grid / 1e6, T_grid / 1e6, pnl_grid / 1e6,
                       shading="auto", cmap="RdYlGn")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Net P&L (M USD)")

    ax.set_xlabel("CUSD circulating supply (M USD)")
    ax.set_ylabel("Pool TVL (M USD)")
    ax.set_title("Figure E — Net P&L heatmap (supply vs TVL)")

    cs = ax.contour(S_grid / 1e6, T_grid / 1e6, pnl_grid,
                    levels=[0.0], colors="black", linewidths=1.2)
    ax.clabel(cs, fmt="P&L = 0", inline=True, fontsize=8)

    ax.scatter([10], [20], c="white", edgecolor="black", s=80,
               label="Example: 10M supply, 20M TVL")
    ax.legend(loc="lower right")

    fig.tight_layout()

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    plt.style.use("default")

    plot_bribes_vs_tvl()
    plot_pnl_vs_supply()
    plot_net_apr_vs_tvl()
    plot_pnl_waterfall(reference_supply=10_000_000, reference_tvl=20_000_000)
    plot_pnl_heatmap()

    plt.show()
