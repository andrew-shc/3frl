"""
Deep Hedging — Net P&L comparison: GBM vs Heston
=================================================
Synthetic data only (Monte Carlo simulation).
Maturity: 1 year (252 trading days).

GBM  — Black-Scholes assumptions hold → BS is the optimal hedge.
Heston — Stochastic volatility → BS is suboptimal; deep hedgers adapt.

Net P&L = hedging gains − option payoff + premium collected.
Centered near zero: positive = profit, negative = loss.
"""

import os, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pfhedge.instruments import BrownianStock, HestonStock, EuropeanOption
from pfhedge.nn import (
    BlackScholes, Hedger, MultiLayerPerceptron,
    EntropicRiskMeasure, ExpectedShortfall,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED     = 42
MATURITY = 300 / 250        # 300 steps (arbitrary — interpretable as hours/minutes)
N_TRAIN  = 10_000
N_EPOCHS = 100
N_EVAL   = 30_000
INPUTS   = ["log_moneyness", "expiry_time", "volatility", "prev_hedge"]
C        = {"BS": "#2196F3", "ERM": "#FF9800", "CVaR": "#4CAF50"}

def train_hedgers(stock_fn, label):
    """Train BS, ERM, CVaR hedgers on a given stock. Returns (hedgers, prices, net_pls)."""
    print(f"\n── {label} ──────────────────────────")

    torch.manual_seed(SEED)
    s = stock_fn(); opt = EuropeanOption(s, maturity=MATURITY)
    bs_h  = Hedger(BlackScholes(opt), BlackScholes(opt).inputs())
    bs_p  = bs_h.price(opt, n_paths=N_EVAL)
    opt.simulate(n_paths=N_EVAL)
    bs_pl = bs_h.compute_pl(opt).detach().numpy() + float(bs_p)
    print(f"  BS    price={float(bs_p):.4f}  std={bs_pl.std():.4f}")

    torch.manual_seed(SEED)
    s = stock_fn(); opt = EuropeanOption(s, maturity=MATURITY)
    erm_h = Hedger(MultiLayerPerceptron(), INPUTS, EntropicRiskMeasure(1.0))
    erm_h.fit(opt, n_paths=N_TRAIN, n_epochs=N_EPOCHS, verbose=True,
              tqdm_kwargs={"desc": "  ERM "})
    erm_p  = erm_h.price(opt, n_paths=N_EVAL)
    opt.simulate(n_paths=N_EVAL)
    erm_pl = erm_h.compute_pl(opt).detach().numpy() + float(erm_p)
    print(f"  ERM   price={float(erm_p):.4f}  std={erm_pl.std():.4f}")

    torch.manual_seed(SEED)
    s = stock_fn(); opt = EuropeanOption(s, maturity=MATURITY)
    cvar_h = Hedger(MultiLayerPerceptron(), INPUTS, ExpectedShortfall(0.1))
    cvar_h.fit(opt, n_paths=N_TRAIN, n_epochs=N_EPOCHS, verbose=True,
               tqdm_kwargs={"desc": "  CVaR"})
    cvar_p  = cvar_h.price(opt, n_paths=N_EVAL)
    opt.simulate(n_paths=N_EVAL)
    cvar_pl = cvar_h.compute_pl(opt).detach().numpy() + float(cvar_p)
    print(f"  CVaR  price={float(cvar_p):.4f}  std={cvar_pl.std():.4f}")

    return (bs_pl, erm_pl, cvar_pl), (float(bs_p), float(erm_p), float(cvar_p))

gbm_pls,   gbm_prices  = train_hedgers(lambda: BrownianStock(cost=1e-4),           "GBM (σ=0.20 fixed)")
heston_pls, heston_prices = train_hedgers(lambda: HestonStock(cost=1e-4),           "Heston (stochastic vol)")

# ── Figure ────────────────────────────────────────────────────────────────────
FS = 15
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Net P&L — ATM European Call",
             fontsize=FS + 3, fontweight="bold")

LABELS = ["BS", "ERM", "CVaR"]
COLORS = [C["BS"], C["ERM"], C["CVaR"]]

def auto_clip(pls):
    return float(np.percentile(np.abs(np.concatenate(pls)), 99))

ROWS = [
    ("GBM",    gbm_pls,    auto_clip(gbm_pls)),
    ("Heston", heston_pls, auto_clip(heston_pls)),
]

for row, (row_title, pls, clip) in enumerate(ROWS):
    ax_d, ax_c = axes[row]
    bins = np.linspace(-clip, clip, 70)

    for pl, label, color in zip(pls, LABELS, COLORS):
        ax_d.hist(np.clip(pl, -clip, clip), bins=bins,
                  density=True, alpha=0.55, label=label, color=color)
        ax_c.plot(np.sort(pl), np.linspace(0, 1, len(pl)),
                  label=label, color=color, lw=2)

    for ax in (ax_d, ax_c):
        ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FS - 1)

    ax_d.set_title(row_title, fontsize=FS, fontweight="bold")
    ax_d.set_xlabel("Net P&L", fontsize=FS)
    ax_d.set_ylabel("Density", fontsize=FS)
    ax_d.legend(fontsize=FS - 1)

    ax_c.set_title(row_title, fontsize=FS, fontweight="bold")
    ax_c.set_xlabel("Net P&L", fontsize=FS)
    ax_c.set_ylabel("CDF", fontsize=FS)
    ax_c.axhline(0.1, color="gray", lw=0.8, ls=":", alpha=0.8)
    ax_c.legend(fontsize=FS - 1)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, "dh_pnl_european.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
