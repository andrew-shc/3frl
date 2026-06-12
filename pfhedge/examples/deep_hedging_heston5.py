"""
Deep Hedging — 5 Heston Stocks, d=10 Hedge Positions (ERM)
Synthetic simulation only. Buehler et al. 2019.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pfhedge.instruments import HestonStock, EuropeanOption, VarianceSwap
from pfhedge.nn import Hedger, MultiLayerPerceptron, EntropicRiskMeasure

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED     = 42
MATURITY = 300 / 250
N_TRAIN  = 10_000
N_EPOCHS = 100
N_VIZ    = 6_000
INPUTS   = ["log_moneyness", "expiry_time", "volatility", "prev_hedge"]

CONFIGS = [
    dict(name='"SPY"',  kappa=1.5, theta=0.04, sigma=0.15, rho=-0.50),
    dict(name='"AAPL"', kappa=1.0, theta=0.06, sigma=0.20, rho=-0.60),
    dict(name='"NVDA"', kappa=0.8, theta=0.09, sigma=0.25, rho=-0.70),
    dict(name='"MSFT"', kappa=2.0, theta=0.04, sigma=0.15, rho=-0.40),
    dict(name='"TSLA"', kappa=0.7, theta=0.06, sigma=0.25, rho=-0.80),
]

COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]
FS     = 16

def make_vs_pricer(kappa, theta, maturity):
    """
    Buehler et al. (2019) Eq. — second hedging instrument for Heston:

        S^(2k)_t = ∫_0^t V_s ds + L(t, V_t)

    where  L(t,v) = (v − b)/α · (1 − exp(−α(T−t))) + b(T−t)
    with   α = kappa,  b = theta   (paper notation → our Heston params).
    """
    def pricer(varswap):
        v   = varswap.ul().variance                          # (N, T)
        n   = v.shape[1]
        dt  = maturity / max(n - 1, 1)

        # Realized ∫_0^t V_s ds  (Euler left-sum)
        realized = torch.cumsum(v * dt, dim=1)               # (N, T)

        # Remaining TTM τ at each column, clamped away from 0 for numerical safety
        tau = torch.linspace(maturity, 0.0, n,
                             dtype=v.dtype, device=v.device).clamp(min=1e-8)

        # E_Q[∫_t^T V_s ds | V_t]
        future = theta * tau + (v - theta) * (1 - torch.exp(-kappa * tau)) / kappa

        return realized + future                             # (N, T)
    return pricer

# ── Train ─────────────────────────────────────────────────────────────────────
results = {}
for cfg in CONFIGS:
    name = cfg["name"]
    print(f"\n── {name} ──────────────────────────────")

    torch.manual_seed(SEED)
    stock   = HestonStock(kappa=cfg["kappa"], theta=cfg["theta"],
                          sigma=cfg["sigma"],  rho=cfg["rho"], cost=1e-4)
    option  = EuropeanOption(stock, maturity=MATURITY)
    varswap = VarianceSwap(stock)
    varswap.list(make_vs_pricer(cfg["kappa"], cfg["theta"], MATURITY))

    model  = MultiLayerPerceptron(out_features=2)
    hedger = Hedger(model, INPUTS, EntropicRiskMeasure(1.0))

    option.simulate(n_paths=1)
    with torch.no_grad():
        _ = hedger.compute_pl(option, hedge=[stock, varswap])

    hedger.fit(option, hedge=[stock, varswap],
               n_paths=N_TRAIN, n_epochs=N_EPOCHS,
               verbose=True, tqdm_kwargs={"desc": f"  {name}"})

    torch.manual_seed(0)
    option.simulate(n_paths=N_VIZ)
    with torch.no_grad():
        pos = hedger.compute_hedge(option, hedge=[stock, varswap])

    spot        = stock.spot.detach().numpy()
    stoch_vol   = stock.variance.clamp(min=0).sqrt().detach().numpy()
    stock_delta = pos[:, 0, :].detach().numpy()
    varswap_pos = pos[:, 1, :].detach().numpy()

    log_m_mean  = np.log(spot[:, :-1]).mean(axis=1)
    sorted_idx  = np.argsort(log_m_mean)
    N           = len(sorted_idx)
    paths = {
        "ITM": sorted_idx[int(N * 0.80)],
        "ATM": sorted_idx[int(N * 0.50)],
        "OTM": sorted_idx[int(N * 0.20)],
    }

    results[name] = dict(spot=spot, stoch_vol=stoch_vol,
                         stock_delta=stock_delta, varswap_pos=varswap_pos,
                         paths=paths)

print("\nDone.\n")

T_DAYS = np.arange(list(results.values())[0]["spot"].shape[1])

# ── Combined figure: 2×2 (asset valuations top, hedge positions bottom) ───────
legend_els = (
    [Line2D([0],[0], color=c, lw=2.5, label=n) for c, n in zip(COLORS, results)]
    + [Line2D([0],[0], color="gray", lw=2,        label="Stock δ"),
       Line2D([0],[0], color="gray", lw=2, ls="--", label="Var-swap δ")]
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("5-Stock Heston — Valuations & ERM Hedge Positions",
             fontsize=FS + 2, fontweight="bold")

ax_s, ax_sv = axes[0]
ax_d, ax_vs = axes[1]

for color, (name, res) in zip(COLORS, results.items()):
    pidx = res["paths"]["ATM"]
    ax_s.plot(T_DAYS,  res["spot"][pidx],        color=color, lw=2, label=name)
    ax_sv.plot(T_DAYS, res["stoch_vol"][pidx],   color=color, lw=2)
    ax_d.plot(T_DAYS,  res["stock_delta"][pidx], color=color, lw=2)
    ax_vs.plot(T_DAYS, res["varswap_pos"][pidx], color=color, lw=2, ls="--")

ax_s.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.5)
ax_s.set_title("Stock price", fontsize=FS, fontweight="bold")
ax_s.set_xlabel("Day", fontsize=FS)
ax_s.set_ylabel("Price", fontsize=FS)
ax_s.legend(fontsize=FS - 1)
ax_s.grid(True, alpha=0.3)
ax_s.tick_params(labelsize=FS - 1)

ax_sv.set_title("Stochastic vol", fontsize=FS, fontweight="bold")
ax_sv.set_xlabel("Day", fontsize=FS)
ax_sv.set_ylabel("Vol", fontsize=FS)
ax_sv.grid(True, alpha=0.3)
ax_sv.tick_params(labelsize=FS - 1)

ax_d.set_title("Stock delta  δ", fontsize=FS, fontweight="bold")
ax_d.set_xlabel("Day", fontsize=FS)
ax_d.set_ylabel("δ", fontsize=FS)
ax_d.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.4)
ax_d.grid(True, alpha=0.3)
ax_d.tick_params(labelsize=FS - 1)

ax_vs.set_title("Var-swap delta  δ", fontsize=FS, fontweight="bold")
ax_vs.set_xlabel("Day", fontsize=FS)
ax_vs.set_ylabel("δ", fontsize=FS)
ax_vs.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.4)
ax_vs.grid(True, alpha=0.3)
ax_vs.tick_params(labelsize=FS - 1)

fig.legend(handles=legend_els, loc="lower center", ncol=7,
           fontsize=FS - 1, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(OUTPUT_DIR, "dh_heston5.png"), dpi=150, bbox_inches="tight")
print("Saved: dh_heston5.png")
