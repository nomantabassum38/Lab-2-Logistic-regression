#!/usr/bin/env python3
"""TNK129 Lab 2. Run this file to save both lab figures and their animations."""

# %% Shared setup
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import StrMethodFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lab2_results"
NEWTON_MAX = 500
GRADIENT_MAX = 50_000


def load_data():
    features = np.loadtxt(ROOT / "q1x.dat", ndmin=2)
    y = np.loadtxt(ROOT / "q1y.dat").reshape(-1)
    if features.shape != (len(y), 2) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected two features and one binary label per example")
    X = np.column_stack((np.ones(len(y)), features))
    return X, y


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


@dataclass
class Fit:
    label: str
    color: str
    style: str
    theta: np.ndarray
    likelihood: np.ndarray


def log_likelihood_history(X, y, theta):
    """Entry k is the training log-likelihood after k parameter updates."""
    signs = 1 - 2 * y
    return np.array([-np.logaddexp(0.0, signs * (X @ weights)).sum()
                     for weights in theta])


# %% Task 1 — Fit logistic regression using Newton's method
def task1_newton(X, y):
    theta = np.zeros((NEWTON_MAX + 1, X.shape[1]))
    for k in range(NEWTON_MAX):
        p = sigmoid(X @ theta[k])
        gradient = X.T @ (y - p)
        hessian = -(X.T * (p * (1 - p))) @ X
        theta[k + 1] = theta[k] - np.linalg.solve(hessian, gradient)

    print("Newton coefficients (θ₀, θ₁, θ₂):", np.round(theta[-1], 6))
    return Fit("Newton", "#252e38", "-", theta, log_likelihood_history(X, y, theta))


# %% Task 2 — Plot the training data and decision boundary
def task2_plot_boundaries(X, y, fits, frames):
    fig, ax = make_axes("Training data and decision boundaries", "$x_1$", "$x_2$")
    for value, marker, color in [(0, "x", "#ad3e36"), (1, "o", "#256597")]:
        points = X[y == value]
        style = {"facecolors": "none", "edgecolors": color} if value else {"color": color}
        ax.scatter(points[:, 1], points[:, 2], marker=marker, s=33,
                   linewidths=1.2, label=f"Class {value}", zorder=3, **style)
    x_range = np.array([X[:, 1].min(), X[:, 1].max()])
    y_range = np.array(ax.get_ylim())
    ax.set_autoscale_on(False)
    lines = [ax.plot([], [], color=f.color, ls=f.style, lw=2, label=f.label)[0]
             for f in fits]

    def update(k):
        for f, line in zip(fits, lines):
            intercept, a, b = f.theta[min(k, len(f.theta) - 1)]
            if abs(b) > np.finfo(float).eps:
                line.set_data(x_range, -(intercept + a * x_range) / b)
            elif abs(a) > np.finfo(float).eps:
                line.set_data([-intercept / a] * 2, y_range)
            else:
                line.set_data([], [])

    save_figure(fig, ax, update, frames, "01_decision_boundaries")


# %% Task 3 — Full-batch gradient ascent and comparison with Newton
def task3_full_batch(X, y):
    fits = []
    for rate, color, style in [(0.005, "#bb581d", "--"), (0.001, "#8a572f", ":")]:
        theta = np.zeros((GRADIENT_MAX + 1, X.shape[1]))
        for k in range(GRADIENT_MAX):
            gradient = X.T @ (y - sigmoid(X @ theta[k]))
            theta[k + 1] = theta[k] + rate * gradient

        fits.append(Fit(f"Batch (α = {rate:g})", color, style, theta,
                        log_likelihood_history(X, y, theta)))
    return fits


# %% Task 4 — Stochastic gradient ascent and comparison with batch and Newton
def task4_stochastic(X, y):
    fits = []
    for rate, color, style in [(0.002, "#226993", "-."), (0.001, "#408978", ":")]:
        theta = np.zeros((GRADIENT_MAX + 1, X.shape[1]))
        for k in range(GRADIENT_MAX):
            i = k % len(y)
            gradient = (y[i] - sigmoid(X[i] @ theta[k])) * X[i]
            theta[k + 1] = theta[k] + rate * gradient

        fits.append(Fit(f"Cyclic SGD (α = {rate:g})", color, style, theta,
                        log_likelihood_history(X, y, theta)))
    return fits


# %% Shared plotting and animation helpers
def make_axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.25, top=0.88)
    ax.set_title(title, loc="left", fontsize=15, fontweight="semibold", pad=12)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(color="#e1e5e9", linewidth=0.7)
    return fig, ax


def save_figure(fig, ax, update, frames, filename):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              frameon=False, fontsize=9)
    update(frames[-1])
    fig.savefig(OUT / f"{filename}.png", dpi=200)
    fig.savefig(OUT / f"{filename}.pdf")

    counter = ax.text(1, 1.025, "", transform=ax.transAxes, ha="right",
                      fontsize=10, color="#445365")

    def draw(k):
        update(k)
        counter.set_text(f"Update {k:,}")

    animation = FuncAnimation(fig, draw, frames=frames, repeat=False)
    with TemporaryDirectory(dir=OUT) as staging:
        gif = Path(staging) / f"{filename}.gif"
        animation.save(gif, writer=PillowWriter(fps=8), dpi=100)
        gif.replace(OUT / gif.name)
    plt.close(fig)


def plot_convergence(fits, frames):
    fig, ax = make_axes("Convergence of logistic regression", "Parameter updates",
                        "Training log-likelihood")
    ax.set_xscale("symlog", linthresh=1, linscale=0.45)
    ax.set_xticks([0, 1, 10, 100, 1000, 10_000, GRADIENT_MAX])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    lines = []
    for f in fits:
        line, = ax.plot(np.arange(len(f.likelihood)), f.likelihood, color=f.color,
                        ls=f.style, lw=1.6, label=f.label)
        lines.append(line)
    ax.autoscale_view()
    ax.set_autoscale_on(False)

    def update(k):
        for f, line in zip(fits, lines):
            end = min(k + 1, len(f.likelihood))
            line.set_data(np.arange(end), f.likelihood[:end])

    save_figure(fig, ax, update, frames, "02_optimization_convergence")


# %% Run the four tasks and save their shared figures
def main():
    X, y = load_data()
    newton = task1_newton(X, y)
    batch, batch_alt = task3_full_batch(X, y)
    stochastic, stochastic_alt = task4_stochastic(X, y)

    # Task 2 includes the baseline boundaries learned in Tasks 3 and 4.
    baseline = [newton, batch, stochastic]
    all_fits = [*baseline, batch_alt, stochastic_alt]
    OUT.mkdir(exist_ok=True)
    frames = np.unique(np.r_[0, np.geomspace(1, GRADIENT_MAX, 80).round().astype(int)])
    task2_plot_boundaries(X, y, baseline, frames)

    # Tasks 3 and 4 compare all methods against the same Newton reference.
    print("First update within 0.5 log-likelihood units of the Newton fit")
    target = newton.likelihood[-1] - 0.5
    for fit in all_fits:
        hits = np.flatnonzero(fit.likelihood >= target)
        print(f"  {fit.label:25} {hits[0] if hits.size else 'unreached'}")
    plot_convergence(all_fits, frames)


if __name__ == "__main__":
    main()
