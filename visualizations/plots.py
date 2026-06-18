"""Shared plotting helpers: layer x pooling heatmaps, layer curves, UMAP."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_spearman_heatmap(results: pd.DataFrame, title: str) -> plt.Figure:
    """results: rows = layer index, columns = pooling strategy, values = Spearman rho."""
    fig, ax = plt.subplots(figsize=(5, max(4, 0.4 * len(results))))
    sns.heatmap(results, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_xlabel("pooling strategy")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_layer_curve(results: pd.DataFrame, title: str) -> plt.Figure:
    """results: rows = layer index, columns = pooling strategy, values = Spearman rho."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for pooling in results.columns:
        ax.plot(results.index, results[pooling], marker="o", label=pooling)
    ax.set_xlabel("layer")
    ax.set_ylabel("Spearman rho vs. human judgments")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
