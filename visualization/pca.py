import os
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def run_pca_visualization(
    data: Union[pd.DataFrame, str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    n_components: int = 2,
    pairs_to_plot: Optional[Union[int, List[int]]] = None,
    save_plots: bool = True,
    show_plots: bool = True,
) -> tuple[pd.DataFrame, PCA, plt.Figure, plt.Figure, go.Figure]:
    """
    Computes PCA on embedding dimensions and generates:
      1. 2D trajectory plot across layers (Matplotlib)
      2. 3D trajectory plot across layers (Matplotlib)
      3. Interactive 3D trajectory plot (Plotly)

    Args:
        data: DataFrame or file path to CSV containing vector dimensions (dim_*) and Layer column.
        output_dir: Directory to save generated images and HTML files.
        n_components: Number of PCA dimensions (default: 2).
        pairs_to_plot: Max number of pairs (e.g. 5) or specific list of Pair_Index values to plot.
        save_plots: Whether to write plots to disk.
        show_plots: Whether to call plt.show() / display in notebook.

    Returns:
        tuple: (df_with_pca, pca_model, fig_2d, fig_3d_mpl, fig_3d_plotly)
    """
    # 1. Load data
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Expected DataFrame or path, got {type(data)}")

    # Ensure required columns
    dim_cols = [col for col in df.columns if col.startswith("dim_")]
    if not dim_cols:
        raise ValueError("No dimension columns ('dim_*') found in dataset.")

    if "Layer" not in df.columns and "layer" in df.columns:
        df["Layer"] = df["layer"]
    if "Layer" not in df.columns:
        df["Layer"] = 0

    if "Pair_Index" not in df.columns and "pair_index" in df.columns:
        df["Pair_Index"] = df["pair_index"]
    if "Pair_Index" not in df.columns:
        df["Pair_Index"] = 0

    X = df[dim_cols].values

    # 2. PCA Reduction
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    var_1 = pca.explained_variance_ratio_[0] * 100
    var_2 = pca.explained_variance_ratio_[1] * 100
    print(f"PCA Reduction Complete: PC1 = {var_1:.1f}% variance, PC2 = {var_2:.1f}% variance")

    # Filter pairs if requested
    available_pairs = list(df["Pair_Index"].unique())
    if pairs_to_plot is not None:
        if isinstance(pairs_to_plot, int):
            selected_pairs = available_pairs[:pairs_to_plot]
        else:
            selected_pairs = [p for p in pairs_to_plot if p in available_pairs]
    else:
        selected_pairs = available_pairs

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(".")

    # 3. Generate 2D Scatter/Trajectory Plot
    fig_2d, ax_2d = plt.subplots(figsize=(9, 6))
    for pair in selected_pairs:
        pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
        ax_2d.plot(
            pair_df["PC1"],
            pair_df["PC2"],
            marker="o",
            linestyle="-",
            linewidth=2,
            label=f"Pair {pair}",
        )
        for _, row in pair_df.iterrows():
            ax_2d.annotate(
                f"L{int(row['Layer'])}",
                (row["PC1"], row["PC2"]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
            )

    ax_2d.set_title("2D PCA Projection of Embeddings Across Layers")
    ax_2d.set_xlabel(f"PC1 ({var_1:.1f}% Variance)")
    ax_2d.set_ylabel(f"PC2 ({var_2:.1f}% Variance)")
    ax_2d.grid(True, linestyle="--", alpha=0.5)
    ax_2d.legend()
    fig_2d.tight_layout()

    if save_plots:
        save_file_2d = out_path / "2d_vector_projection.png"
        fig_2d.savefig(save_file_2d, dpi=300)
        print(f"Saved 2D plot: {save_file_2d}")

    # 4. Generate 3D Matplotlib Trajectory Plot (PC1, PC2, Layer)
    fig_3d_mpl = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d_mpl.add_subplot(111, projection="3d")

    for pair in selected_pairs:
        pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
        ax_3d.plot(
            pair_df["PC1"],
            pair_df["PC2"],
            pair_df["Layer"],
            marker="o",
            linewidth=2.5,
            label=f"Pair {pair}",
        )
        for _, row in pair_df.iterrows():
            ax_3d.text(
                row["PC1"],
                row["PC2"],
                row["Layer"],
                f"  L{int(row['Layer'])}",
                size=9,
            )

    ax_3d.set_title("3D Trajectory in Latent Space (PC1, PC2, Layer)")
    ax_3d.set_xlabel(f"PC1 ({var_1:.1f}%)")
    ax_3d.set_ylabel(f"PC2 ({var_2:.1f}%)")
    ax_3d.set_zlabel("Model Layer")
    ax_3d.grid(True)
    ax_3d.legend()
    fig_3d_mpl.tight_layout()

    if save_plots:
        save_file_3d = out_path / "3d_vector_projection.png"
        fig_3d_mpl.savefig(save_file_3d, dpi=300)
        print(f"Saved 3D Matplotlib plot: {save_file_3d}")

    # 5. Generate Interactive 3D Plotly HTML
    fig_3d_plotly = go.Figure()
    for pair in selected_pairs:
        pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
        fig_3d_plotly.add_trace(
            go.Scatter3d(
                x=pair_df["PC1"],
                y=pair_df["PC2"],
                z=pair_df["Layer"],
                mode="lines+markers+text",
                name=f"Pair {pair}",
                text=[f"L{int(l)}" for l in pair_df["Layer"]],
                textposition="top center",
            )
        )

    fig_3d_plotly.update_layout(
        title="Interactive 3D Trajectory in Latent Space",
        scene=dict(
            xaxis_title=f"PC1 ({var_1:.1f}%)",
            yaxis_title=f"PC2 ({var_2:.1f}%)",
            zaxis_title="Model Layer",
        ),
    )

    if save_plots:
        save_file_html = out_path / "interactive_3d_latent_space.html"
        fig_3d_plotly.write_html(str(save_file_html))
        print(f"Saved Interactive 3D Plotly plot: {save_file_html}")

    if show_plots:
        plt.show()

    return df, pca, fig_2d, fig_3d_mpl, fig_3d_plotly


# Module aliases
plot_pca_embeddings = run_pca_visualization


if __name__ == "__main__":
    test_files = [
        "vector_diffs.csv",
        "toy_data_vector_qwen_diffs.csv",
        "output/embeddings/vector_diffs.csv",
    ]
    found_file = next((f for f in test_files if os.path.exists(f)), None)
    if found_file:
        print(f"Running standalone PCA visualization on {found_file}...")
        run_pca_visualization(found_file, output_dir=".", show_plots=False)
    else:
        print("No default vector differences CSV found for standalone run.")