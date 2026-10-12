import os
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def run_umap_visualization(
    data: Union[pd.DataFrame, str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    n_pca: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    pairs_to_plot: Optional[Union[int, List[int]]] = None,
    save_plots: bool = True,
    show_plots: bool = True,
) -> tuple[pd.DataFrame, plt.Figure, plt.Figure, go.Figure]:
    """
    Computes PCA -> UMAP reduction on embedding dimensions and generates:
      1. 2D trajectory plot across layers (Matplotlib)
      2. 3D trajectory plot across layers (Matplotlib)
      3. Interactive 3D trajectory plot (Plotly)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("The 'umap-learn' package is required for UMAP visualization. Install with `pip install umap-learn`.")

    # 1. Load data
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Expected DataFrame or path, got {type(data)}")

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

    # 2. PCA -> UMAP
    actual_pca = min(n_pca, X.shape[1], X.shape[0])
    pca = PCA(n_components=actual_pca, random_state=42)
    X_pca = pca.fit_transform(X)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
    X_umap = reducer.fit_transform(X_pca)

    df["UMAP1"] = X_umap[:, 0]
    df["UMAP2"] = X_umap[:, 1]
    print(f"PCA ({actual_pca} dims) -> UMAP (2 dims) reduction complete.")

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

    # 3. 2D Matplotlib
    fig_2d, ax_2d = plt.subplots(figsize=(9, 6))
    for pair in selected_pairs:
        pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
        ax_2d.plot(
            pair_df["UMAP1"],
            pair_df["UMAP2"],
            marker="o",
            linestyle="-",
            linewidth=2,
            label=f"Pair {pair}",
        )
        for _, row in pair_df.iterrows():
            ax_2d.annotate(
                f"L{int(row['Layer'])}",
                (row["UMAP1"], row["UMAP2"]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
            )

    ax_2d.set_title("2D PCA-UMAP Projection of Vector Differences Across Layers")
    ax_2d.set_xlabel("UMAP Dimension 1")
    ax_2d.set_ylabel("UMAP Dimension 2")
    ax_2d.grid(True, linestyle="--", alpha=0.5)
    ax_2d.legend()
    fig_2d.tight_layout()

    if save_plots:
        save_file_2d = out_path / "2d_vector_projection_umap.png"
        fig_2d.savefig(save_file_2d, dpi=300)
        print(f"Saved 2D UMAP plot: {save_file_2d}")

    # 4. 3D Matplotlib
    fig_3d_mpl = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d_mpl.add_subplot(111, projection="3d")

    for pair in selected_pairs:
        pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
        ax_3d.plot(
            pair_df["UMAP1"],
            pair_df["UMAP2"],
            pair_df["Layer"],
            marker="o",
            linewidth=2.5,
            label=f"Pair {pair}",
        )
        for _, row in pair_df.iterrows():
            ax_3d.text(
                row["UMAP1"],
                row["UMAP2"],
                row["Layer"],
                f"  L{int(row['Layer'])}",
                size=9,
            )

    ax_3d.set_title("3D Trajectory in Latent Space (UMAP1, UMAP2, Layer)")
    ax_3d.set_xlabel("UMAP Dimension 1")
    ax_3d.set_ylabel("UMAP Dimension 2")
    ax_3d.set_zlabel("Model Layer")
    ax_3d.grid(True)
    ax_3d.legend()
    fig_3d_mpl.tight_layout()

    if save_plots:
        save_file_3d = out_path / "3d_vector_projection_umap.png"
        fig_3d_mpl.savefig(save_file_3d, dpi=300)
        print(f"Saved 3D Matplotlib UMAP plot: {save_file_3d}")

    # 5. Interactive 3D Plotly
    fig_3d_plotly = go.Figure()
    for pair in selected_pairs:
        pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
        fig_3d_plotly.add_trace(
            go.Scatter3d(
                x=pair_df["UMAP1"],
                y=pair_df["UMAP2"],
                z=pair_df["Layer"],
                mode="lines+markers+text",
                name=f"Pair {pair}",
                text=[f"L{int(l)}" for l in pair_df["Layer"]],
                textposition="top center",
            )
        )

    fig_3d_plotly.update_layout(
        title="Interactive 3D Trajectory in Latent Space (PCA-UMAP)",
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="Model Layer",
        ),
    )

    if save_plots:
        save_file_html = out_path / "interactive_3d_latent_space_umap.html"
        fig_3d_plotly.write_html(str(save_file_html))
        print(f"Saved Interactive 3D UMAP plot: {save_file_html}")

    if show_plots:
        plt.show()

    return df, fig_2d, fig_3d_mpl, fig_3d_plotly


plot_umap_embeddings = run_umap_visualization


if __name__ == "__main__":
    test_files = [
        "vector_diffs.csv",
        "toy_data_vector_qwen_diffs.csv",
        "output/embeddings/vector_diffs.csv",
    ]
    found_file = next((f for f in test_files if os.path.exists(f)), None)
    if found_file:
        print(f"Running standalone UMAP visualization on {found_file}...")
        run_umap_visualization(found_file, output_dir=".", show_plots=False)
    else:
        print("No default vector differences CSV found for standalone run.")