import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import plotly.graph_objects as go

#input vector data
df = pd.read_csv("vector_diffs.csv")
dim_cols = [col for col in df.columns if col.startswith("dim_")]
X = df[dim_cols].values

#Dimensionality Reduction: PCA (50 Dims) -> UMAP (2 Dims)
n_pca = min(50, X.shape[1], X.shape[0])
pca = PCA(n_components=n_pca, random_state=42)
X_pca = pca.fit_transform(X)

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_pca)

df["UMAP1"] = X_umap[:, 0]
df["UMAP2"] = X_umap[:, 1]

#Generate 2D Scatter/Trajectory Plot
plt.figure(figsize=(9, 6))
pairs = df["Pair_Index"].unique()

for pair in pairs:
    pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
    plt.plot(
        pair_df["UMAP1"],
        pair_df["UMAP2"],
        marker="o",
        linestyle="-",
        linewidth=2,
        label=f"Pair {pair}",
    )
    for _, row in pair_df.iterrows():
        plt.annotate(
            f"L{int(row['Layer'])}",
            (row["UMAP1"], row["UMAP2"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
        )

plt.title("2D PCA-UMAP Projection of Vector Differences Across Layers")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("2d_vector_projection_umap.png", dpi=300)
plt.close()

#Generate 3D Matplotlib Trajectory Plot (UMAP1, UMAP2, Layer)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for pair in pairs:
    pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
    ax.plot(
        pair_df["UMAP1"],
        pair_df["UMAP2"],
        pair_df["Layer"],
        marker="o",
        linewidth=2.5,
        label=f"Pair {pair}",
    )

    for _, row in pair_df.iterrows():
        ax.text(
            row["UMAP1"],
            row["UMAP2"],
            row["Layer"],
            f"  L{int(row['Layer'])}",
            size=9,
        )

ax.set_title("3D Trajectory in Latent Space (UMAP1, UMAP2, Layer)")
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.set_zlabel("Model Layer")
ax.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("3d_vector_projection_umap.png", dpi=300)
plt.close()

# 5. Generate Interactive 3D Plotly HTML
fig_3d = go.Figure()

for pair in pairs:
    pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
    fig_3d.add_trace(
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

fig_3d.update_layout(
    title="Interactive 3D Trajectory in Latent Space (PCA-UMAP)",
    scene=dict(
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        zaxis_title="Model Layer",
    ),
)

fig_3d.write_html("interactive_3d_latent_space_umap.html")
print(
    "Plotting complete! Saved '2d_vector_projection.png', '3d_vector_projection.png', and 'interactive_3d_latent_space.html'."
)