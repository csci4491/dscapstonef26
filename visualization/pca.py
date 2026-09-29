import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go

#input vector data
df = pd.read_csv("vector_diffs.csv")
dim_cols = [col for col in df.columns if col.startswith("dim_")]
X = df[dim_cols].values

#pca reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df["PC1"]=X_pca[:,0]
df["PC2"]=X_pca[:,1]


var_1 = pca.explained_variance_ratio_[0] * 100
var_2 = pca.explained_variance_ratio_[1] * 100

print(f"Explained Variance: PC1 = {var_1:.1f}%, PC2 = {var_2:.1f}%")


#Generate 2D Scatter/Trajectory Plot

plt.figure(figsize=(9, 6))
pairs = df["Pair_Index"].unique()

for pair in pairs:
    pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
    plt.plot(
        pair_df["PC1"],
        pair_df["PC2"],
        marker="o",
        linestyle="-",
        linewidth=2,
        label=f"Pair {pair}",
    )
    for _, row in pair_df.iterrows():
        plt.annotate(
            f"L{int(row['Layer'])}",
            (row["PC1"], row["PC2"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
        )

plt.title("2D PCA Projection of Vector Differences Across Layers")
plt.xlabel(f"PC1 ({var_1:.1f}% Variance)")
plt.ylabel(f"PC2 ({var_2:.1f}% Variance)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("2d_vector_projection.png", dpi=300)
plt.close()


#Generate 3D Matplotlib Trajectory Plot (PC1, PC2, Layer)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for pair in pairs:
    pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
    ax.plot(
        pair_df["PC1"],
        pair_df["PC2"],
        pair_df["Layer"],
        marker="o",
        linewidth=2.5,
        label=f"Pair {pair}",
    )

    for _, row in pair_df.iterrows():
        ax.text(
            row["PC1"],
            row["PC2"],
            row["Layer"],
            f"  L{int(row['Layer'])}",
            size=9,
        )

ax.set_title("3D Trajectory in Latent Space (PC1, PC2, Layer)")
ax.set_xlabel(f"PC1 ({var_1:.1f}%)")
ax.set_ylabel(f"PC2 ({var_2:.1f}%)")
ax.set_zlabel("Model Layer")
ax.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("3d_vector_projection.png", dpi=300)
plt.close()


#Generate Interactive 3D Plotly HTML (Rotatable in Browser)

fig_3d = go.Figure()

for pair in pairs:
    pair_df = df[df["Pair_Index"] == pair].sort_values("Layer")
    fig_3d.add_trace(
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

fig_3d.update_layout(
    title="Interactive 3D Trajectory in Latent Space",
    scene=dict(
        xaxis_title=f"PC1 ({var_1:.1f}%)",
        yaxis_title=f"PC2 ({var_2:.1f}%)",
        zaxis_title="Model Layer",
    ),
)

fig_3d.write_html("interactive_3d_latent_space.html")
print(
    "Plotting complete! Saved '2d_vector_projection.png', '3d_vector_projection.png', and 'interactive_3d_latent_space.html'."
)