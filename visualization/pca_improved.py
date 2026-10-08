import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import plotly.graph_objects as go
import matplotlib as mpl


def get_inputs():
    """collect file paths and algorithm parameters from the user"""
    print("=" * 60)
    print(" LATENT SPACE TRAJECTORY VISUALIZER ")
    print("=" * 60)

    #collect CSV files
    file_paths = []
    print("\nEnter CSV file paths one by one. Press ENTER on an empty line when finished:\n")
    while True:
        path = input(f"File #{len(file_paths) + 1} path: ").strip().strip('"').strip("'")
        if not path:
            if not file_paths:
                print("Error: You must provide at least one CSV file.")
                continue
            break
        if not os.path.exists(path):
            print(f"  --> Warning: File '{path}' does not exist. Please re-enter.")
            continue
        file_paths.append(path)

    #choose algorithm method
    print("\nSelect Reduction Method:")
    print("  [1] PCA Only (2 Dimensions)")
    print("  [2] PCA + UMAP (PCA pre-reduction -> UMAP to 2D)")
    
    while True:
        choice = input("Choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("  --> Invalid input. Please enter 1 or 2.")

    #choose PCA components for UMAP pre-step (if applicable)
    n_pca = None
    if choice == "2":
        while True:
            try:
                n_pca = int(input("Enter number of PCA components before UMAP (e.g., 50): ").strip())
                if n_pca < 2:
                    print("  --> Must be at least 2")
                    continue
                break
            except ValueError:
                print("  --> Please enter a valid integer.")

    return file_paths, choice, n_pca


def main():
    file_paths, method_choice, n_pca = get_inputs()
#read and combine CSVs
    dfs = []
    for path in file_paths:
        temp_df = pd.read_csv(path)
        # Store filename without extension to identify source file
        temp_df["Source_File"] = os.path.splitext(os.path.basename(path))[0]
        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)

#extract dimension columns
    dim_cols = [col for col in df.columns if col.startswith("dim_")]
    if not dim_cols:
        raise ValueError("No columns starting with 'dim_' were found in the provided CSV files.")

    X = df[dim_cols].values
    total_samples, total_features = X.shape

#apply selected reduction strategy
    if method_choice == "1":
        print("\nRunning PCA reduction directly to 2D...")
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        
        df["Dim1"] = X_reduced[:, 0]
        df["Dim2"] = X_reduced[:, 1]
        
        var_1 = pca.explained_variance_ratio_[0] * 100
        var_2 = pca.explained_variance_ratio_[1] * 100
        
        dim1_label = f"PC1 ({var_1:.1f}% Var)"
        dim2_label = f"PC2 ({var_2:.1f}% Var)"
        method_title = "PCA"
        file_suffix = "pca"

    else:
        #cap PCA components to available sample/feature dimensions if needed
        max_possible = min(total_samples, total_features)
        actual_pca_components = min(n_pca, max_possible)
        
        if actual_pca_components < n_pca:
            print(f"Note: Adjusted PCA pre-reduction from {n_pca} to {actual_pca_components} due to data shape.")

        print(f"\nRunning PCA pre-reduction ({actual_pca_components} components) followed by UMAP (2D)...")
        pca = PCA(n_components=actual_pca_components, random_state=42)
        X_pca = pca.fit_transform(X)
    #n_neighbors will be tweaked to find best balance as it determines the integrity of local structure vs global 
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_pca)

        df["Dim1"] = X_umap[:, 0]
        df["Dim2"] = X_umap[:, 1]

        dim1_label = "UMAP Dimension 1"
        dim2_label = "UMAP Dimension 2"
        method_title = f"PCA ({actual_pca_components} dims) -> UMAP"
        file_suffix = "umap"

    #color and styling setup
    unique_files = df["Source_File"].unique()
    multiple_files = len(unique_files) > 1
    cmap = plt.get_cmap("tab10", max(len(unique_files), 1))
    file_colors = {file_name: cmap(i) for i, file_name in enumerate(unique_files)}

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    #generate 2D Plot (Matplotlib)
    plt.figure(figsize=(10, 7))
    for file_name, file_df in df.groupby("Source_File"):
        color = file_colors[file_name]
        pairs = file_df["Pair_Index"].unique()
        
        for i, pair in enumerate(pairs):
            pair_df = file_df[file_df["Pair_Index"] == pair].sort_values("Layer")
            plt.plot(
                pair_df["Dim1"],
                pair_df["Dim2"],
                marker=markers[i % len(markers)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                color=color,
                label=f"{file_name} (Pair {pair})",
            )
            

    plt.title(f"2D Vector Projection Across Layers ({method_title})")
    plt.xlabel(dim1_label)
    plt.ylabel(dim2_label)
    plt.grid(True, linestyle="--", alpha=0.5)
    if multiple_files:
        #build legend showing file colors
        legend_elements = [
            Line2D([0], [0], color=file_colors[f_name], lw=2.5, label=f_name)
            for f_name in unique_files
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    png_2d_name = f"2d_vector_projection_{file_suffix}.png"
    plt.savefig(png_2d_name, dpi=300)
    plt.close()

    #generate 3D Static Plot (Matplotlib)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for file_name, file_df in df.groupby("Source_File"):
        color = file_colors[file_name]
        pairs = file_df["Pair_Index"].unique()
        
        for i, pair in enumerate(pairs):
            pair_df = file_df[file_df["Pair_Index"] == pair].sort_values("Layer")
            ax.plot(
                pair_df["Dim1"],
                pair_df["Dim2"],
                pair_df["Layer"],
                marker=markers[i % len(markers)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                color=color,
                label=f"{file_name} (Pair {pair})",
            )
            

    ax.set_title(f"3D Trajectory in Latent Space ({method_title})")
    ax.set_xlabel(dim1_label)
    ax.set_ylabel(dim2_label)
    ax.set_zlabel("Model Layer")
    ax.grid(True)
    if multiple_files:
        legend_elements = [
            Line2D([0], [0], color=file_colors[f_name], lw=2.5, label=f_name)
            for f_name in unique_files
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    png_3d_name = f"3d_vector_projection_{file_suffix}.png"
    plt.savefig(png_3d_name, dpi=300)
    plt.close()

    #generate Interactive 3D Plot (Plotly HTML)
    fig_3d = go.Figure()
    plotly_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    for f_idx, (file_name, file_df) in enumerate(df.groupby("Source_File")):
        color = plotly_colors[f_idx % len(plotly_colors)]
        pairs = file_df["Pair_Index"].unique()
        
        for p_idx, pair in enumerate(pairs):
            pair_df = file_df[file_df["Pair_Index"] == pair].sort_values("Layer")
            
            #show legend trace only for the first pair of each file when multiple files are loaded
            show_in_legend = (p_idx == 0) if multiple_files else True
            trace_name = file_name if multiple_files else f"Pair {pair}"

            fig_3d.add_trace(
                go.Scatter3d(
                    x=pair_df["Dim1"],
                    y=pair_df["Dim2"],
                    z=pair_df["Layer"],
                    mode="lines+markers",
                    name=trace_name,
                    legendgroup=file_name if multiple_files else f"Pair {pair}",
                    showlegend=show_in_legend,
                    line=dict(color=color, width=3),
                    marker=dict(color=color, size=3),
                    hovertext=[f"File: {file_name}<br>Pair: {pair}<br>Layer: {int(l)}" for l in pair_df["Layer"]],
                    hoverinfo="text+x+y+z",
                )
            )

    fig_3d.update_layout(
        title=f"Interactive 3D Trajectory in Latent Space ({method_title})",
        scene=dict(
            xaxis_title=dim1_label,
            yaxis_title=dim2_label,
            zaxis_title="Model Layer",
        ),
    )

    html_3d_name = f"interactive_3d_latent_space_{file_suffix}.html"
    fig_3d.write_html(html_3d_name)

    print("\nSuccess! Generated the following output files:")
    print(f"  1. {png_2d_name}")
    print(f"  2. {png_3d_name}")
    print(f"  3. {html_3d_name}")


if __name__ == "__main__":
    main()














