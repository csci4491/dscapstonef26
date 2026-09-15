import os
import sys
import argparse
import numpy as np
import pandas as pd

METADATA_COLS = {"layer", "pair_index", "hypothesis", "conclusion"}

def load_vector_csv(filepath: str) -> tuple[pd.DataFrame, list[str]]:
    """Loads and validates an embedding diff CSV file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")

    df = pd.read_csv(filepath)
    normalized_cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=normalized_cols)

    if "layer" not in df.columns:
        raise ValueError("Missing required 'layer' column in CSV.")

    # Identify all dimension feature columns (e.g., dim_0, dim_1, ...)
    dim_cols = [c for c in df.columns if c.startswith("dim_") or c not in METADATA_COLS]
    # Filter only numeric dimension columns
    dim_cols = [c for c in dim_cols if np.issubdtype(df[c].dtype, np.number)]

    if not dim_cols:
        raise ValueError("No numeric embedding vector dimensions found in CSV.")

    return df, dim_cols


def compute_layer_statistics(df: pd.DataFrame, dim_cols: list[str]) -> pd.DataFrame:
    """
    Computes directional consistency, magnitude distributions,
    and dispersion statistics grouped by transformer layer.
    """
    summary_rows = []
    layers = sorted(df["layer"].unique())

    for layer in layers:
        layer_df = df[df["layer"] == layer]
        vectors = layer_df[dim_cols].to_numpy(dtype=np.float64)  # Shape: (N, D)
        n_samples, dim = vectors.shape

        # 1. L2 Norm distributions ||Δv||
        norms = np.linalg.norm(vectors, axis=1)
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))
        min_norm = float(np.min(norms))
        max_norm = float(np.max(norms))

        # 2. Mean translation vector (centroid)
        centroid = np.mean(vectors, axis=0)
        centroid_norm = float(np.linalg.norm(centroid))

        # 3. Directional Consistency / Collinearity
        # Unit-normalized vectors for angular evaluation
        non_zero_mask = norms > 1e-12
        if np.any(non_zero_mask):
            unit_vectors = vectors[non_zero_mask] / norms[non_zero_mask, None]
            
            # Mean Pairwise Cosine Similarity (sampled if N > 500 for efficiency)
            if len(unit_vectors) > 1:
                if len(unit_vectors) > 500:
                    sample_indices = np.random.choice(len(unit_vectors), size=500, replace=False)
                    sub_units = unit_vectors[sample_indices]
                else:
                    sub_units = unit_vectors
                
                cos_sim_matrix = np.dot(sub_units, sub_units.T)
                # Extract upper triangle without diagonal self-similarity
                triu_indices = np.triu_indices_from(cos_sim_matrix, k=1)
                pairwise_cos = cos_sim_matrix[triu_indices]
                mean_cos_sim = float(np.mean(pairwise_cos))
                std_cos_sim = float(np.std(pairwise_cos))
            else:
                mean_cos_sim = 1.0
                std_cos_sim = 0.0

            # Alignment ratio (Rayleigh mean resultant length R): ||mean(v_unit)||
            mean_unit_dir = np.mean(unit_vectors, axis=0)
            directional_coherence = float(np.linalg.norm(mean_unit_dir))
        else:
            mean_cos_sim, std_cos_sim, directional_coherence = 0.0, 0.0, 0.0

        summary_rows.append({
            "Layer": int(layer),
            "Samples": n_samples,
            "Hidden_Dim": dim,
            "Mean_L2_Norm": round(mean_norm, 4),
            "Std_L2_Norm": round(std_norm, 4),
            "Min_L2_Norm": round(min_norm, 4),
            "Max_L2_Norm": round(max_norm, 4),
            "Centroid_Norm": round(centroid_norm, 4),
            "Directional_Coherence": round(directional_coherence, 4),
            "Mean_Pairwise_Cosine": round(mean_cos_sim, 4),
            "Std_Pairwise_Cosine": round(std_cos_sim, 4)
        })

    return pd.DataFrame(summary_rows)


def print_formatted_report(summary_df: pd.DataFrame, filepath: str):
    """Outputs a formatted CLI overview table and analysis."""
    print("\n" + "=" * 95)
    print(f" STATISTICAL OVERVIEW: {os.path.basename(filepath)}")
    print("=" * 95)

    headers = [
        "Layer", "Pairs", "Dim", "Mean Norm", "Std Norm", 
        "Centroid", "Coherence", "Mean CosSim", "Std CosSim"
    ]
    col_keys = [
        "Layer", "Samples", "Hidden_Dim", "Mean_L2_Norm", "Std_L2_Norm",
        "Centroid_Norm", "Directional_Coherence", "Mean_Pairwise_Cosine", "Std_Pairwise_Cosine"
    ]

    row_format = "{:<7} {:<7} {:<6} {:<11} {:<10} {:<10} {:<11} {:<12} {:<10}"
    print(row_format.format(*headers))
    print("-" * 95)

    for _, row in summary_df.iterrows():
        print(row_format.format(
            int(row["Layer"]),
            int(row["Samples"]),
            int(row["Hidden_Dim"]),
            f"{row['Mean_L2_Norm']:.4f}",
            f"{row['Std_L2_Norm']:.4f}",
            f"{row['Centroid_Norm']:.4f}",
            f"{row['Directional_Coherence']:.4f}",
            f"{row['Mean_Pairwise_Cosine']:.4f}",
            f"{row['Std_Pairwise_Cosine']:.4f}"
        ))

    print("-" * 95)

    # Global Diagnostics
    best_layer = summary_df.loc[summary_df["Directional_Coherence"].idxmax()]
    print("\nKey Observations:")
    print(f"• Peak Directional Coherence: Layer {int(best_layer['Layer'])} (Score: {best_layer['Directional_Coherence']:.4f})")
    print(f"• Peak Mean Cosine Similarity: Layer {int(summary_df.loc[summary_df['Mean_Pairwise_Cosine'].idxmax()]['Layer'])}")
    print(f"• Layer with Largest Norm Shift: Layer {int(summary_df.loc[summary_df['Mean_L2_Norm'].idxmax()]['Layer'])} ({summary_df['Mean_L2_Norm'].max():.4f})")


def main():
    parser = argparse.ArgumentParser(description="Statistical Analyzer for Representation Difference Embeddings.")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to the vector differences CSV file.")
    parser.add_argument("--export", "-e", type=str, default=None, help="Optional output path to save summary table CSV.")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path:
        csv_path = input("Enter path to vector differences CSV file: ").strip()

    if not csv_path:
        print("Error: No file path provided.")
        sys.exit(1)

    try:
        df, dim_cols = load_vector_csv(csv_path)
        summary_df = compute_layer_statistics(df, dim_cols)
        print_formatted_report(summary_df, csv_path)

        export_path = args.export
        if not export_path:
            save_choice = input("\nSave summary table to CSV? (y/n, default n): ").strip().lower()
            if save_choice == "y":
                export_path = input("Enter destination filename (default: 'vector_stats_summary.csv'): ").strip()
                if not export_path:
                    export_path = "vector_stats_summary.csv"

        if export_path:
            summary_df.to_csv(export_path, index=False)
            print(f"Summary metrics exported to '{export_path}'.")

    except Exception as err:
        print(f"\nExecution error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()