import numpy as np
import pandas as pd

def run_environment_check():
    print("--- Checking Core Dependencies ---")
    print(f"Numpy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    print("\n--- Simulating Data Pipeline ---")
    # Simulating the target arXiv data structure[cite: 1]
    mock_arxiv_data = {
        "Paper_ID": ["arXiv:2605.05892", "arXiv:2401.12345"],
        "Section": ["Methodology", "Conclusion"],
        "Content": [
            "We propose Flow-based Activation Steering (FLAS)...",
            "FLAS outperforms standard in-context prompting..."
        ]
    }
    
    df = pd.DataFrame(mock_arxiv_data)
    print("Successfully created Pandas DataFrame for paper sections:")
    print(df)
    
    print("\n--- Simulating Vector Math ---")
    
    mock_hypothesis_vector = np.array([0.8, 0.1, 0.4])
    mock_conclusion_vector = np.array([0.7, 0.2, 0.5])
    
    dot_product = np.dot(mock_hypothesis_vector, mock_conclusion_vector)
    norm_a = np.linalg.norm(mock_hypothesis_vector)
    norm_b = np.linalg.norm(mock_conclusion_vector)
    cosine_sim = dot_product / (norm_a * norm_b)
    
    print(f"Successfully calculated mock semantic angle (Cosine Similarity): {cosine_sim:.4f}")
    print("\nSUCCESS: Base numpy and pandas environment is ready.")

if __name__ == "__main__":
    run_environment_check()