import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_OPTIONS = {
    "1": ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B (Ultra-lightweight / Fast CPU or Low VRAM)"),
    "2": ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B (Fast experimentation)"),
    "3": ("Qwen/Qwen2.5-3B-Instruct", "3B (Balanced performance)"),
    "4": ("Qwen/Qwen2.5-7B-Instruct", "7B (High accuracy / Standard baseline)"),
    "5": ("Qwen/Qwen2.5-14B-Instruct", "14B (High capacity / Multi-GPU recommended)"),
}

DEFAULT_TOY_DATA = {
    "hypothesis": [
        "I finished my work for the day.",
        "The weather is clear outside."
    ],
    "conclusion": [
        "I just finished my work and I am absolutely ecstatic!",
        "The weather is incredibly glorious today!"
    ]
}


def select_model_id() -> str:
    """Displays a CLI menu allowing the user to select the Qwen model size."""
    print("\n" + "=" * 50)
    print(" Select Qwen Model Size")
    print("=" * 50)
    for key, (model_name, desc) in MODEL_OPTIONS.items():
        print(f" [{key}] {model_name:<30} - {desc}")

    choice = input("\nEnter selection (default: 4): ").strip()

    return MODEL_OPTIONS.get(choice, MODEL_OPTIONS["4"])[0]


def load_dataset() -> pd.DataFrame:
    """Prompts the user for a CSV path or loads default toy pairs."""
    print("\n" + "=" * 50)
    print(" Dataset Selection")
    print("=" * 50)
    csv_path = input("Enter CSV path (press Enter to use default toy pairs): ").strip()

    if csv_path:
        if not os.path.isfile(csv_path):
            print(f"File '{csv_path}' not found. Falling back to toy dataset.")
            return pd.DataFrame(DEFAULT_TOY_DATA)

        df = pd.read_csv(csv_path)
        # Normalize column names for casing/spacing
        df.columns = df.columns.str.strip().str.lower()

        if "hypothesis" not in df.columns or "conclusion" not in df.columns:
            print("CSV missing 'hypothesis' or 'conclusion' columns! Falling back to toy data.")
            return pd.DataFrame(DEFAULT_TOY_DATA)

        # Drop invalid/empty rows
        clean_df = df[["hypothesis", "conclusion"]].dropna().reset_index(drop=True)
        print(f"Successfully loaded {len(clean_df)} pairs from '{csv_path}'.")
        return clean_df

    print("Using default toy hypothesis-conclusion dataset.")
    return pd.DataFrame(DEFAULT_TOY_DATA)


def extract_last_token_hidden_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layer_idx: int
) -> torch.Tensor:
    """
    Extracts the hidden activation of the final token at a given layer
    using a forward hook without relying on global state.
    """
    captured = {}

    def hook(module, inp, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Isolate the final prompt token: shape (hidden_dim,)
        captured["state"] = hidden_states[0, -1].detach().clone()

    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(hook)

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return captured["state"]


def compute_vector_diffs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    layer_range: range
) -> pd.DataFrame:
    """Computes h(conclusion) - h(hypothesis) across specified layers."""
    records = []
    total_steps = len(layer_range) * len(df)
    current_step = 0

    print(f"\nComputing directional vector differences across layers {list(layer_range)}...")

    for layer in layer_range:
        for idx, row in df.iterrows():
            hyp_text = str(row["hypothesis"])
            con_text = str(row["conclusion"])

            h_con = extract_last_token_hidden_state(model, tokenizer, con_text, layer)
            h_hyp = extract_last_token_hidden_state(model, tokenizer, hyp_text, layer)

            # Vector difference: delta_v = h_conclusion - h_hypothesis
            diff = (h_con - h_hyp).to(torch.float32).cpu().tolist()

            record = {
                "Layer": layer,
                "Pair_Index": idx,
                "Hypothesis": hyp_text,
                "Conclusion": con_text
            }
            # Append embedding coordinates
            for dim_idx, val in enumerate(diff):
                record[f"dim_{dim_idx}"] = val

            records.append(record)
            current_step += 1
            print(f"\rProgress: [{current_step}/{total_steps}] pairs processed", end="", flush=True)

    print("\nExtraction complete.")
    return pd.DataFrame(records)


def main():
    # 1. User selects model size
    model_id = select_model_id()
    print(f"\nInitializing {model_id}...")

    # Set compute precision
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.eval()

    # 2. User selects dataset
    df = load_dataset()

    # 3. Configure layer span
    total_layers = len(model.model.layers)
    default_start, default_end = max(0, total_layers // 3), min(total_layers, (total_layers // 3) + 6)
    
    layer_input = input(
        f"\nEnter layer range 'start,end' (Available: 0 to {total_layers - 1}, default {default_start},{default_end}): "
    ).strip()

    if layer_input and "," in layer_input:
        try:
            start_l, end_l = map(int, layer_input.split(","))
            layers = range(start_l, min(end_l + 1, total_layers))
        except ValueError:
            print("Invalid range input. Falling back to default range.")
            layers = range(default_start, default_end + 1)
    else:
        layers = range(default_start, default_end + 1)

    # 4. Compute vector differences
    diff_df = compute_vector_diffs(model, tokenizer, df, layers)

    # 5. Export output
    output_filename = input("\nEnter output CSV name (default: 'vector_diffs.csv'): ").strip()
    if not output_filename:
        output_filename = "vector_diffs.csv"

    diff_df.to_csv(output_filename, index=False)
    print(f"Results saved to '{output_filename}'. Shape: {diff_df.shape}")


if __name__ == "__main__":
    main()