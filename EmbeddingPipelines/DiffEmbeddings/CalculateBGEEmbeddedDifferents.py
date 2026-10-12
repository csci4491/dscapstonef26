import os
import sys
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer

MODEL_OPTIONS = {
    "1": ("BAAI/bge-small-en-v1.5", "Small (384D, 12 layers - fast local prototyping)"),
    "2": ("BAAI/bge-base-en-v1.5", "Base (768D, 12 layers - standard English)"),
    "3": ("BAAI/bge-large-en-v1.5", "Large (1024D, 24 layers - English, 512 context limit)"),
    "4": ("BAAI/bge-m3", "M3 (1024D, 24 layers - 8,192 context window, multilingual)"),
    "5": ("BAAI/bge-m3-unsupervised", "M3 Unsupervised (1024D, 24 layers - blank slate without QA bias)"),
}

DEFAULT_TOY_DATA = {
    "hypothesis": [
        "We apply dropout regularization of 0.5 to prevent overfitting in deep transformer layers.",
        "We replace standard dense attention with linear sparse attention to reduce quadratic memory complexity."
    ],
    "conclusion": [
        "The model demonstrates improved validation accuracy with a significant reduction in training variance.",
        "Peak VRAM consumption dropped by 64%, enabling scaling to 32k context windows without throughput degradation."
    ]
}


def select_model_id() -> str:
    """Displays a CLI menu allowing the user to select the BGE model size and checkpoint."""
    print("\n" + "=" * 50)
    print(" Select BGE Model Checkpoint")
    print("=" * 50)
    for key, (model_name, desc) in MODEL_OPTIONS.items():
        print(f" [{key}] {model_name:<28} - {desc}")

    choice = input("\nEnter selection (default: 4): ").strip()

    return MODEL_OPTIONS.get(choice, MODEL_OPTIONS["4"])[0]


def load_dataset() -> pd.DataFrame:
    """Prompts the user for a CSV path or loads default scientific pairs."""
    print("\n" + "=" * 50)
    print(" Dataset Selection")
    print("=" * 50)
    csv_path = input("Enter CSV path (press Enter to use default pairs): ").strip()

    if csv_path:
        if not os.path.isfile(csv_path):
            print(f"File '{csv_path}' not found. Falling back to default data.")
            return pd.DataFrame(DEFAULT_TOY_DATA)

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()

        if "hypothesis" not in df.columns or "conclusion" not in df.columns:
            print("CSV missing 'hypothesis' or 'conclusion' columns! Falling back to default data.")
            return pd.DataFrame(DEFAULT_TOY_DATA)

        clean_df = df[["hypothesis", "conclusion"]].dropna().reset_index(drop=True)
        print(f"Successfully loaded {len(clean_df)} pairs from '{csv_path}'.")
        return clean_df

    print("Using default scientific hypothesis-conclusion dataset.")
    return pd.DataFrame(DEFAULT_TOY_DATA)


def get_encoder_layers(model: AutoModel):
    """Retrieves the transformer layer ModuleList from standard encoder architectures."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Unable to locate transformer layers in the loaded architecture.")


def extract_cls_token_hidden_state(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    text: str,
    layer_idx: int
) -> torch.Tensor:
    """
    Extracts the hidden activation of the [CLS] token at a given layer
    using a forward hook without global state.
    """
    captured = {}

    def hook(module, inp, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # In bidirectional encoders (BERT/RoBERTa/BGE-M3), index 0 holds the [CLS] representation
        captured["state"] = hidden_states[0, 0].detach().clone()

    layers = get_encoder_layers(model)
    target_layer = layers[layer_idx]
    handle = target_layer.register_forward_hook(hook)

    device = next(model.parameters()).device
    # Support extended context where applicable (e.g., 8192 tokens for BGE-M3)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(device)

    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return captured["state"]


def compute_vector_diffs(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    layer_range: range
) -> pd.DataFrame:
    """Computes the relational trajectory h(conclusion) - h(hypothesis) across specified layers."""
    records = []
    total_steps = len(layer_range) * len(df)
    current_step = 0

    print(f"\nComputing directional vector differences across layers {list(layer_range)}...")

    for layer in layer_range:
        for idx, row in df.iterrows():
            hyp_text = str(row["hypothesis"])
            con_text = str(row["conclusion"])

            h_con = extract_cls_token_hidden_state(model, tokenizer, con_text, layer)
            h_hyp = extract_cls_token_hidden_state(model, tokenizer, hyp_text, layer)

            # Relational vector subtraction: Δv = h_conclusion - h_hypothesis
            diff = (h_con - h_hyp).to(torch.float32).cpu().tolist()

            record = {
                "Layer": layer,
                "Pair_Index": idx,
                "Hypothesis": hyp_text,
                "Conclusion": con_text
            }
            for dim_idx, val in enumerate(diff):
                record[f"dim_{dim_idx}"] = val

            records.append(record)
            current_step += 1
            print(f"\rProgress: [{current_step}/{total_steps}] pairs processed", end="", flush=True)

    print("\nExtraction complete.")
    return pd.DataFrame(records)


def main():
    # 1. User selects model size / checkpoint
    model_id = select_model_id()
    print(f"\nInitializing {model_id}...")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Bi-encoders use AutoModel rather than AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()

    # 2. User selects dataset
    df = load_dataset()

    # 3. Configure layer span dynamically based on encoder depth
    layers = get_encoder_layers(model)
    total_layers = len(layers)
    default_start, default_end = max(0, total_layers // 2), min(total_layers, (total_layers // 2) + 4)

    layer_input = input(
        f"\nEnter layer range 'start,end' (Available: 0 to {total_layers - 1}, default {default_start},{default_end}): "
    ).strip()

    if layer_input and "," in layer_input:
        try:
            start_l, end_l = map(int, layer_input.split(","))
            layer_range = range(max(0, start_l), min(end_l + 1, total_layers))
        except ValueError:
            print("Invalid input format. Using default range.")
            layer_range = range(default_start, default_end + 1)
    else:
        layer_range = range(default_start, default_end + 1)

    # 4. Compute vector differences
    diff_df = compute_vector_diffs(model, tokenizer, df, layer_range)

    # 5. Export output
    output_filename = input("\nEnter output CSV name (default: 'bge_vector_diffs.csv'): ").strip()
    if not output_filename:
        output_filename = "bge_vector_diffs.csv"

    diff_df.to_csv(output_filename, index=False)
    print(f"Results saved to '{output_filename}'. Shape: {diff_df.shape}")


if __name__ == "__main__":
    main()