from __future__ import annotations

import os
import sys
from typing import Optional, Union
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
    "x_hypothesis": [
        "I finished my work for the day.",
        "The weather is clear outside."
    ],
    "y_conclusion": [
        "I just finished my work and I am absolutely ecstatic!",
        "The weather is incredibly glorious today!"
    ],
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


def prepare_dataset(data: Union[str, os.PathLike, pd.DataFrame, None] = None) -> pd.DataFrame:
    """Prepares and validates input data from a DataFrame, CSV file path, or default toy data."""
    if data is None:
        return pd.DataFrame(DEFAULT_TOY_DATA)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (str, os.PathLike)):
        if not os.path.isfile(data):
            raise FileNotFoundError(f"Dataset file not found: {data}")
        df = pd.read_csv(data)
    else:
        raise TypeError(f"Expected DataFrame or str path, got {type(data)}")

    # Normalize column names for case-insensitive matching
    normalized_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=normalized_map)

    # Harmonize alternative column naming conventions
    if "x_hypothesis" not in df.columns and "hypothesis" in df.columns:
        df["x_hypothesis"] = df["hypothesis"]
    if "y_conclusion" not in df.columns and "conclusion" in df.columns:
        df["y_conclusion"] = df["conclusion"]

    if "x_hypothesis" not in df.columns or "y_conclusion" not in df.columns:
        raise ValueError(
            f"Dataset must contain 'x_hypothesis'/'hypothesis' and 'y_conclusion'/'conclusion' columns. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop invalid/empty rows while keeping metadata identifiers
    clean_df = df.dropna(subset=["x_hypothesis", "y_conclusion"]).reset_index(drop=True)
    return clean_df


def load_dataset() -> pd.DataFrame:
    """Prompts the user for a CSV path or loads default toy pairs (for CLI use)."""
    print("\n" + "=" * 50)
    print(" Dataset Selection")
    print("=" * 50)
    csv_path = input("Enter CSV path (press Enter to use default toy pairs): ").strip()

    if csv_path:
        try:
            clean_df = prepare_dataset(csv_path)
            print(f"Successfully loaded {len(clean_df)} pairs from '{csv_path}'.")
            return clean_df
        except Exception as err:
            print(f"Error loading '{csv_path}': {err}. Falling back to toy dataset.")
            return pd.DataFrame(DEFAULT_TOY_DATA)

    print("Using default toy hypothesis-conclusion dataset.")
    return pd.DataFrame(DEFAULT_TOY_DATA)


def load_model_and_tokenizer(
    model_name_or_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Helper to initialize and return tokenizer and causal LM with appropriate precision."""
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)
    device_map = device if device else ("auto" if torch.cuda.is_available() else None)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_id,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    model.eval()
    return model, tokenizer


def resolve_layer_range(
    total_layers: int,
    layer_range: Optional[Union[range, list[int], tuple[int, int], str]] = None
) -> range:
    """Normalizes various layer specification formats into a valid range."""
    if layer_range is None:
        default_start = max(0, total_layers // 3)
        default_end = min(total_layers, default_start + 6)
        return range(default_start, default_end + 1)

    if isinstance(layer_range, range):
        return layer_range
    elif isinstance(layer_range, (list, tuple)):
        if len(layer_range) == 2:
            start_l, end_l = int(layer_range[0]), int(layer_range[1])
            return range(max(0, start_l), min(end_l + 1, total_layers))
        else:
            return range(max(0, min(layer_range)), min(max(layer_range) + 1, total_layers))
    elif isinstance(layer_range, str):
        if "," in layer_range:
            start_l, end_l = map(int, layer_range.split(","))
            return range(max(0, start_l), min(end_l + 1, total_layers))
        else:
            l = int(layer_range)
            return range(max(0, l), min(l + 1, total_layers))

    return range(0, total_layers)


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

    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = getattr(model, "device", torch.device("cpu"))

    inputs = tokenizer(text, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    if "state" not in captured:
        raise RuntimeError(f"Forward hook on layer {layer_idx} did not capture activations.")

    return captured["state"]


def compute_single_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    layer_range: range
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes individual hidden activation vectors for x_Hypothesis and y_conclusion
    across the specified layer range, outputting separate DataFrames.
    """
    hyp_records = []
    con_records = []
    total_steps = len(layer_range) * len(df)
    current_step = 0

    print(f"\nComputing single representation vectors across layers {list(layer_range)}...")

    for layer in layer_range:
        for idx, row in df.iterrows():
            hyp_text = str(row["x_hypothesis"])
            con_text = str(row["y_conclusion"])

            h_hyp = extract_last_token_hidden_state(model, tokenizer, hyp_text, layer)
            h_con = extract_last_token_hidden_state(model, tokenizer, con_text, layer)

            hyp_vec = h_hyp.to(torch.float32).cpu().tolist()
            con_vec = h_con.to(torch.float32).cpu().tolist()

            # Base tracking identifiers
            base_identifiers = {
                "Layer": layer,
                "Pair_Index": idx,
            }
            # Preserve dataset paper identifiers if available
            if "paper_id" in row and pd.notna(row["paper_id"]):
                base_identifiers["Paper_ID"] = row["paper_id"]
            elif "arxiv_id" in row and pd.notna(row["arxiv_id"]):
                base_identifiers["Paper_ID"] = row["arxiv_id"]
            elif "source_row" in row and pd.notna(row["source_row"]):
                base_identifiers["Source_Row"] = row["source_row"]

            # Construct Hypothesis Record
            hyp_record = dict(base_identifiers)
            hyp_record["x_Hypothesis"] = hyp_text
            for dim_idx, val in enumerate(hyp_vec):
                hyp_record[f"dim_{dim_idx}"] = val
            hyp_records.append(hyp_record)

            # Construct Conclusion Record
            con_record = dict(base_identifiers)
            con_record["y_conclusion"] = con_text
            for dim_idx, val in enumerate(con_vec):
                con_record[f"dim_{dim_idx}"] = val
            con_records.append(con_record)

            current_step += 1
            print(f"\rProgress: [{current_step}/{total_steps}] pairs processed", end="", flush=True)

    print("\nExtraction complete.")
    return pd.DataFrame(hyp_records), pd.DataFrame(con_records)


def calculate_qwen_single_embeddings(
    data: Union[str, os.PathLike, pd.DataFrame, None] = None,
    model_name_or_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    layer_range: Optional[Union[range, list[int], tuple[int, int], str]] = None,
    hyp_output_path: Optional[str] = None,
    con_output_path: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exportable pipeline function to compute Qwen latent representation single embeddings
    for both x_Hypothesis and y_conclusion.

    Args:
        data: CSV filepath, in-memory DataFrame, or None (uses toy dataset).
        model_name_or_id: Hugging Face model checkpoint (used if model/tokenizer is not provided).
        model: Optional pre-loaded AutoModelForCausalLM instance.
        tokenizer: Optional pre-loaded AutoTokenizer instance.
        layer_range: Specific layer range, tuple (start, end), or None for middle layers.
        hyp_output_path: Optional destination file path to save x_Hypothesis CSV.
        con_output_path: Optional destination file path to save y_conclusion CSV.
        device: Device string (e.g. 'cuda', 'cpu', 'auto').
        torch_dtype: Compute precision (torch.bfloat16, torch.float32, etc.).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (hyp_df, con_df) containing metadata identifiers and coordinates.
    """
    # 1. Prepare Dataset
    df = prepare_dataset(data)

    # 2. Initialize Model & Tokenizer if not provided
    if model is None or tokenizer is None:
        print(f"Initializing {model_name_or_id}...")
        loaded_model, loaded_tokenizer = load_model_and_tokenizer(
            model_name_or_id=model_name_or_id,
            device=device,
            torch_dtype=torch_dtype
        )
        model = model or loaded_model
        tokenizer = tokenizer or loaded_tokenizer

    # 3. Configure Layers
    total_layers = len(model.model.layers)
    resolved_layers = resolve_layer_range(total_layers, layer_range)

    # 4. Compute Single Embeddings
    hyp_df, con_df = compute_single_embeddings(model, tokenizer, df, resolved_layers)

    # 5. Optional Export
    if hyp_output_path:
        out_dir = os.path.dirname(hyp_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        hyp_df.to_csv(hyp_output_path, index=False)
        print(f"x_Hypothesis embeddings saved to: '{hyp_output_path}' (Shape: {hyp_df.shape})")

    if con_output_path:
        out_dir = os.path.dirname(con_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        con_df.to_csv(con_output_path, index=False)
        print(f"y_conclusion embeddings saved to: '{con_output_path}' (Shape: {con_df.shape})")

    return hyp_df, con_df


# Pipeline alias
run_qwen_single_pipeline = calculate_qwen_single_embeddings


def main():
    # 1. User selects model size
    model_id = select_model_id()

    # 2. User selects dataset
    df = load_dataset()

    # 3. Load model & tokenizer
    print(f"\nInitializing {model_id}...")
    model, tokenizer = load_model_and_tokenizer(model_id)

    # 4. Configure layer span interactively
    total_layers = len(model.model.layers)
    default_start = max(0, total_layers // 3)
    default_end = min(total_layers, default_start + 6)

    layer_input = input(
        f"\nEnter layer range 'start,end' (Available: 0 to {total_layers - 1}, default {default_start},{default_end}): "
    ).strip()

    layers = resolve_layer_range(total_layers, layer_input if layer_input else None)

    # 5. Export destination prompt
    base_prefix = input("\nEnter output prefix / base name (default: 'qwen_single'): ").strip()
    if base_prefix:
        hyp_filename = f"{base_prefix}_x_hypothesis.csv"
        con_filename = f"{base_prefix}_y_conclusion.csv"
    else:
        hyp_filename = "x_hypothesis_embeddings.csv"
        con_filename = "y_conclusion_embeddings.csv"

    # 6. Run pipeline via exportable function
    calculate_qwen_single_embeddings(
        data=df,
        model=model,
        tokenizer=tokenizer,
        layer_range=layers,
        hyp_output_path=hyp_filename,
        con_output_path=con_filename
    )


__all__ = [
    "calculate_qwen_single_embeddings",
    "run_qwen_single_pipeline",
    "compute_single_embeddings",
    "extract_last_token_hidden_state",
    "load_model_and_tokenizer",
    "prepare_dataset",
    "resolve_layer_range",
    "MODEL_OPTIONS",
]


if __name__ == "__main__":
    main()
