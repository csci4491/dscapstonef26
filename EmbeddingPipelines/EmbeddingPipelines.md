# Embedding Pipelines

This directory serves as housing for the embedding pipelines. Both scripts perform the same(ish) process and return CSV files of the shape:


### Dataset Shape

| Column Name | Data Type | Description | Range / Example |
| :--- | :--- | :--- | :--- |
| `Layer` | `int` | Model layer index where the hidden activations were extracted. | `0` to $L - 1$ (varies by model size/depth) |
| `Pair_Index` | `int` | Zero-indexed identifier corresponding to the input dataset row. | `0` to $N - 1$ |
| `Hypothesis` | `string` | Raw text of the input hypothesis/premise. | `"We apply dropout regularization..."` |
| `Conclusion` | `string` | Raw text of the input conclusion/result. | `"The model demonstrates improved..."` |
| `dim_0` ... `dim_{D-1}` | `float` | Component coordinates of the vector difference: $\Delta v = h_{\text{conclusion}} - h_{\text{hypothesis}}$. | Floating-point values (e.g., `-0.0142`, `0.0891`) |

---

### Dataset Dimensions Summary

* **Total Rows:** $M \times N$ *(Number of layers analyzed $\times$ Number of input pairs)*
* **Total Columns:** $4 + D$ *(4 metadata columns $+$ $D$ hidden dimensions)*

| Pipeline | Model Checkpoint | Hidden Dimension ($D$) | Layer Count ($L$) | Output Columns |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5** | `Qwen2.5-0.5B-Instruct` | 896 | 24 | 900 columns (`dim_0` to `dim_895`) |
| **Qwen 2.5** | `Qwen2.5-1.5B-Instruct` | 1536 | 28 | 1540 columns (`dim_0` to `dim_1535`) |
| **Qwen 2.5** | `Qwen2.5-3B-Instruct` | 2048 | 36 | 2052 columns (`dim_0` to `dim_2047`) |
| **Qwen 2.5** | `Qwen2.5-7B-Instruct` | 3584 | 28 | 3588 columns (`dim_0` to `dim_3583`) |
| **Qwen 2.5** | `Qwen2.5-14B-Instruct` | 5120 | 48 | 5124 columns (`dim_0` to `dim_5119`) |
| **BGE** | `BAAI/bge-small-en-v1.5` | 384 | 12 | 388 columns (`dim_0` to `dim_383`) |
| **BGE** | `BAAI/bge-base-en-v1.5` | 768 | 12 | 772 columns (`dim_0` to `dim_767`) |
| **BGE** | `BAAI/bge-large-en-v1.5` | 1024 | 24 | 1028 columns (`dim_0` to `dim_1023`) |
| **BGE** | `BAAI/bge-m3` / `bge-m3-unsupervised` | 1024 | 24 | 1028 columns (`dim_0` to `dim_1023`) |

## Pipeline Architecture

### Qwen (LLM)

1. Load Model Tokenizer (currently using a simple CLI menu)
    - Uses `transformers` to load pre-trained tokenizer
    - Since we are only loading the tokenizer for calculating embeddings, luckily we can run this scipt on __much__ weaker machines as opposed to loading for model weights.
2. Load Dataset into memory
    - shout out pandas
3. Calculate embedding diff for hypothsis -> answer pair
    - 3.1 Embed the hypothesis hV
    - 3.2 Embed the conclusion cV
    - 3.3 Calculate cV - hV

4. Construct Output dataframe
    - once again shout out pandas

### BAAI/BGE-3 (Semantic Model)

_The current pipeline does not support fine training the model on data set before vector calculations_

1. Load Model & Tokenizer (currently using a simple CLI menu)
    - Uses `transformers` to load the pre-trained bi-encoder and tokenizer
    - Lets you pick checkpoints ranging from lightweight small/base variants up to BGE-M3 to leverage its massive 8,192-token context window.

2. Load Dataset into memory
    - shout out pandas

3. Calculate embedding diff for hypothesis -> answer pair
    - 3.1 Embed the hypothesis hV (extracting the pooled `[CLS]` token)
    - 3.2 Embed the conclusion cV
    - 3.3 Calculate cV - hV


4. Construct Output dataframe
    - once again shout out pandas