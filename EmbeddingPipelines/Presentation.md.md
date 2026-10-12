# Embedding Pipelines

## Diff. Embeddings

### `CalculateBGEEmbeddedDifferents.py`

This script takes in a CSV of hypothesis and conclusion pairs, then embeds them into BGE's latent space (extracting the pooled `[CLS]` token representation). It then calculates the difference between these two vectors to create the *Difference Vector*:

$$\Delta v = h_{\text{conclusion}} - h_{\text{hypothesis}}$$

#### Output Dataset Schema

| Column Name | Data Type | Description | Range / Example |
| :--- | :--- | :--- | :--- |
| `Layer` | `int` | Model layer index where the hidden activations were extracted. | `0` to $L - 1$ |
| `Pair_Index` | `int` | Zero-indexed identifier corresponding to the input dataset row. | `0` to $N - 1$ |
| `Hypothesis` | `string` | Raw text of the input hypothesis/premise. | `"We apply dropout regularization..."` |
| `Conclusion` | `string` | Raw text of the input conclusion/result. | `"The model demonstrates improved..."` |
| `dim_0` ... `dim_{D-1}` | `float` | Component coordinates of the vector difference: $\Delta v = h_{\text{conclusion}} - h_{\text{hypothesis}}$. | Floating-point values (e.g., `-0.0142`, `0.0891`) |

---

### `CalculateQwenEmbeddedDifferents.py`

Similar to the BGE script, this script also takes in a CSV of hypothesis and conclusion pairs; however, this script loads the selected Qwen model and tokenizer, along with the chosen layers to embed at. Since the latent space differs from layer to layer within the LLM, this can be used to visualize how the vector difference changes throughout the model's *thinking* process.

#### Supported Model Dimensions

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

---

### `SummarizeEmbeddings.py`

This is a summary script for the output CSV. It calculates statistical metrics across layers to provide insight into how vector differences behave at scale.

#### Statistical Summary Metrics Table (`vector_stats_summary.csv`)

| Column / Metric | Formulation | Description & Interpretation | Range / Expected Values |
| :--- | :--- | :--- | :--- |
| `Layer` | $l$ | Model transformer layer where activations were hooked. | `0` to $L - 1$ |
| `Samples` | $N$ | Total number of hypothesis-conclusion pairs evaluated at this layer. | $N \ge 1$ |
| `Hidden_Dim` | $D$ | Latent activation dimension ($d_{\text{model}}$) of the model architecture. | `384` to `5120` (model-dependent) |
| `Mean_L2_Norm` | $\frac{1}{N}\sum_{i=1}^N \|\Delta v_i\|_2$ | Average Euclidean magnitude of the translation vector $\Delta v = h_{\text{conclusion}} - h_{\text{hypothesis}}$. | $\ge 0$ (higher indicates larger semantic displacement) |
| `Std_L2_Norm` | $\sqrt{\frac{1}{N}\sum_{i=1}^N (\|\Delta v_i\|_2 - \mu_{\text{norm}})^2}$ | Standard deviation of displacement lengths across pairs; reflects scale stability. | $\ge 0$ |
| `Min_L2_Norm` | $\min_i \|\Delta v_i\|_2$ | Smallest displacement observed across all evaluated pairs. | $\ge 0$ |
| `Max_L2_Norm` | $\max_i \|\Delta v_i\|_2$ | Largest displacement observed across all evaluated pairs. | $\ge 0$ |
| `Centroid_Norm` | $\|\frac{1}{N}\sum_{i=1}^N \Delta v_i\|_2$ | Magnitude of the unnormalized centroid vector $\overline{\Delta v}$. | High values indicate non-canceling directional drift. |
| `Directional_Coherence` | $\|\frac{1}{N}\sum_{i=1}^N \frac{\Delta v_i}{\|\Delta v_i\|_2}\|_2$ | Rayleigh mean resultant length ($R$) evaluating directional concentration. | $[0, 1]$ (values near $1.0$ indicate shared trajectory) |
| `Mean_Pairwise_Cosine` | $\frac{2}{N(N-1)}\sum_{i < j} \frac{\Delta v_i \cdot \Delta v_j}{\|\Delta v_i\|_2 \|\Delta v_j\|_2}$ | Average collinearity between distinct trajectory pairs. | $[-1, 1]$ ($1.0 =$ identical direction, $0 =$ orthogonal) |
| `Std_Pairwise_Cosine` | $\sigma(\cos(\theta_{i,j}))$ | Spread of pairwise angular alignment across dataset samples. | $\ge 0$ |

#### Formatted CLI Overview Report Metrics

| Metric Column | CLI Header | Description |
| :--- | :--- | :--- |
| `Layer` | `Layer` | Transformer layer index |
| `Samples` | `Pairs` | Number of hypothesis-conclusion pairs evaluated |
| `Hidden_Dim` | `Dim` | Latent activation dimension ($D$) |
| `Mean_L2_Norm` | `Mean Norm` | Mean Euclidean displacement magnitude |
| `Std_L2_Norm` | `Std Norm` | Standard deviation of displacement magnitude |
| `Centroid_Norm` | `Centroid` | Magnitude of the average displacement vector |
| `Directional_Coherence` | `Coherence` | Rayleigh mean resultant length ($R \in [0, 1]$) |
| `Mean_Pairwise_Cosine` | `Mean CosSim` | Mean cosine similarity across trajectory pairs |
| `Std_Pairwise_Cosine` | `Std CosSim` | Standard deviation of pairwise cosine similarity |

---

## Single Embeddings

### `CalculateQwenSingleEmbeddings.py`

This script takes in a CSV of hypothesis and conclusion pairs, embeds each data point separately, and returns two CSV files: one containing the embedded hypotheses (`x_hypothesis`) and one containing the embedded conclusions (`y_conclusion`).

#### Single Embedding Dataset Schema

| Column Name | Data Type | Description | Example / Range |
| :--- | :--- | :--- | :--- |
| `Layer` | `int` | Model layer index where hidden activations were extracted. | `0` to $L - 1$ |
| `Pair_Index` | `int` | Row index from the input dataset. | `0` to $N - 1$ |
| `Paper_ID` / `Source_Row` | `string` / `int` | Optional paper identifier or provenance row from source dataset. | `"2301.12345"`, `42` |
| `x_Hypothesis` / `y_conclusion` | `string` | Raw text of the evaluated hypothesis or conclusion. | `"The weather is clear outside."` |
| `dim_0` ... `dim_{D-1}` | `float` | Component coordinates of the single activation vector extracted from the final token. | Floating-point values (e.g., `0.0215`, `-0.0531`) |
