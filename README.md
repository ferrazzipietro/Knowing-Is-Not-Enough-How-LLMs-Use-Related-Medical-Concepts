# Knowing You Might Need It Is Not Enough

Code accompanying the anonymous short-paper submission **“Knowing You Might Need It Is Not Enough: How LLMs Use Related Medical Concepts in Clinical Note Information Extraction.”**

This repository evaluates instruction-tuned language models on clinical-note information extraction framed as question answering over case report form (CRF) items. In particular, the experiments study whether supplying the value of a related medical concept helps a model extract a target concept, and whether the source and correctness of that related value matter.

> **Anonymity notice**
>
> This repository is currently anonymized for peer review. Author names, affiliations, permanent dataset/model identifiers, and a final citation will be added after the review period.

## Repository contents

```text
.
├── data/
│   └── crf_schema.json              # CRF items, answer types, and clinical groups
├── src/
│   ├── prompt.py                    # Three system-prompt variants
│   └── training/                    # Dataset, model, trainer, and metric utilities
├── pairs_correlation/              # Related-item correlation analysis and diagnostics
├── e_zero_shot.py                   # Main zero-shot evaluation entry point
├── e_zero_shot_seeting_0_1_2.sh     # Example launcher for Settings 0–2
├── e_zero_shot_setting_3.sh         # Example launcher for Setting 3
└── summarize_zero_shot.py           # Export completed runs to an XLSX table
```

The evaluation runs every requested setting with each of the three prompts defined in `src/prompt.py`. It writes validation and test metrics as well as example-level predictions.

## Experimental settings

The code supports four settings:

| Setting | Input to the model |
| --- | --- |
| `setting_0` | The note, target CRF item, and valid answer options, using the dataset's Setting 0 configuration. |
| `setting_1` | The same extraction format, using the dataset's Setting 1 configuration. |
| `setting_2` | Setting 1 plus the gold value of a related/helper CRF item. |
| `setting_3` | Setting 1 plus a model-predicted value of the related/helper CRF item. |

Settings 0 and 1 differ in how examples are constructed in the accompanying dataset; Settings 2 and 3 isolate the effect of providing related-concept information. See the paper for the full construction procedure and experimental rationale.

## Setup

The experiments are intended for a CUDA-capable environment. A recent Python 3 version is recommended. Install PyTorch for your CUDA version, followed by the packages imported by the evaluation code:

```bash
python -m venv .venv
source .venv/bin/activate

# Install the appropriate PyTorch build first: https://pytorch.org/get-started/locally/
pip install transformers datasets evaluate accelerate peft python-dotenv huggingface-hub numpy tqdm scikit-learn pandas scipy matplotlib seaborn openpyxl
```

Some models may additionally require `flash-attn`, `sentencepiece`, or model-specific dependencies. Flash Attention is disabled by default in `e_zero_shot.py`.

Create a `.env` file containing a Hugging Face access token:

```text
HF_TOKEN=your_huggingface_token
```

The token may be required both for the anonymous dataset and for gated model repositories. Do not commit `.env` or any credentials.

## Data

The default dataset identifier used by the scripts is:

```text
anonymousubmission/notes_with_related_items
```

The evaluator expects `dev` and `test` splits and loads one of the dataset configurations `setting_0`, `setting_1_2`, or `setting_3`. At minimum, examples use the following fields:

- `text`: clinical note text;
- `crf_item`: target CRF concept;
- `options`: allowed output values;
- `label`: gold target value;
- `helper_crf_item` and `helper_label`: related concept and its gold value for Setting 2;
- a model/setting/prompt-specific prediction column for Setting 3.

The complete CRF definition is provided in `data/crf_schema.json`. Clinical notes are not stored in this repository. Access remains subject to the terms and restrictions of the source dataset.

## Running zero-shot evaluation

From the repository root, evaluate Settings 0–2 with a Hugging Face model identifier:

```bash
python e_zero_shot.py \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --val_data_path anonymousubmission/notes_with_related_items \
  --val_data_split dev \
  --test_data_split test \
  --task_type_list setting_0 setting_1 setting_2 \
  --per_device_eval_batch_size 4
```

For a quick smoke run, limit the number of examples:

```bash
python e_zero_shot.py \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --task_type_list setting_0 \
  --max_n_examples_val 8 \
  --max_n_examples_test 8 \
  --per_device_eval_batch_size 1
```

Setting 3 consumes helper predictions generated in an earlier run. The relevant prediction column must already be present in the `setting_3` dataset configuration:

```bash
python e_zero_shot.py \
  --model_name_or_path google/gemma-3-27b-it \
  --val_data_path anonymousubmission/notes_with_related_items \
  --task_type_list setting_3 \
  --setting_for_example_in_setting_3 setting_1 \
  --per_device_eval_batch_size 4
```

The shell scripts provide cluster-oriented examples. Update their model, cache, and data paths before running them. Note that `e_zero_shot_seeting_0_1_2.sh` retains the original filename spelling.

## Outputs

Results are written under:

```text
data/zero_shot/<model>/<setting>/system_prompt_<n>/
```

Each completed run contains:

- `validation_metrics.json`
- `test_metrics.json`
- `validation_predictions.json`
- `test_predictions.json`

Prediction records include the CRF item, gold label, decoded prediction, and normalized prediction used by the metrics. Metrics include aggregate scores and, where supported, per-item and per-class scores.

Create a single Excel workbook from all completed runs with:

```bash
python summarize_zero_shot.py \
  --input-dir data/zero_shot \
  --output data/zero_shot_summary.xlsx
```

Use `--skip-model MODEL_DIRECTORY_NAME` more than once to omit selected models.

## Computing related-item correlations

The scripts in `pairs_correlation/` identify related CRF item-value pairs from the annotated training notes. `correlation_matrix.py` represents every `(item, value)` as a binary variable and ranks pairs by the absolute phi coefficient. Items that allow repeated measurements are split into `[first]` and `[following]` pseudo-items; missing annotations are represented as `__unknown__`.

The script currently reads the schema and training annotations from the paths configured near the top of `pairs_correlation/correlation_matrix.py`. Point `SCHEMA_PATH`, `DATA_PATH`, and `OUT_DIR` to the corresponding local files, then run from the repository root:

```bash
python pairs_correlation/correlation_matrix.py \
  --min_phi 0.3 \
  --max_p 0.05 \
  --n_bins 5 \
  --min_support 10 \
  --min_cooccurrence 5 \
  --save_full
```

The analysis writes the filtered pairs to `correlation_pairs_n_bins_<n>.xlsx`, an optional unfiltered table to `correlation_pairs_full.xlsx`, and a heatmap of the highest-ranked pairs to `correlation_heatmap.png`. Additional controls include `--group_filter {all,same,cross}`, `--unknown_filter {none,both,all}`, and `--top_k`.

The directory also contains supporting diagnostics:

- `sweep_correlation_params.py` measures how the selected pair set changes across correlation thresholds, support requirements, and numeric bin counts;
- `check_binnable_types.py` and `check_numeric_sparsity.py` explain whether numeric binning can affect the selected items;
- `analyse_unknown_by_position.py` checks whether `unknown` predictions become more frequent later in long structured generations;
- `sanity_check_train.py` evaluates the correlation-consistency scorer on the training labels and requires the scorer and its generated correlation workbook.

## Reproducibility notes

- Model outputs can vary with hardware, library versions, and model revisions. Record these alongside reported results.
- The evaluator uses greedy next-token predictions through the custom trainer rather than free-form text generation.
- Prompt variants are evaluated separately and preserved in the output directory structure.
- Setting 3 is tied to the model, source setting, and prompt used to produce its helper predictions; keep these identifiers aligned with the dataset column names.

## Citation

Citation information is intentionally withheld during anonymous review. If you use this repository, please cite the paper once the de-anonymized reference becomes available.

## License

A license has not yet been included in this anonymous release. Unless a license is added, the code should be treated as available for review and reproducibility inspection only; no broader permission is granted.
