# Attention as a Weighted Average

A kernel tutorial for practicing data scientists, by Agus Sudjianto.

An attention head computes a weighted average. Queries and keys determine the weights; values supply what is averaged. This tutorial starts with three numbers and follows that calculation through kernel regression, residual corrections, and a small trainable attention network.

[Open the notebook](attention_tutorial.ipynb)

## What you will learn

1. Interpret queries, keys, values, and softmax through a weighted average.
2. Use Gaussian similarities to compute Nadaraya–Watson (NW) regression.
3. Express Gaussian NW as attention, including the key bias needed for the dot-product form.
4. Use context self-attention and query cross-attention to correct residuals across layers.
5. Select depth using validation data and see why positive weights alone do not guarantee stable residual updates.
6. Compare iterated NW with kernel ridge regression (KRR).
7. Train a four-layer attention network from NW and random initializations.
8. Check whether the trained network uses the pairing between context inputs and labels.

The kernel is the similarity rule. Normalizing its similarities gives the averaging weights. The tutorial keeps this distinction explicit.

## Install

Python 3.10 or newer is required; the tutorial was tested with Python 3.12. All examples run on the CPU. A GPU is not required.

```bash
git clone https://github.com/asudjianto-xml/Attention-as-Kernel.git
cd Attention-as-Kernel
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the notebook

```bash
python -m jupyterlab attention_tutorial.ipynb
```

Select the Python kernel from the environment where you installed the dependencies, then choose **Restart Kernel and Run All Cells**. The notebook contains its own implementations; it does not import files from another project.

To execute it without opening Jupyter:

```bash
python execute_notebook.py
```

This runner uses the same Python interpreter that starts it and saves an executed copy to `outputs/attention_tutorial.executed.ipynb`. It leaves the source notebook unchanged. Each cell has a five-minute timeout.

For an offline run that skips the optional housing example:

```bash
python execute_notebook.py --skip-housing
```

You can choose an output directory:

```bash
python execute_notebook.py --output-dir my-results
```

The runner locates the notebook relative to its own script, so it also works when invoked from another directory.

## Run as a Python script

```bash
python attention_tutorial.py
```

The script contains the same code cells, in the same order, as the notebook. It saves figures without opening plot windows. The notebook is the version to read for the explanations.

Both versions write `results.json` and figures under `outputs/` in the current working directory by default. They create that directory if needed. No username, home directory, repository location, or data path is built into the code.

## Optional California Housing example

The synthetic examples and training experiment require no downloads. California Housing runs only if its data are already in scikit-learn's cache. Otherwise the tutorial prints a skip message and completes normally. It never downloads a dataset automatically.

To obtain the data intentionally, run:

```bash
python -c "from sklearn.datasets import fetch_california_housing; fetch_california_housing()"
```

Then rerun the tutorial. The dataset's default cache is managed by scikit-learn.

These environment variables are optional:

| Variable | Purpose | Default |
|---|---|---|
| `ATTENTION_TUTORIAL_OUTPUT_DIR` | Directory for results and figures | `outputs/` relative to the working directory |
| `ATTENTION_TUTORIAL_DATA_DIR` | California Housing cache directory | scikit-learn's default cache |
| `ATTENTION_TUTORIAL_SKIP_HOUSING` | Set to `1` to skip housing even if cached | Not set |

For a custom data cache, populate that same directory explicitly with `fetch_california_housing(data_home="your-cache-directory")`. The runner's `--output-dir` option takes precedence over the output environment variable.

## What the experiments establish

The notebook checks that Gaussian scores and biased dot-product scores give the same weights, that the token network matches iterated NW before training, and that context reordering and query batching preserve predictions. It also constructs an example where positive attention weights produce diverging residuals.

The training comparison uses three seeds, the same episodes and optimizer for each initialization, and validation-selected checkpoints. Test functions have new coefficients from the same family used in training. This is a teaching experiment, not evidence of transfer to an unseen function family. The optional housing example is a small random split, not a spatial generalization study.

Query labels never enter prediction tokens. Preprocessing and context centering use context data. Validation chooses parameters and checkpoints; test labels are used only for evaluation.

## Files

| File | Purpose |
|---|---|
| `attention_tutorial.ipynb` | Explanations, code, and numerical checks |
| `attention_tutorial.py` | Script version of all notebook calculations |
| `execute_notebook.py` | Runs the notebook from a fresh kernel and saves an executed copy |
| `requirements.txt` | Python dependencies |

Generated results, datasets, papers, and LaTeX sources are not included. Package versions and elapsed runtime are recorded in each run's `results.json`. Dependency versions are not locked; small numerical differences between environments are possible. The tutorial was tested with NumPy 2.4.6, PyTorch 2.12.0 (CPU execution), Matplotlib 3.11.1, and scikit-learn 1.9.0.

## Theoretical paper

For the detailed theory, see Agus Sudjianto and Ming Yuan, *[Multilayer Cross-Attention as Iterated Kernel Smoothing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7611539)* (2026), SSRN working paper 7611539. The tutorial is authored by Agus Sudjianto.

## Review checks

The revised examples search synthetic depths through 1,024 and housing depths through 256, with wider bandwidth grids. Every selected setting at a grid boundary is recorded in `results.json`. Training compares validation-selected checkpoints at 250 and 1,000 updates along the same trajectories. The wider housing search selects an iterated-NW fit with worse test error than one-layer NW; the tutorial reports this result rather than assuming residual correction always helps.

An independent review reproduced the earlier 250-update version to within 1e-9 in spark-venv with NumPy 2.2.6 and PyTorch 2.10. That cross-version check applies to the earlier run; the revised results record their own package versions in `results.json`.
