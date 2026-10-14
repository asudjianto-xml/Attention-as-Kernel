# Understanding Cross-Attention Transformers

A tutorial through weighted averages for practicing data scientists, by Agus Sudjianto.

An attention head computes a weighted average. Writing its exponential dot-product scores as a kernel gives exactly the same calculation. This tutorial checks that identity first, then follows the same average through changes to values, score rules, residual updates and a complete cross-attention block.

[Open the executed notebook](attention_tutorial.ipynb) to read the explanations and view the figures, or follow the instructions below to run it yourself.

## What you will learn

1. Compute dot-product attention and a normalized kernel sum from the same queries, keys and values, and verify their equality.
2. Hold the weights fixed while averaging labels, feature vectors or residuals.
3. Change the score rule separately. Gaussian attention generally differs from plain dot-product attention; the code checks the required key-norm bias.
4. Use context self-attention and query cross-attention for successive residual corrections.
5. Put the average inside a block with multiple heads, output projections, skip connections, layer normalization and a feed-forward network.
6. Train the restricted residual model and check its use of context labels.

KRR comparisons, validation grids and spectral analysis are supporting appendices. Throughout, `Q`, `K` and `V` denote projected matrices; `A_context` and `A_query` denote normalized attention weights. A kernel supplies similarities, while normalization turns those similarities into averaging weights.

## How attention connects to kernel smoothing

For one query, dot-product attention assigns each key a score and averages its value:

$$
a_i = \frac{\exp(q^\top k_i/\sqrt{d_k})}{\sum_j \exp(q^\top k_j/\sqrt{d_k})},
\qquad o = \sum_i a_i v_i.
$$

Calling the numerator $\kappa(q,k_i)$ writes the same calculation as a normalized kernel sum. No Gaussian kernel is needed for this identity. If the values are observed labels, this is Nadaraya–Watson regression with the exponential dot-product similarity. Vector values give a weighted average of vectors.

Gaussian scores, $-\|q-k_i\|^2/T$, are introduced later as another choice. At $T=2\sqrt{d_k}$, their weights equal those of scaled dot-product scores with an added key bias, $-\|k_i\|^2/(2\sqrt{d_k})$. Plain dot-product scores generally give different weights; the bias cancels when all keys have the same norm.

The notebook changes the values and the score rule in separate steps. It then uses residual values to explain repeated corrections before showing the average inside a complete cross-attention transformer block. A full transformer learns its scores and values; its layers do not automatically implement residual Nadaraya–Watson updates.

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

The notebook first checks that dot-product softmax and the normalized exponential kernel give identical weights and outputs. It then checks that Gaussian scores and biased dot-product scores give the same weights, that the token network matches iterated NW before training, and that context reordering and query batching preserve predictions. It also constructs an example where positive attention weights produce diverging residuals.

The training comparison uses three seeds, the same episodes and optimizer for each initialization, and validation-selected checkpoints. Test functions have new coefficients from the same family used in training. This is a teaching experiment, not evidence of transfer to an unseen function family. The optional housing example is a small random split, not a spatial generalization study.

Query labels never enter prediction tokens. Preprocessing and context centering use context data. Validation chooses parameters and checkpoints; test labels are used only for evaluation.

## Files

| File | Purpose |
|---|---|
| `attention_tutorial.ipynb` | Explanations, code, and numerical checks |
| `attention_tutorial.py` | Script version of all notebook calculations |
| `execute_notebook.py` | Runs the notebook from a fresh kernel and saves an executed copy |
| `requirements.txt` | Python dependencies |

The notebook includes executed plots for viewing on GitHub. Separate generated result files, datasets, papers and LaTeX sources are not included. Package versions and elapsed runtime are recorded in each run's `results.json`. Dependency versions are not locked; small numerical differences between environments are possible. The tutorial was tested with NumPy 2.4.6, PyTorch 2.12.0 (CPU execution), Matplotlib 3.11.1, and scikit-learn 1.9.0.

## Theoretical paper

For the detailed theory, see Agus Sudjianto and Ming Yuan, *[Multilayer Cross-Attention as Iterated Kernel Smoothing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7611539)* (2026), SSRN working paper 7611539. The tutorial is authored by Agus Sudjianto.

## Review checks

The revised examples search synthetic depths through 1,024 and housing depths through 256, with wider bandwidth grids. Every selected setting at a grid boundary is recorded in `results.json`. Training compares validation-selected checkpoints at 250 and 1,000 updates along the same trajectories. The wider housing search selects an iterated-NW fit with worse test error than one-layer NW; the tutorial reports this result rather than assuming residual correction always helps.

An independent review reproduced the earlier 250-update version to within 1e-9 in spark-venv with NumPy 2.2.6 and PyTorch 2.10. That cross-version check applies to the earlier run; the revised results record their own package versions in `results.json`.

## Two architectures in the notebook

The main sequence is weighted average → exact dot-product/kernel identity → change the values → change the score rule → residual updates → full transformer block.

`SmoothingNetwork` stores the residual calculation in tokens. It has no layer normalization or feed-forward network, and its initial weights compute residual NW exactly. The initialization comparisons train this restricted model.

`CrossAttentionTransformer` then places the same weighted average inside standard multi-head dot-product attention, output projections, layer normalization, feed-forward networks and skip connections. Its untrained weights are used to inspect intermediate calculations, not to report prediction performance. Both examples are context/query models for unordered regression rows, not autoregressive decoders.

## Figures

The first figure computes the same weights and outputs as dot-product attention and as a normalized exponential-kernel sum. A separate numerical check holds queries, keys and values fixed while comparing dot-product and Gaussian scores. The regression figure then uses Gaussian similarity with label values to show how the weights produce a prediction. The full-block figures expose query/key/value projections, the context/query mask, skip connections, weights across heads and layers, and token states. The full-block numerical figures use untrained weights and do not establish prediction quality.

The public notebook includes executed plots for viewing on GitHub. Running it regenerates the figures from the code. Figures are saved under `figures/` inside the configured output directory.
