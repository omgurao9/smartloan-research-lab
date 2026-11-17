# SmartLoan Research Lab

A reproducible research workspace for benchmarking, optimizing, calibrating, and auditing tabular machine-learning models for the supplied loan-status dataset.

> **Research boundary:** The available target is `loan_status`. Its exact provenance and outcome semantics must be verified. Until then, this project predicts the recorded dataset label only; it must not be described as future default, repayment, or credit-loss prediction.

## Research principles

- Audit the data and target-feature relationships before model selection.
- Keep preprocessing inside model pipelines and cross-validation folds.
- Use identical, documented splits for fair comparisons.
- Use validation data for model selection and threshold decisions.
- Keep the final test set sealed until the protocol is frozen.
- Report discrimination, calibration, subgroup metrics, uncertainty, runtime, and limitations—not accuracy alone.
- Track experiments with configurations and machine-readable results.

## Quick start (Windows / Python 3.12)

1. Copy the original dataset from the previous SmartLoan project to `data/raw/loan_data.csv`. The dataset is deliberately not committed.
2. Create an environment and install the core dependencies:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```

3. Run the initial audit:

   ```powershell
   python -m research_lab.audit --input data/raw/loan_data.csv --output reports/data_audit
   ```

4. Run tests:

   ```powershell
   pytest
   ```

## Repository layout

```text
configs/                 Versioned experiment configurations
data/raw/                Local source data (gitignored)
data/processed/          Local derived data (gitignored)
docs/                    Research protocol and methodology
experiments/             Experiment registry and result conventions
ml/research_lab/         Reusable research code
reports/                 Generated local reports (selected summaries may be committed)
scripts/                 Reproducible command-line entry points
tests/                   Automated tests
```

## Local loan application (React + FastAPI)

The repository includes a local applicant assessment application:

- React + Vite frontend in `frontend/`
- FastAPI service in `backend/main.py`
- Selected model: sigmoid-calibrated LightGBM, full 13-feature configuration
- The API fits the frozen model on train + validation rows only; it does not load test rows for fitting.
- The UI includes Assessment, Model Lab, What-if Studio, and Research Notes.

Run from the repository root in two PowerShell terminals:

**Terminal 1 — backend**

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[boosting,web]"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

**Terminal 2 — frontend**

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The API health endpoint is `http://127.0.0.1:8000/api/health`.

The first backend startup trains and caches the calibrated LightGBM using 38,242 training + validation rows, so allow it time to initialize. The raw dataset and frozen split must exist locally.

## Current status

**Phase 1 — foundation and data audit.** The first milestone establishes project conventions and a reproducible audit. Model benchmarking begins only after the data audit and split protocol are reviewed.

## Reusing the previous project

The earlier `smartloan-intelligence` repository remains the source of the starter dataset schema, cleaning rule, feature pipeline, baseline metrics, and initial findings. Copy the raw CSV locally; do not copy old test metrics into new experiment results. New runs must be regenerated under this repository's documented protocol.

## Data and target caveats

The prior audit found a strong one-way relationship between `previous_loan_defaults_on_file` and `loan_status`: all records with target class 1 had previous-default value `No`, while the `Yes` group contained no class-1 records. This is a research flag, not proof of leakage. Establish feature timing, provenance, and label semantics before deciding whether it belongs in the primary model.

## License

Research code is provided for educational and experimental use. Dataset rights and model-specific licenses must be checked separately before redistribution or commercial use.
