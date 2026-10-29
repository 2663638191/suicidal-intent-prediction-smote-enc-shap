# English code entry points

Run commands from the repository root with Python 3.13 and the versions in `requirements.txt`. Source field names remain Chinese; exported labels use British English. Input fields are documented in `data/schema`.

## Run with authorised inputs

Name the five input workbooks `2021.xlsx` through `2025.xlsx` and place them in `data/private/original`.

```powershell
python code/en/run_analysis.py --fit --data-dir data/private/original --output results/generated/en --workers 4
```

Use a fresh output directory when inputs or source fingerprints change. The command writes generated outputs locally; original data, fitted objects and private caches are excluded from the public repository.

## Reuse an authorised local run

```powershell
python code/en/run_analysis.py --source-results /path/to/authorised/run --output results/generated/en
```

The source must contain the original-language tables in `表格` or `tables` and `私密记录_禁止公开/绘图固定输入.joblib`. Without the cache, redraw stops before creating outputs. Add `--skip-workbook` to omit the workbook or `--dependency-dir` to use an existing Python package directory.

## Workbook export and output checks

The exporter uses `openpyxl` when the optional Node.js workbook package is unavailable. Set `WORKBOOK_EXPORT_BACKEND=openpyxl` to select the Python backend directly.

```powershell
python code/en/export_workbook.py results/generated/en
python code/en/validate_english_results.py --source-results /path/to/authorised/run --output results/generated/en
```

Workbook export reads `reproducibility/workbook_input.json` from the local output. Validation compares the supplied source and English CSV cells, checks metric identities and the expected figure inventory, and checks a workbook or figure-text sidecar when present. It uses no stored empirical baseline. Add `--report /path/to/report.json` to save a report.
