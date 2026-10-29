# Installation and usage

## Environment

Use Python 3.13 and the dependencies in [requirements.txt](../requirements.txt). Create a virtual environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux. Then install the dependencies:

```bash
python -m pip install -r requirements.txt
python code/en/run_analysis.py --help
```

## Run with authorised inputs

Prepare the workbooks described in [data/schema](../data/schema/) and place them in `data/private/original`, or select their directory with `--data-dir`:

```bash
python code/en/run_analysis.py --fit --data-dir data/private/original --output results/generated --workers 4
```

Choose a fresh output directory for a new run. Generated tables, workbooks, figures and intermediate files are written under that directory. The analysis source retains the configuration and output functions needed to run the implementation.

## Export from an existing authorised run

To regenerate English outputs from an original-language run containing its private plotting cache:

```bash
python code/en/run_analysis.py --source-results /path/to/authorised/run --output results/generated
```

Use `--skip-workbook` to omit workbook export. The Python workbook exporter works with the recorded Python dependencies. Further options are listed by `--help` and in [code/en/README.md](../code/en/README.md). Chinese commands are documented in [code/zh/运行说明.md](../code/zh/运行说明.md).

## Check the public code

```bash
python -m unittest discover -s tests -v
```

The public checks cover source files, input definitions and synthetic inputs. They do not require the original screening dataset. See [data availability](data.md) for the scope of the public inputs.
