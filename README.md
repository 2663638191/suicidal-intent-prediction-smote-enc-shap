# Interpretable machine learning for suicidal ideation screening

[简体中文](README_zh-CN.md)

Source code and input definitions for a university mental health screening study. The repository provides the complete Chinese analysis implementation and English entry points, labels and export tools.

## Contents

| Location | Contents |
|---|---|
| [code/en](code/en/) | English commands and output tools, with the computational implementation |
| [code/zh](code/zh/) | Chinese source code |
| [data/schema](data/schema/) | Empty input headers and field dictionaries |
| [docs/data.md](docs/data.md) | Input definitions and data availability |
| [docs/reproducibility.md](docs/reproducibility.md) | Installation, commands and checks |
| [tests](tests/) | Checks using public metadata and synthetic inputs |

## Quick start

Use Python 3.13. Create and activate a virtual environment, then install the recorded dependencies:

```bash
git clone https://github.com/2663638191/suicidal-intent-prediction-smote-enc-shap.git
cd suicidal-intent-prediction-smote-enc-shap
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux. Then install the dependencies:

```bash
python -m pip install -r requirements.txt
python code/en/run_analysis.py --help
```

With authorised inputs matching the [input schema](data/schema/input_schema.json), run:

```bash
python code/en/run_analysis.py --fit --data-dir data/private/original --output results/generated --workers 4
```

For Chinese commands and other output options, see the [code documentation](code/zh/运行说明.md) and [usage guide](docs/reproducibility.md).

## Data availability

Individual-level screening data are not publicly available under the data provider’s authorisation and confidentiality restrictions. The public header and dictionaries describe the expected input structure; they do not constitute a dataset. Reproducing the paper’s empirical estimates requires authorised access to the original data.

## Checks

```bash
python -m unittest discover -s tests -v
```

The public checks use source files, metadata and synthetic inputs. The analysis code retains its table and figure generation functions; empirical output files and manuscript artwork are outside the current public repository scope.

## Citation and licence

Use [CITATION.cff](CITATION.cff) or GitHub’s **Cite this repository** entry and record the commit used. The code is distributed under the existing [MIT licence](LICENSE). See [CONTRIBUTING.md](CONTRIBUTING.md) for changes and [CHANGELOG.md](CHANGELOG.md) for repository updates.
