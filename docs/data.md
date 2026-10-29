# Input definitions and data availability

Individual-level screening data are not publicly available under the data provider’s authorisation and confidentiality restrictions. The repository supplies an empty input header and field definitions. Reproducing the paper’s empirical estimates requires authorised access to the original data.

## Input files

The code expects annual workbooks named `2021.xlsx` through `2025.xlsx`. The required worksheet names are recorded in [input_schema.json](../data/schema/input_schema.json).

| File | Purpose |
|---|---|
| [input_headers_zh.csv](../data/schema/input_headers_zh.csv) | Original column names, with no observation rows |
| [input_schema.json](../data/schema/input_schema.json) | Workbook and worksheet requirements |
| [source_field_mapping.csv](../data/schema/source_field_mapping.csv) | Source fields, analytical roles and English labels |
| [variable_dictionary_en.csv](../data/schema/variable_dictionary_en.csv) | Predictor labels and code keys |
| [terminology_en_zh.csv](../data/schema/terminology_en_zh.csv) | Corresponding English and Chinese terms |

The English dimension labels are consistent author translations for this project. They do not represent a separately validated English questionnaire. The supplied files document input structure and field meanings rather than a dataset.

Place authorised inputs in `data/private/original`, or supply another directory through `--data-dir`. See the [usage guide](reproducibility.md) for commands.
