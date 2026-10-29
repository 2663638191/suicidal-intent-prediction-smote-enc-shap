"""Exercise public IO interfaces using temporary files and synthetic values."""
from pathlib import Path
import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code' / 'en'
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE))
import localisation
import run_analysis

HAS_OPENPYXL = importlib.util.find_spec('openpyxl') is not None


class PortableInterfaces(unittest.TestCase):
    def test_metric_validation_uses_synthetic_counts(self):
        from validate_english_results import check_metric_identities
        row = {'Model-strategy key': 'Synthetic A', 'TN': '6', 'FP': '2',
               'FN': '1', 'TP': '3', 'ROC-AUC': '0.73', 'Average precision': '0.61',
               'Threshold': '0.42', 'Sensitivity': '0.75', 'Specificity': '0.75',
               'Records flagged per 1,000 records': str(1000 * 5 / 12)}
        other = dict(row, **{'Model-strategy key': 'Synthetic B'})
        self.assertEqual(check_metric_identities([row, other]), 2)

    def test_metric_validation_rejects_inconsistent_synthetic_sensitivity(self):
        from validate_english_results import check_metric_identities
        row = {'Model-strategy key': 'Synthetic', 'TN': '6', 'FP': '2',
               'FN': '1', 'TP': '3', 'ROC-AUC': '0.73', 'Average precision': '0.61',
               'Threshold': '0.42', 'Sensitivity': '0.5', 'Specificity': '0.75',
               'Records flagged per 1,000 records': str(1000 * 5 / 12)}
        with self.assertRaisesRegex(AssertionError, 'Sensitivity'):
            check_metric_identities([row])

    def test_missing_private_cache_stops_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            process = subprocess.run([sys.executable, str(CODE/'run_analysis.py'),
                                      '--source-results', str(base/'source'),
                                      '--output', str(base/'output')], capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn('private plotting cache', process.stderr)
            self.assertFalse((base/'output').exists())

    def test_fit_requires_five_authorised_workbooks(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            process = subprocess.run([sys.executable, str(CODE/'run_analysis.py'), '--fit',
                                      '--data-dir', str(base/'inputs'), '--output', str(base/'output')],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn('Missing authorised input workbooks', process.stderr)
            self.assertIn('2021.xlsx', process.stderr)
            self.assertIn('2025.xlsx', process.stderr)
            self.assertFalse((base/'output').exists())

    def test_mock_fit_routes_inputs_and_workers_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            inputs, output, source = base/'inputs', base/'output', base/'frozen'
            inputs.mkdir()
            for year in range(2021, 2026):
                (inputs/f'{year}.xlsx').write_bytes(b'Synthetic placeholder; never read as a workbook')
            cache = source/'私密记录_禁止公开'/'绘图固定输入.joblib'
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b'Synthetic placeholder; loaded only by the mock')
            frozen = object()
            loader = types.SimpleNamespace(load=lambda path: frozen)
            figures = types.SimpleNamespace(create_figures=lambda src, out, args: ({}, {'manuscript_files': {}}))
            checks = [{'numeric_or_empty_cells_unchanged': 3}]
            with patch.object(run_analysis, 'fit_original_analysis', return_value=source) as fit, \
                 patch.object(run_analysis, 'configure_dependencies'), \
                 patch.object(localisation, 'export_tables', return_value=(checks, [{'name': 'Synthetic', 'values': [['A'], [1]]}])), \
                 patch.dict(sys.modules, {'joblib': loader, 'english_figures': figures}), \
                 patch.object(sys, 'argv', ['run_analysis.py', '--fit', '--data-dir', str(inputs),
                                           '--output', str(output), '--workers', '3', '--skip-workbook']), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_analysis.main()
            fit.assert_called_once_with(inputs.resolve(), output.resolve(), 3)
            report = json.loads((output/'reproducibility'/'localisation_validation.json').read_text(encoding='utf-8'))
            self.assertTrue(report['model_training_rerun'])  # Requested mode; core was mocked in this test.
            self.assertEqual(report['numerical_cells_preserved'], 3)

    def test_both_source_table_layouts_preserve_numeric_strings(self):
        table = {'Synthetic': ('synthetic', 'Synthetic')}
        rows = [['模型', 'ROC-AUC'], ['逻辑回归', '0.12345678901234567'], ['决策树', '1e-6']]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for folder in ('表格', 'tables'):
                with self.subTest(folder=folder):
                    source = base/folder/'source'
                    (source/folder).mkdir(parents=True)
                    with (source/folder/'Synthetic.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                        csv.writer(stream).writerows(rows)
                    with patch.object(localisation, 'TABLES', table):
                        checks, workbook = localisation.export_tables(source, base/folder/'output')
                    with (base/folder/'output'/'tables'/'synthetic.csv').open(encoding='utf-8-sig', newline='') as stream:
                        actual = list(csv.reader(stream))
                    self.assertEqual([r[1] for r in actual], [r[1] for r in rows])
                    self.assertTrue(checks[0]['passed'])
                    self.assertEqual(workbook[0]['name'], 'Synthetic')

    @unittest.skipUnless(HAS_OPENPYXL, 'Workbook fallback requires optional openpyxl')
    def test_python_workbook_roundtrip_preserves_types_and_precision(self):
        from export_workbook import export_workbook
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            payload = self.write_payload(output, 'reproducibility', 'workbook_input.json')
            report = export_workbook(output)
            self.assertTrue(report['values_preserved'])
            workbook = load_workbook(output/'analysis_results_English.xlsx', data_only=False)
            try:
                self.assertEqual(list(workbook['Synthetic'].values), [tuple(r) for r in payload[0]['values']])
                self.assertEqual(workbook['Synthetic']['D2'].data_type, 's')
                self.assertEqual(workbook['Synthetic']['B2'].number_format, '0.00E+00')
                self.assertEqual(workbook['Synthetic']['C2'].number_format, '0.0000')
                self.assertEqual(workbook['Synthetic'].freeze_panes, 'A2')
            finally:
                workbook.close()

    @unittest.skipUnless(HAS_OPENPYXL and shutil.which('node'), 'Node fallback test requires Node.js and openpyxl')
    def test_node_exporters_fall_back_without_artifact_package(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            env = dict(os.environ, ANALYSIS_PYTHON=sys.executable, WORKBOOK_EXPORT_BACKEND='auto',
                       ARTIFACT_NODE_MODULES=str(base/'package-not-installed'))
            for language, script, record_dir, input_name, filename in [
                ('en', CODE/'export_workbook.mjs', 'reproducibility', 'workbook_input.json', 'analysis_results_English.xlsx'),
                ('zh', ROOT/'code'/'zh'/'工作簿导出.mjs', '复现记录', '工作簿表格输入.json', '论文结果汇总.xlsx'),
            ]:
                with self.subTest(language=language):
                    output = base/language
                    self.write_payload(output, record_dir, input_name)
                    process = subprocess.run([shutil.which('node'), str(script), str(output)],
                                             env=env, capture_output=True, text=True, encoding='utf-8')
                    self.assertEqual(process.returncode, 0, process.stderr)
                    self.assertTrue((output/filename).is_file())
                    report = json.loads((output/record_dir/'workbook_export_checks.json').read_text(encoding='utf-8'))
                    self.assertEqual(report['backend'], 'openpyxl')
                    self.assertTrue(report['values_preserved'])

    @staticmethod
    def write_payload(output, record_dir, input_name):
        payload = [{'name': 'Synthetic', 'values': [
            ['Count', 'P value', 'Mean absolute SHAP value', 'Label', 'Empty'],
            [7, 2.3e-8, 0.1234567890123456, '=literal text', None],
            [0, 0.00017, -0.0012345678901234, 'Synthetic aggregate label', None],
        ]}]
        records = output/record_dir
        records.mkdir(parents=True)
        (records/input_name).write_text(json.dumps(payload), encoding='utf-8')
        return payload


if __name__ == '__main__':
    unittest.main()
