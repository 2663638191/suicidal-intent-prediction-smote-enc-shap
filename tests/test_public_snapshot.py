"""Check the source distribution and input definitions without participant data."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def visible_files():
    command = ['git', '-c', 'safe.directory=' + ROOT.as_posix(),
               'ls-files', '--cached', '--others', '--exclude-standard', '-z']
    data = subprocess.check_output(command, cwd=ROOT).decode('utf-8')
    return {name for name in data.split('\0') if name and (ROOT/name).is_file()}


class PublicSnapshotTests(unittest.TestCase):
    def test_file_manifest(self):
        manifest = json.loads((ROOT/'docs/repository_manifest.json').read_text(encoding='utf-8'))
        actual = visible_files() - {'docs/repository_manifest.json'}
        self.assertEqual(actual, {item['path'] for item in manifest['files']})
        for item in manifest['files']:
            with self.subTest(path=item['path']):
                path = (ROOT/item['path']).resolve()
                self.assertTrue(path.is_relative_to(ROOT))
                content = path.read_bytes()
                self.assertEqual(len(content), item['bytes'])
                self.assertEqual(hashlib.sha256(content).hexdigest(), item['sha256'])

    def test_field_only_input_structure(self):
        folder = ROOT/'data/schema'
        with (folder/'input_headers_zh.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        schema = json.loads((folder/'input_schema.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 30)
        self.assertEqual(len(set(rows[0])), 30)
        self.assertEqual(rows[0], schema['source_headers'])
        self.assertTrue(schema['header_only'])
        self.assertEqual(schema['participant_response_rows'], 0)
        self.assertIn(schema['outcome_source'], rows[0])

    def test_predictor_dictionary_and_field_roles(self):
        folder = ROOT/'data/schema'
        with (folder/'variable_dictionary_en.csv').open(encoding='utf-8-sig', newline='') as stream:
            variables = list(csv.DictReader(stream))
        with (folder/'source_field_mapping.csv').open(encoding='utf-8-sig', newline='') as stream:
            fields = list(csv.DictReader(stream))
        self.assertEqual(len(variables), 25)
        self.assertEqual(len(fields), 30)
        self.assertEqual(len({item['code_key_en'] for item in variables}), len(variables))
        self.assertEqual(sum(item['type']=='continuous' for item in variables), 21)
        self.assertEqual(sum(item['type']=='nominal categorical' for item in variables), 4)
        predictors = {item['source_header_zh'] for item in fields if 'predictor' in item['role']}
        self.assertEqual(predictors, {item['source_header_zh'] for item in variables})
        for item in variables:
            self.assertTrue(item['label_en'])
            self.assertRegex(item['code_key_en'], r'^[a-z][a-z0-9_]*$')

    def test_localisation_matches_variable_dictionary(self):
        spec = importlib.util.spec_from_file_location('public_localisation', ROOT/'code/en/localisation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with (ROOT/'data/schema/variable_dictionary_en.csv').open(encoding='utf-8-sig', newline='') as stream:
            variables = list(csv.DictReader(stream))
        expected = {item['analysis_name_zh']:item['label_en'] for item in variables}
        self.assertEqual(module.FEATURES, expected)
        for source, label in expected.items():
            self.assertEqual(module.translate(source), label)

    def test_python_sources_compile_without_importing_scientific_packages(self):
        paths = sorted((ROOT/'code').rglob('*.py'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')

    def test_publication_boundary(self):
        forbidden_suffixes = {'.joblib','.pkl','.pickle','.npz','.npy','.doc','.docx',
                              '.xlsx','.xls','.pdf','.png','.jpg','.jpeg','.svg','.zip','.tar','.pyc'}
        forbidden_parts = {'results','private','private_records_do_not_publish',
                           '私密记录_禁止公开','依赖环境','运行缓存','__pycache__'}
        for name in visible_files():
            path = Path(name)
            with self.subTest(path=name):
                self.assertNotIn(path.suffix.lower(), forbidden_suffixes)
                self.assertFalse(set(path.parts) & forbidden_parts)
                if path.suffix.lower()=='.csv':
                    self.assertEqual(path.parts[:2], ('data','schema'))
        self.assertFalse((ROOT/'code/en/resources/frozen_baseline.json').exists())
        glossary = json.loads((ROOT/'code/en/resources/frozen_glossary.json').read_text(encoding='utf-8'))
        forbidden_keys = {'observed_minimum','observed_maximum','observed_range_note','nonmissing_n'}
        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden_keys)
                for child in value.values(): inspect(child)
            elif isinstance(value, list):
                for child in value: inspect(child)
        inspect(glossary)

    def test_relative_documentation_links(self):
        documents = [ROOT/'README.md',ROOT/'README_zh-CN.md',ROOT/'CONTRIBUTING.md']
        documents += list((ROOT/'docs').glob('*.md')) + list((ROOT/'code').rglob('README.md'))
        count = 0
        for path in documents:
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',target) or target.startswith('#'):
                    continue
                target = unquote(target.split('#',1)[0].strip('<>'))
                if not target: continue
                resolved = (path.parent/target).resolve()
                with self.subTest(document=path.relative_to(ROOT).as_posix(),target=target):
                    self.assertTrue(resolved.is_relative_to(ROOT))
                    self.assertTrue(resolved.exists())
                count += 1
        self.assertGreater(count, 0)


if __name__=='__main__':
    unittest.main()
