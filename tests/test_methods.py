"""Check renamed methods against exact archived implementations without inference."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MethodParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.pep = load('test_pep', ROOT / 'methods/pep/solution.py')
        cls.nrc = load('test_nrc', ROOT / 'methods/nrc/solution.py')
        for name in ['PEP', 'NRC']:
            path = Path(cls.temporary.name) / name
            with zipfile.ZipFile(ROOT / 'frozen' / f'{name}.zip') as archive:
                archive.extractall(path)
            setattr(cls, 'original_' + name.lower(), load('original_' + name, path / 'solution.py'))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_archives_and_shared_classifier(self):
        expected_hashes = {
            'PEP': '6bb0b761e3572a94e31992e546dad25cbc7a0cb91e4f1326eddf80a8618d2b25',
            'NRC': '4b482b26cf8ec0bd51f5cb2bcb1ad6c964479091e0f828f27414114aef8312d2',
        }
        for method, expected in expected_hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / 'frozen' / f'{method}.zip').read_bytes()).hexdigest(),
                             expected)
            self.assertEqual((ROOT / 'methods' / method.lower() / 'classifier.json').read_bytes(),
                             (Path(self.temporary.name) / method / 'classifier.json').read_bytes())

    def test_pep_scores_and_contract(self):
        vector = np.random.default_rng(2026).normal(size=4096)
        for model in self.pep.MODELS:
            for effort in ['default', 'low', 'medium']:
                for text in ['', 'Let x + 2 = 5.', 'Some prose without numbers.']:
                    self.assertEqual(self.pep.score_from_vector(model, effort, text, vector),
                                     self.original_pep.score_from_vector(model, effort, text, vector))
        with patch.object(self.pep, 'encode_vector', return_value=vector) as encode:
            self.pep.hidden_margin.cache_clear()
            for model in self.pep.MODELS:
                out = self.pep.are_robust(model, 'default', ['A', 'B', 'A'])
                self.assertEqual(out[0], out[2])
                self.assertTrue(all(type(p) is bool for p in out))
            self.assertEqual(encode.call_count, 2)
        self.assertEqual(self.pep.are_robust(self.pep.MODELS[0], 'default', []), [])
        with self.assertRaises(ValueError):
            self.pep.are_robust('unsupported', 'default', [])

    def test_masks_and_regional_arithmetic(self):
        rng = np.random.default_rng(7)
        for text in ['integer x = -12.5; y \\geq 3', r'\frac{a}{b} = 42',
                     'No numeric literals here.', '123', '']:
            offsets = [[0, 0]] + [[i + 8, i + 9] for i in range(len(text))] + [[0, 0]]
            row = dict(problem=text, original_span=[8, 8 + len(text)],
                       mask=list(range(1, len(text) + 1)), offsets=offsets)
            states = rng.normal(size=(len(offsets), 4096)).astype(np.float32)
            old = self.original_nrc.E.role_masks(row)
            new = self.nrc.E.role_masks(row)
            for key in ['body', 'numeric', 'relation', 'identifier']:
                self.assertEqual(new[key], old[key])
            # An empty body is rejected before these features enter inference.
            if row['mask']:
                self.assertEqual(self.nrc.role_features(row, states),
                                 self.original_nrc.role_features(row, states))

    def test_numerical_and_alignment(self):
        for text in ['', '-2.5 + 10', '1/3 and x_2', '9' * 400]:
            self.assertEqual(self.nrc.B.numerical(text), self.original_nrc.B.numerical(text))
        for vectors in [([0, 0], [0, 0], [1, 1]), ([1, 2], [3, 4], [-2, 5])]:
            self.assertEqual(self.nrc.B.anchored_alignment(*vectors),
                             self.original_nrc.B.anchored_alignment(*vectors))
        mask = self.nrc.B.problem_mask('prefix x + 2 suffix', 'x + 2',
                                       [1, 2, 3], [[7, 8], [8, 9], [9, 12]], set())
        self.assertEqual(mask, self.original_nrc.B.problem_mask(
            'prefix x + 2 suffix', 'x + 2', [1, 2, 3], [[7, 8], [8, 9], [9, 12]], set()))

    def test_nrc_head_and_fallbacks(self):
        rng = np.random.default_rng(101)
        for margin in [-1000, -1, -1e-20, 0, 1, 1000]:
            features = rng.normal(size=len(self.nrc.PIPELINE['keys'])).tolist()
            self.assertEqual(self.nrc.score_probability(margin, features),
                             self.original_nrc.score_probability(margin, features))
        features = dict(zip(self.nrc.PIPELINE['keys'], rng.normal(size=8)))
        observation = dict(hidden=0.25, features=features, valid=True)
        for effort in ['default', 'low', 'medium']:
            for valid in [True, False]:
                for nonfinite in [False, True]:
                    row = dict(observation, valid=valid, features=dict(features))
                    if nonfinite:
                        row['features'][self.nrc.PIPELINE['extra_keys'][0]] = math.nan
                    with patch.object(self.nrc, 'measure', return_value=row), \
                         patch.object(self.original_nrc, 'measure', return_value=row):
                        for model in self.nrc.MODELS:
                            self.assertEqual(self.nrc.probability(model, effort, 'x = 12'),
                                             self.original_nrc.probability(model, effort, 'x = 12'))
                            self.assertEqual(self.nrc.are_robust(model, effort, ['x = 12'] * 2),
                                             self.original_nrc.are_robust(model, effort, ['x = 12'] * 2))
        self.assertEqual(self.nrc.are_robust(self.nrc.MODELS[0], 'default', []), [])

    def test_packaged_modules_are_standalone(self):
        packager = load('test_packager', ROOT / 'scripts/package.py')
        for method in ['pep', 'nrc']:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                archive = directory / 'submission.zip'
                digest = packager.package(method, archive)
                self.assertEqual(digest, packager.package(method, archive))
                with zipfile.ZipFile(archive) as bundle:
                    self.assertIn('solution.py', bundle.namelist())
                    self.assertIn('small.txt', bundle.namelist())
                    bundle.extractall(directory / 'extracted')
                module = load('packaged_' + method, directory / 'extracted/solution.py')
                self.assertEqual(module.are_robust(module.MODELS[0], 'default', []), [])


if __name__ == '__main__':
    unittest.main()
