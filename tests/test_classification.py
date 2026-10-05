"""Numerical and data-isolation checks; fixtures below are synthetic, not Gaia."""
import unittest
import ast
import nbformat
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import classification as clf


def synthetic_catalog(n=300):
    """Seeded fake measurements for offline verification only."""
    rng = np.random.default_rng(21)
    color = rng.uniform(-0.3, 3.5, n)
    magnitude = 2 + 2 * color + rng.normal(0, 1.8, n)
    parallax = rng.uniform(2, 15, n)
    return pd.DataFrame({'source_id': [str(6000000000000000000 + i) for i in range(n)],
                         'bp_rp': color, 'phot_g_mean_mag': magnitude - 5*np.log10(parallax) + 10,
                         'parallax': parallax, 'parallax_error': parallax/20})


def notebook_math():
    """Test the actual teaching functions without importing a second implementation."""
    path = Path(__file__).resolve().parents[1] / 'notebooks/week3_day_05_logistic_gradient_descent.ipynb'
    notebook = nbformat.read(path, as_version=4)
    namespace = {'np': np}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.FunctionDef):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


math = notebook_math()


class ClassificationTests(unittest.TestCase):
    def test_missing_default_cache_fetches_once_then_reuses_exact_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'data/processed/gaia_clean_day1.csv'

            def fetch_and_cache(paths, **kwargs):
                self.assertEqual(paths, [cache.resolve()])
                self.assertEqual(kwargs, {'top_n': 500, 'verify_ssl': True})
                cache.parent.mkdir(parents=True)
                synthetic_catalog().to_csv(cache, index=False)

            with patch.object(clf, '__file__', str(root / 'src/classification.py')):
                with patch('src.data_load.load_clean_gaia_sample', side_effect=fetch_and_cache) as fetch:
                    first, _ = clf.load_week3_data(top_n=500)
                    second, _ = clf.load_week3_data()
                    fetch.assert_called_once()
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(first.source_id.iloc[0], '6000000000000000000')

    def test_missing_explicit_path_and_offline_mode_never_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(clf, '__file__', str(root / 'src/classification.py')):
                with patch('src.data_load.load_clean_gaia_sample') as fetch:
                    with self.assertRaises(FileNotFoundError):
                        clf.load_week3_data(root / 'typo.csv')
                    with self.assertRaises(FileNotFoundError):
                        clf.load_week3_data(use_query_if_missing=False)
                    fetch.assert_not_called()

    def test_explicit_path_expands_home(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stars.csv'
            synthetic_catalog().to_csv(path, index=False)
            with patch('pathlib.Path.expanduser', return_value=path) as expand:
                frame, _ = clf.load_week3_data('~/stars.csv')
                expand.assert_called_once()
            self.assertEqual(len(frame), 300)

    def test_measurement_cleaning_and_no_mutation(self):
        raw = synthetic_catalog(30)
        raw.loc[0, ['parallax', 'phot_g_mean_mag']] = [10, 9]  # M_G exactly 4
        raw.loc[1, 'bp_rp'] = np.inf
        raw.loc[2, 'parallax'] = -1
        raw.loc[3, 'parallax_error'] = 0
        raw.loc[4, 'parallax_error'] = raw.loc[4, 'parallax']  # low SNR
        raw = pd.concat([raw, raw.iloc[[5]]], ignore_index=True)
        original = raw.copy(deep=True)
        df, audit = clf.clean_week3_sample(raw)
        pd.testing.assert_frame_equal(raw, original)
        self.assertEqual(len(df), 26)
        self.assertTrue(df.source_id.is_unique)
        self.assertTrue((audit.rows.diff().dropna() <= 0).all())

    def test_stable_extremes_and_validation(self):
        np.testing.assert_allclose(math['sigmoid']([-1000, 0, 1000]), [0, .5, 1])
        cost = math['compute_cost'](np.array([[1000.], [-1000.]]), np.array([0, 1]), np.array([1.]), 0)
        self.assertEqual(cost, 1000)

    def test_regularized_gradients_match_finite_differences(self):
        rng = np.random.default_rng(42)
        X, y, w, b = rng.normal(size=(12, 3)), rng.integers(0, 2, 12), rng.normal(size=3), .2
        dw, db = math['compute_gradient'](X, y, w, b, lambda_=2.3)
        eps = 1e-6
        for i in range(len(w)):
            delta = np.eye(len(w))[i]*eps
            numerical = (math['compute_cost'](X, y, w+delta, b, lambda_=2.3) - math['compute_cost'](X, y, w-delta, b, lambda_=2.3))/(2*eps)
            self.assertAlmostEqual(dw[i], numerical, places=7)
        numerical_b = (math['compute_cost'](X, y, w, b+eps, lambda_=2.3) - math['compute_cost'](X, y, w, b-eps, lambda_=2.3))/(2*eps)
        self.assertAlmostEqual(db, numerical_b, places=7)

    def test_scratch_matches_sklearn_regularization(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 3))
        y = rng.binomial(1, math['sigmoid'](X @ np.array([1, -.4, .8]) + .2))
        w, b, history = math['gradient_descent'](X, y, iterations=4000, lambda_=2)
        model = LogisticRegression(C=.5, tol=1e-10, max_iter=5000).fit(X, y)
        np.testing.assert_allclose(math['sigmoid'](X@w+b), model.predict_proba(X)[:, 1], atol=1e-6)
        self.assertTrue((np.diff(history) <= 1e-12).all())



if __name__ == '__main__':
    unittest.main()
