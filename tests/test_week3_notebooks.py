"""Execute every Week 3 notebook in an isolated copy using synthetic test data."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import pandas as pd

from test_classification import synthetic_catalog


class NotebookSmokeTests(unittest.TestCase):
    def test_all_week3_notebooks(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='gaia-week3-test-') as directory:
            root = Path(directory)
            shutil.copytree(repository / 'src', root / 'src')
            (root / 'notebooks').mkdir()
            cache = root / 'data/processed'
            cache.mkdir(parents=True)
            synthetic_catalog().to_csv(cache / 'gaia_clean_day1.csv', index=False)
            paths = sorted((repository / 'notebooks').glob('week3_day_*.ipynb'))
            self.assertEqual(len(paths), 7)
            paths.insert(0, repository / 'notebooks/week3_prerequisites.ipynb')
            for index, path in enumerate(paths):
                with self.subTest(notebook=path.name):
                    notebook = nbformat.read(path, as_version=4)
                    nbformat.validate(notebook)
                    # Use this project's interpreter, regardless of the user's default kernel.
                    manager = KernelManager(kernel_name='python3')
                    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
                    cwd = root if index % 2 == 0 else root / 'notebooks'
                    client = NotebookClient(notebook, km=manager, timeout=120,
                                            resources={'metadata': {'path': str(cwd)}})
                    try:
                        # Extra assertions run only in the test copy, not in learning cells.
                        if path.name.startswith('week3_day_01'):
                            notebook.cells.append(nbformat.v4.new_code_cell('''
assert prepared.source_id.is_unique
assert set(prepared.split) == {'train', 'validation', 'test'}
assert (prepared.is_bright == (prepared.abs_g_mag < 4).astype(int)).all()
assert all(set(part.is_bright) == {0, 1} for part in (train, validation, test))
assert not (set(train.source_id) & set(test.source_id))
assert not (set(validation.source_id) & set(test.source_id))
assert not (set(train.source_id) & set(validation.source_id))
'''))
                        if path.name.startswith('week3_day_06'):
                            notebook.cells.append(nbformat.v4.new_code_cell('''
expanded = best_model.named_steps['polynomialfeatures'].transform(X_train)
np.testing.assert_allclose(best_model.named_steps['standardscaler'].mean_, expanded.mean(axis=0))
assert len(comparison) == 9
assert choice.validation_log_loss == comparison.validation_log_loss.min()
'''))
                        client.execute()
                        self.assertTrue(any(output.output_type == 'display_data'
                                            for cell in notebook.cells if cell.cell_type == 'code'
                                            for output in cell.outputs))
                        print(f'Executed {path.name} (isolated data copy)', flush=True)
                    finally:
                        if client.kc is not None:
                            client.kc.stop_channels()
                        if manager.has_kernel:
                            manager.shutdown_kernel(now=True)
            output = cache / 'week3'
            predictions = pd.read_csv(output / 'test_predictions.csv', dtype={'source_id': str})
            manifest = pd.read_csv(output / 'split_manifest.csv', dtype={'source_id': str})
            self.assertEqual(set(predictions.source_id), set(manifest.loc[manifest.split == 'test', 'source_id']))
            self.assertTrue(manifest.source_id.is_unique)
            self.assertEqual(len(list(output.glob('*.csv'))), 6)


if __name__ == '__main__':
    unittest.main()
