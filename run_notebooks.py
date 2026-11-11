"""Run both notebooks, preserve outputs, and export report artifacts."""
from pathlib import Path
import base64
import os
import sys

ROOT = Path(__file__).resolve().parent
runtime = Path(os.environ.get('DZ1_RUNTIME_DIR', ROOT / '.runtime'))
runtime.mkdir(parents=True, exist_ok=True)
for name, folder in {
    'JUPYTER_CONFIG_DIR': 'jupyter-config',
    'JUPYTER_DATA_DIR': 'jupyter-data',
    'JUPYTER_RUNTIME_DIR': 'jupyter-runtime',
    'IPYTHONDIR': 'ipython',
    'MPLCONFIGDIR': 'matplotlib',
}.items():
    os.environ[name] = str(runtime / folder)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec


def streams(cell):
    return ''.join(out.get('text', '') for out in cell.get('outputs', []))


def main():
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    for kind in ('attention', 'bughunt'):
        path = ROOT / 'notebooks' / f'dz1-{kind}.ipynb'
        notebook = nbformat.read(path, as_version=4)
        km = KernelManager(kernel_name='python3')
        km._kernel_spec = KernelSpec(
            argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            display_name='Python 3', language='python',
        )
        client = NotebookClient(
            notebook, km=km, timeout=1800,
            resources={'metadata': {'path': str(ROOT)}},
        )

        def completed(cell, cell_index, **kwargs):
            print(f'{kind}: executed cell {cell_index + 1}', flush=True)
            print(streams(cell), end='', flush=True)

        client.on_cell_executed = completed
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
        saved = nbformat.read(path, as_version=4)
        nbformat.validate(saved)
        code_cells = [c for c in saved.cells if c.cell_type == 'code']
        if any(c.execution_count is None for c in code_cells):
            raise RuntimeError(f'{kind}: unexecuted cell')
        check_cell = next(c for c in code_cells if 'failures = run(implementation)' in c.source)
        checks = streams(check_cell)
        if 'пройдено 17 из 17' not in checks or 'Все проверки зелёные.' not in checks:
            raise RuntimeError(f'{kind}: checks failed')
        (results / f'{kind}-checks.txt').write_text(checks, encoding='utf-8')
        if kind == 'bughunt':
            training = next(c for c in code_cells if 'for step in range(600)' in c.source)
            (results / 'training.txt').write_text(streams(training), encoding='utf-8')
            plot = next(o for c in code_cells for o in c.outputs if 'image/png' in o.get('data', {}))
            (results / 'loss.png').write_bytes(base64.b64decode(plot.data['image/png']))
            generation = next(c for c in code_cells if 'max_new_tokens=200' in c.source)
            generated = streams(generation)
            if len(generated.rstrip('\n')) != 201:
                raise RuntimeError('Unexpected generated text length')
            (results / 'generated_text.txt').write_text(generated, encoding='utf-8')
    print('Both notebooks passed all 17 checks; results saved.', flush=True)


if __name__ == '__main__':
    main()
