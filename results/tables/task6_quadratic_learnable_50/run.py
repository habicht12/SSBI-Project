from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

run_dir = Path(__file__).resolve().parent
root = run_dir.parents[2]
scope = json.loads((run_dir / 'run_scope.json').read_text())
source = root / scope['source_notebook']
assert hashlib.sha256(source.read_bytes()).hexdigest() == scope['source_notebook_sha256']
os.chdir(root)
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TASK6_SPLIT_LIMIT'] = '0'
os.environ['TASK6_RUN_TRAINING'] = '1'

import nbformat
from nbclient import NotebookClient
import torch

assert torch.cuda.is_available(), 'GPU-Lauf darf nicht auf CPU zurückfallen.'
notebook = nbformat.read(source, as_version=4)
notebook.cells[1].source += '\n\n# Angeforderter 50-Split-Lauf; Modell und Training unverändert.\n'
notebook.cells[1].source += 'BONUS_SPLIT_IDS = list(range(50))\nSPLIT_IDS = BONUS_SPLIT_IDS\n'
notebook.cells[1].source += 'TABLES = PROJECT_ROOT / "results/tables/task6_quadratic_learnable_50"\n'
notebook.cells[1].source += 'SPLITS_PATH = TABLES / "task4_donor_splits.csv"\n'
notebook.cells[1].source += 'print(f"Quadratic Soft-Pooling: {len(BONUS_SPLIT_IDS)} Splits; Ausgaben: {TABLES}")\n'
notebook.cells[0].source += '\n\n**Dieser ausgeführte Lauf umfasst 50 Splits (0–49).** '
notebook.cells[0].source += 'Die kompatiblen Soft-Pooling-Checkpoints 0–9 werden wiederverwendet, 10–49 neu trainiert. '
notebook.cells[0].source += 'Die Hard-Pooling-Referenz umfasst weiterhin ausschließlich die gemeinsamen zehn Splits 0–9.\n'
nbformat.validate(notebook)
started = time.monotonic()
status = {'status': 'running', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
          'requested_splits': 50, 'reused_splits': list(range(10)), 'device': torch.cuda.get_device_name()}

def write_status():
    status['checkpoint_count'] = len(list(run_dir.glob('task6_quadratic_learnable_split_*.pt')))
    status['elapsed_seconds'] = time.monotonic() - started
    temporary = run_dir / 'status.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(run_dir / 'status.json')

class LoggingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            text = msg['content']['text']
            print(text, end='', flush=True)
            if 'Quadratischer Split ' in text:
                write_status()
        return super().process_message(msg, cell, cell_index)

write_status()
client = LoggingClient(notebook, timeout=None, kernel_name='ssbi-group-project',
                       resources={'metadata': {'path': str(root)}})
try:
    client.execute()
    assert len(list(run_dir.glob('task6_quadratic_learnable_split_*.pt'))) == 50
    status['status'] = 'completed'
except BaseException as error:
    status['status'] = 'failed'
    status['error'] = repr(error)
    raise
finally:
    nbformat.write(notebook, run_dir / 'task6_quadratic_learnable_50_executed.ipynb')
    write_status()
    print(f"Status: {status['status']}; Checkpoints: {status['checkpoint_count']}/50; Laufzeit: {status['elapsed_seconds']:.1f} s", flush=True)
