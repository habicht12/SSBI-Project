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
assert hashlib.sha256(source.read_bytes()).hexdigest() == scope['source_notebook_sha256'], 'Notebook seit Vorbereitung verändert.'
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
# Nur Laufumfang und lokale Artefaktpfade überschreiben; Trainingscode bleibt unverändert.
notebook.cells[1].source += '\n\n# Explizit angeforderter separater 50-Split-Lauf; siehe run_scope.json.\n'
notebook.cells[1].source += 'BONUS_SPLIT_IDS = list(range(50))\nSPLIT_IDS = BONUS_SPLIT_IDS\n'
notebook.cells[1].source += 'TABLES = PROJECT_ROOT / "results/tables/task6_learnable_pooling_relu_50"\n'
notebook.cells[1].source += 'SPLITS_PATH = TABLES / "task4_donor_splits.csv"\n'
notebook.cells[1].source += 'print(f"ReLU-Radius und Learnable Pooling; Geometrieregularisierung: {GEOMETRY_COEFFICIENT}; {len(BONUS_SPLIT_IDS)} Splits; Ergebnisse: {TABLES}")\n'
notebook.cells[0].source += '\n\n**Separat angeforderter 50-Split-Lauf:** vorhandene Spender-Splits 0–49 mit Radius/ReLU, Learnable Pooling und unveränderter Geometriestrafe `1e-3`. Laufüberschreibungen und Quellstand stehen in `run_scope.json`. Keine alten Modell-Checkpoints werden übernommen.\n'
paragraphs = notebook.cells[12].source.split('\n\n')
paragraphs[1] = 'Dieser Lauf verwendet die vorhandenen Splits 0–49 und dazu gefilterte, bereits berechnete Baseline-Auswertungen. Alle neuen Artefakte liegen unter `results/tables/task6_learnable_pooling_relu_50`. Der Trainingscode stammt unverändert aus dem geprüften Quellnotebook mit Radius/ReLU und Learnable Pooling; lediglich Splitliste und Artefaktpfade werden überschrieben. Jeder Split wird neu trainiert, soweit noch kein Checkpoint mit exakt passender Konfiguration in diesem Laufordner vorliegt.'
notebook.cells[12].source = '\n\n'.join(paragraphs)
notebook.cells[16].source = notebook.cells[16].source.replace('Die wenigen Splits begründen weder Signifikanz noch allgemeine Überlegenheit.', 'Auch 50 überlappende Splits ersetzen keine zusätzlichen unabhängigen Spender und begründen für sich weder Signifikanz noch allgemeine Überlegenheit.')
notebook.cells[17].source = notebook.cells[17].source.replace('Er bewertet nicht die hier ergänzte Regularisierung; dafür wurde kein neuer Benchmark gestartet.', 'Er bewertet nicht die hier ergänzte Regularisierung; der separat angeforderte neue Lauf umfasst 50 Splits und ist davon getrennt.')
nbformat.validate(notebook)
snapshot = run_dir / 'task6_learnable_pooling_relu_50.ipynb'
if not snapshot.exists():
    nbformat.write(notebook, snapshot)
started = time.monotonic()
status = {
    'status': 'running', 'pid': os.getpid(),
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'requested_splits': 50, 'geometry_coefficient': 1e-3,
    'device': torch.cuda.get_device_name(),
}
def write_status():
    status['checkpoint_count'] = len(list(run_dir.glob('task6_learnable_pooling_relu_split_*.pt')))
    status['elapsed_seconds'] = time.monotonic() - started
    temporary = run_dir / 'status.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(run_dir / 'status.json')

class LoggingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            text = msg['content']['text']
            print(text, end='', flush=True)
            if 'Learnable-Pooling-Split ' in text:
                write_status()
        return super().process_message(msg, cell, cell_index)

write_status()
client = LoggingClient(notebook, timeout=None, kernel_name='ssbi-group-project',
                       resources={'metadata': {'path': str(root)}})
try:
    client.execute()
    assert len(list(run_dir.glob('task6_learnable_pooling_relu_split_*.pt'))) == 50
    status['status'] = 'completed'
except BaseException as error:
    status['status'] = 'failed'
    status['error'] = repr(error)
    raise
finally:
    nbformat.write(notebook, run_dir / 'task6_learnable_pooling_relu_50_executed.ipynb')
    write_status()
    print(f"Status: {status['status']}; Checkpoints: {status['checkpoint_count']}/50; Laufzeit: {status['elapsed_seconds']:.1f} s", flush=True)
