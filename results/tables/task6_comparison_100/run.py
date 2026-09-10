"""Execute the four existing notebooks on the common 100 donor splits."""
from datetime import datetime,timezone
import hashlib
import json
import os
import sys
from pathlib import Path
import time

run=Path(__file__).resolve().parent
root=run.parents[2]
sys.path.insert(0,str(root))
scope=json.loads((run/'run_scope.json').read_text())
for name,digest in scope['source_notebook_sha256'].items():
    assert hashlib.sha256((root/'notebooks'/name).read_bytes()).hexdigest()==digest,name
os.chdir(root)
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['TASK6_SPLIT_LIMIT']='0'
os.environ['TASK6_RUN_TRAINING']='1'
os.environ['TASK6_HARD_TABLES']=str(run)
os.environ['MPLCONFIGDIR']=str(run/'mpl_cache')
import nbformat
from nbclient import NotebookClient
import torch
assert torch.cuda.is_available(),'GPU-Benchmark darf nicht auf CPU zurückfallen.'
started=time.monotonic()
status={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'pid':os.getpid(),'device':torch.cuda.get_device_name(),'completed_models':[],
        'current_model':None,'requested_splits_per_model':100}
prefixes={'quadratic_top1':'task6_quadratic_split_','prototype_soft':'task6_learnable_pooling_relu_split_',
          'quadratic_soft':'task6_quadratic_learnable_split_'}
def save_status():
    status['elapsed_seconds']=time.monotonic()-started
    status['checkpoint_counts']={k:len(list(run.glob(p+'*.pt'))) for k,p in prefixes.items()}
    temp=run/'status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(run/'status.json')
class LoggingClient(NotebookClient):
    def process_message(self,msg,cell,cell_index):
        if msg['msg_type']=='stream':
            text=msg['content']['text'];print(text,end='',flush=True)
            if 'Split ' in text or 'Baseline-Split ' in text:save_status()
        return super().process_message(msg,cell,cell_index)

save_status()
try:
    for key,name in scope['notebooks'].items():
        status['current_model']=key;save_status()
        print(f'BEGIN MODEL {key}',flush=True)
        note=nbformat.read(root/'notebooks'/name,as_version=4)
        note.cells[1].source += '\n\n# Gemeinsamer, explizit angeforderter 100-Split-Vergleich.\n'
        note.cells[1].source += 'BONUS_SPLIT_IDS = list(range(100))\nSPLIT_IDS = BONUS_SPLIT_IDS\n'
        note.cells[1].source += 'TABLES = PROJECT_ROOT / "results/tables/task6_comparison_100"\n'
        note.cells[1].source += 'SPLITS_PATH = TABLES / "task4_donor_splits.csv"\n'
        note.cells[0].source += '\n\n**Dieser ausgeführte Lauf umfasst dieselben Splits 0–99 wie die drei Vergleichsmodelle.** '
        note.cells[0].source += 'Half-Max ist eine Selektion, kein Clustering. Modell und Trainingskonfiguration sind unverändert.\n'
        nbformat.validate(note)
        try:
            LoggingClient(note,timeout=None,kernel_name='ssbi-group-project',
                resources={'metadata':{'path':str(root)}}).execute()
        finally:
            nbformat.write(note,run/f'{key}_100_executed.ipynb')
        if key!='cellcnn':assert len(list(run.glob(prefixes[key]+'*.pt')))==100
        status['completed_models'].append(key);save_status()
        print(f'END MODEL {key}',flush=True)
    from src.task6_comparison import compare_all
    print(compare_all(run,run/'comparison').to_string(index=False),flush=True)
    status['status']='completed';status['current_model']=None
except BaseException as exc:
    status['status']='failed';status['error']=repr(exc);raise
finally:
    save_status();print('FINAL STATUS '+json.dumps(status),flush=True)
