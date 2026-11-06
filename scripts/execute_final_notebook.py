"""Execute the release showcase using this Python environment's kernel."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1]
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
path=ROOT/'notebooks/final_results.ipynb'
nb=nbformat.read(path,as_version=4)
km=KernelManager(kernel_name='python3');km.kernel_spec.argv[0]=sys.executable
native=Path(sys.prefix)/'Library/bin'
if native.is_dir():os.environ['PATH']=str(native)+os.pathsep+os.environ.get('PATH','')
client=NotebookClient(nb,km=km,timeout=900,resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(nb,path)
    if km.has_kernel:km.shutdown_kernel(now=True)
nbformat.validate(nb)
print('Executed final_results.ipynb successfully with',sys.executable)
