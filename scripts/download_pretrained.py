"""Obtain the original ImageNet initialization; no dataset download."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/'configs/training.json').read_text())['weights']['resnet18']
target=ROOT/spec['path'];target.parent.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
if target.exists():
    if sha(target)!=spec['sha256']:raise ValueError('Existing file differs; no overwrite performed')
else:
    partial=target.with_suffix('.download')
    if partial.exists():raise FileExistsError(partial)
    urllib.request.urlretrieve('https://download.pytorch.org/models/resnet18-f37072fd.pth',partial)
    if sha(partial)!=spec['sha256']:raise ValueError('Downloaded initialization hash mismatch')
    partial.rename(target)
print('Pretrained initialization verified.')
