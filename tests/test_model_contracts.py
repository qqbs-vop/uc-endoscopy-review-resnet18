import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import run_table
from models import UCModel,task_loss
from application.runtime import digest

class Contracts(unittest.TestCase):
    def test_resnet_schedule(self):
        runs=run_table();self.assertEqual(len(runs),9)
        self.assertEqual({b for b,v,s in runs},{'resnet18'})
        self.assertEqual({s for b,v,s in runs},{17,29,43})
    def test_shared_classifier_initialization(self):
        a=UCModel('resnet18','A',29);h=UCModel('resnet18','H',29)
        self.assertTrue(torch.equal(a.classifier.weight,h.classifier.weight))
    def test_huber_training_target(self):
        logits=torch.tensor([[0.1,0.2,0.3,0.4]])
        aux=torch.tensor([0.5]);grade=torch.tensor([2]);boundary=torch.tensor([1])
        expected=torch.nn.functional.cross_entropy(logits,grade)+0.3*torch.nn.functional.huber_loss(aux,grade.float(),delta=1.)
        self.assertTrue(torch.equal(task_loss('H',logits,aux,grade,boundary),expected))
    def test_release_weight_integrity(self):
        import json
        entry=json.loads((ROOT/'models/manifest.json').read_text())['models']['resnet18_H_seed29']
        self.assertEqual(digest(ROOT/'models'/entry['file']),entry['sha256'])
    def test_frozen_tensor_layout(self):
        model=UCModel('resnet18','H',29)
        model.load_state_dict(torch.load(ROOT/'models/resnet18_H_seed29.pt',map_location='cpu',weights_only=True),strict=True)
        model.eval()
        with torch.no_grad():logits,aux=model(torch.zeros(1,3,224,224))
        self.assertEqual(tuple(logits.shape),(1,4));self.assertEqual(tuple(aux.shape),(1,))
        self.assertTrue(torch.isfinite(logits).all())
if __name__=='__main__':unittest.main()
