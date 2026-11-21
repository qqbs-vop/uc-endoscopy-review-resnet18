"""Run the released frozen model and export local results."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from application.runtime import Predictor
from application.desktop import write_reports
import json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True);p.add_argument('--output',required=True)
    p.add_argument('--shap',action='store_true');p.add_argument('--permutations',type=int,default=64)
    args=p.parse_args()
    result=Predictor(ROOT/'models/manifest.json','resnet18_H_seed29').export(args.image,args.output,args.shap,args.permutations)
    file=Path(args.output)/'shap.json'
    explanation=json.loads(file.read_text()) if file.exists() else None
    write_reports(args.output,result,explanation)
    print(json.dumps({'predicted_MES':result['predicted_mes'],'probabilities':result['raw_probabilities'],'output':args.output},indent=2))
if __name__=='__main__':main()
