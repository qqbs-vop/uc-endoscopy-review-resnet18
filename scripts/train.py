"""ResNet-18 training on user-provided data; Linux training entry point."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import run_id,run_table,load_config

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(ROOT/'configs/training.json'))
    p.add_argument('--plan',action='store_true')
    p.add_argument('--run',choices=[run_id(*r) for r in run_table()])
    p.add_argument('--output',default=str(ROOT/'outputs/training'))
    p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    if args.plan:print(json.dumps([run_id(*r) for r in run_table()],indent=2));return
    cfg=load_config(args.config)
    cfg['data']['data_root']=str((ROOT/cfg['data']['data_root']).resolve())
    for item in cfg['data']['manifests'].values():item['path']=str((ROOT/item['path']).resolve())
    for item in cfg['weights'].values():item['path']=str((ROOT/item['path']).resolve())
    from validate import validate
    print(json.dumps(validate(cfg),indent=2))
    if args.validate_only:return
    if not args.run:p.error('--run is required for training')
    if os.environ.get('SLURM_JOB_ID'):
        from runtime_device import install
        install()
    from engine import train
    chosen=next(r for r in run_table() if run_id(*r)==args.run)
    print(json.dumps(train(cfg,args.output,*chosen,device=args.device,resume=args.resume),indent=2))
if __name__=='__main__':main()
