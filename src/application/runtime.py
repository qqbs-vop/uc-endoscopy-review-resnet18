"""Shared frozen-image inference, Grad-CAM and grouped permutation SHAP."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF

CODE=Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:sys.path.insert(0,str(CODE))
from models import UCModel
from data import preprocess, MEAN, STD


class Cancelled(RuntimeError):pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


class Predictor:
    def __init__(self,manifest,model_id,device='cpu'):
        self.manifest_path=Path(manifest).resolve()
        self.entry=json.loads(self.manifest_path.read_text(encoding='utf-8'))['models'][model_id]
        self.identifier=model_id;self.device=torch.device(device)
        self.weight=self.manifest_path.parent/self.entry['file']
        if digest(self.weight)!=self.entry['sha256']:raise ValueError('Model weight integrity check failed.')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=self.device.type=='cuda'
        self.precision_profile='frozen_gpu_tf32_batch32' if self.device.type=='cuda' else 'cpu_fp32'
        self.model=UCModel(self.entry['backbone'],self.entry['variant'],self.entry['seed']).to(self.device)
        payload=torch.load(self.weight,map_location='cpu',weights_only=True)
        self.model.load_state_dict(payload,strict=True);self.model.eval()
        self.temperature=float(self.entry.get('temperature',1.))
        if not np.isfinite(self.temperature) or self.temperature<=0:raise ValueError('Invalid probability calibration parameter.')

    def probabilities(self,x):
        logits=self.logits(x)
        return logits.softmax(1)

    def logits(self,x):
        count=len(x)
        if self.device.type=='cuda' and count<32:
            x=torch.cat([x,x[:1].expand(32-count,-1,-1,-1)])
        logits,_=self.model(x)
        return logits[:count]

    def prepare(self,path):
        with Image.open(path) as im:
            original=im.convert('RGB');original.load()
        return original,preprocess(original).unsqueeze(0).to(self.device)

    @staticmethod
    def geometry(original):
        w,h=original.size;scale=224/max(w,h)
        nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
        return {'original_width':w,'original_height':h,'input_width':224,'input_height':224,
                'left':(224-nw)//2,'top':(224-nh)//2,'width':nw,'height':nh,
                'resize':'PIL bilinear; long side 224; ImageNet mean-color letterbox'}

    def infer(self,path):
        start=time.perf_counter();original,x=self.prepare(path)
        with torch.no_grad():
            logits=self.logits(x);p=logits.softmax(1)[0].cpu().numpy()
            calibrated=(logits/self.temperature).softmax(1)[0].cpu().numpy()
        return {'status':'complete','image_sha256':digest(path),'model_id':self.identifier,
                'weight_sha256':self.entry['sha256'],'source_checkpoint_sha256':self.entry['source_checkpoint_sha256'],
                'raw_probabilities':p.tolist(),'calibrated_probabilities':calibrated.tolist(),
                'temperature':self.temperature,'predicted_mes':int(p.argmax()),'q_raw':float(p[2:].sum()),
                'q_calibrated':float(calibrated[2:].sum()),'geometry':self.geometry(original),
                'preprocessing_version':'uc_input_letterbox224_v1','device':str(self.device),
                'precision_profile':self.precision_profile,
                'seconds':time.perf_counter()-start}

    def cam(self,path,target='predicted'):
        if target not in ('predicted','boundary'):raise ValueError('Unknown CAM target.')
        original,x=self.prepare(path);holder={}
        layer=self.model.encoder.layer4[-1] if self.entry['backbone']=='resnet18' else self.model.encoder.features.norm5
        def capture(module,args,output):holder['activation']=output
        handle=layer.register_forward_hook(capture)
        try:
            logits=self.logits(x)
            cls=int(logits.argmax(1)) if target=='predicted' else None
            objective=logits[0,cls] if cls is not None else torch.logsumexp(logits[0,2:],0)-torch.logsumexp(logits[0],0)
            activation=holder['activation']
            gradient=torch.autograd.grad(objective,activation)[0]
            raw=torch.relu((gradient.mean((2,3),keepdim=True)*activation).sum(1,keepdim=True))
            heat=torch.nn.functional.interpolate(raw,(224,224),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
            heat=(heat-heat.min())/(heat.max()-heat.min()+1e-12)
        finally:handle.remove()
        return heat,{'target':f'MES{cls} logit' if cls is not None else 'log(p2+p3)',
                     'method':'Grad-CAM','zero_activation':bool(np.max(heat)==0),
                     'geometry':self.geometry(original)}

    def baseline(self,x,geometry):
        rgb=(x[0].detach().cpu()*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]).clamp(0,1)
        blurred=TF.to_tensor(TF.to_pil_image(rgb).filter(ImageFilter.GaussianBlur(8)))
        base=TF.normalize(blurred,MEAN,STD).unsqueeze(0).to(self.device)
        inside=torch.zeros((224,224),dtype=torch.bool,device=self.device)
        g=geometry;inside[g['top']:g['top']+g['height'],g['left']:g['left']+g['width']]=True
        return torch.where(inside[None,None],base,x)

    def shap(self,path,permutations=64,seed=20261111,cancel=None):
        import shap
        if not 1<=permutations<=128:raise ValueError('SHAP permutations must be between 1 and 128.')
        original,x=self.prepare(path);geo=self.geometry(original);base=self.baseline(x,geo)
        labels=torch.tensor(np.repeat(np.repeat(np.arange(64).reshape(8,8),28,0),28,1),device=self.device)
        start=time.perf_counter();evaluated=0
        def predict_flags(flags):
            nonlocal evaluated
            outputs=[]
            for offset in range(0,len(flags),32):
                if cancel is not None and cancel.is_set():raise Cancelled('SHAP computation cancelled.')
                mask=torch.as_tensor(flags[offset:offset+32],dtype=x.dtype,device=self.device)[:,labels]
                batch=torch.where(mask[:,None].bool(),x,base)
                actual=len(batch)
                if actual<32:batch=torch.cat([batch,batch[:1].expand(32-actual,-1,-1,-1)])
                with torch.no_grad():
                    p=self.probabilities(batch)[:actual]
                    outputs.append(torch.cat([p,p[:,2:].sum(1,keepdim=True)],1).cpu().numpy())
                evaluated+=len(batch)
            return np.concatenate(outputs)
        explainer=shap.PermutationExplainer(predict_flags,np.zeros((1,64)),seed=seed,
                                          feature_names=[f'R{i//8+1}C{i%8+1}' for i in range(64)],
                                          output_names=['MES0','MES1','MES2','MES3','MES>=2'])
        explanation=explainer(np.ones((1,64)),max_evals=129*permutations,batch_size=32,silent=True)
        values=np.asarray(explanation.values)[0];reference=np.asarray(explanation.base_values)[0]
        with torch.no_grad():p=self.probabilities(x)[0].cpu().numpy()
        direct=np.r_[p,p[2:].sum()]
        expected=predict_flags(np.ones((1,64)))[0]
        if np.max(np.abs(expected-direct))>2e-5:raise ValueError('SHAP batch and single-image outputs do not agree.')
        residual=reference+values.sum(0)-expected
        if np.max(np.abs(residual))>1e-5:raise ValueError(f'SHAP additivity check failed: {residual}')
        target=int(p.argmax());top={};remainder={}
        for name,index in [('predicted_class',target),('MES>=2',4)]:
            ranked=np.argsort(-np.abs(values[:,index]),kind='stable')[:10]
            top[name]=[{'region':int(i),'row':int(i//8+1),'column':int(i%8+1),
                        'shap_value':float(values[i,index]),'bounds_input224':[int(i%8*28),int(i//8*28),28,28]} for i in ranked]
            remainder[name]=float(values[:,index].sum()-values[ranked,index].sum())
        return {'status':'complete','method':'SHAP PermutationExplainer, 64 fixed spatial regions',
                'shap_version':shap.__version__,'outputs':['MES0','MES1','MES2','MES3','MES>=2'],
                'values':values.tolist(),'base_values':reference.tolist(),'model_output':expected.tolist(),
                'additivity_residual':residual.tolist(),'max_abs_residual':float(np.max(np.abs(residual))),
                'batch_single_output_difference':(expected-direct).tolist(),
                'top10':top,'remaining54_sum':remainder,'precision_profile':self.precision_profile,
                'geometry':geo,'permutations':permutations,'seed':seed,
                'baseline':'input224 Gaussian blur radius8, original letterbox restored',
                'mask_model_evaluations':evaluated,'seconds':time.perf_counter()-start}

    @staticmethod
    def overlay(original,heat,geometry):
        g=geometry;crop=heat[g['top']:g['top']+g['height'],g['left']:g['left']+g['width']]
        scalar=Image.fromarray(np.uint8(np.clip(crop,0,1)*255)).resize(original.size,Image.Resampling.BILINEAR)
        v=np.array(scalar,dtype=float)/255
        color=np.stack([v,np.sqrt(v)*.7,1-v],-1)*255
        alpha=(v*.5)[...,None]
        return Image.fromarray(np.uint8(np.asarray(original)*(1-alpha)+color*alpha))

    def export(self,path,destination,explain=False,permutations=64,cancel=None,progress=None):
        out=Path(destination)
        if out.exists():raise FileExistsError(out)
        try:
            return self._export(path,out,explain,permutations,cancel,progress)
        except Exception as exc:
            if out.exists():
                file=out/'result.json'
                result=json.loads(file.read_text(encoding='utf-8')) if file.exists() else {}
                result.update({'status':'cancelled' if isinstance(exc,Cancelled) else 'failed',
                               'error':str(exc),'requested_shap':bool(explain)})
                save_json(file,result)
            raise

    def _export(self,path,out,explain,permutations,cancel,progress):
        out.mkdir(parents=True,exist_ok=False)
        result=self.infer(path);original,_=self.prepare(path)
        result.update({'status':'running','requested_shap':bool(explain)})
        save_json(out/'result.json',result)
        if progress:progress('classification',dict(result))
        original.save(out/'original.png')
        for target in ['predicted','boundary']:
            if cancel is not None and cancel.is_set():raise Cancelled('Cancelled.')
            heat,meta=self.cam(path,target)
            self.overlay(original,heat,meta['geometry']).save(out/f'cam_{target}.png')
            np.save(out/f'cam_{target}.npy',heat,allow_pickle=False)
            result[f'cam_{target}']=meta
        save_json(out/'result.json',result)
        if progress:progress('cam',dict(result))
        if explain:
            explanation=self.shap(path,permutations,cancel=cancel)
            save_json(out/'shap.json',explanation)
            values=np.array(explanation['values'])[:,result['predicted_mes']].reshape(8,8)
            heat=np.repeat(np.repeat(values,28,0),28,1)
            maximum=np.max(np.abs(heat))
            # Diverging blue-negative / red-positive map, with opacity proportional to magnitude.
            alpha=np.abs(heat)/(maximum+1e-12)*.6
            color=np.zeros((224,224,3));color[heat>=0]=[220,40,40];color[heat<0]=[35,85,210]
            g=result['geometry'];sl=np.s_[g['top']:g['top']+g['height'],g['left']:g['left']+g['width']]
            rgb=Image.fromarray(color[sl].astype('uint8')).resize(original.size,Image.Resampling.NEAREST)
            a=Image.fromarray(np.uint8(alpha[sl]*255)).resize(original.size,Image.Resampling.NEAREST)
            original_rgba=original.convert('RGBA');rgb=rgb.convert('RGBA');rgb.putalpha(a)
            Image.alpha_composite(original_rgba,rgb).convert('RGB').save(out/'shap_regions.png')
            result['shap']={'file':'shap.json','seconds':explanation['seconds'],'max_abs_residual':explanation['max_abs_residual']}
        result['status']='complete'
        save_json(out/'result.json',result)
        return result
