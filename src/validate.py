"""Validate user-supplied training data without publishing study partitions."""
from pathlib import Path
from common import manifest,sha256

def validate(config,full_files=False):
    root=Path(config['data']['data_root']).resolve()
    partitions={key:manifest(config,key) for key in ['train','validation']}
    for key,rows in partitions.items():
        if not rows:raise ValueError(f'Empty partition: {key}')
        for row in rows:
            grade=int(row['mes_grade'])
            if grade not in range(4) or int(row['boundary_label'])!=int(grade>=2):raise ValueError('Inconsistent MES/boundary label')
            if not row.get('patient_id') or int(row['eligible_ordinal'])!=1:raise ValueError('Patient ID and exact MES labels are required')
            path=(root/row['relative_path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():raise ValueError(f'Image unavailable: {path}')
            if full_files and (not row.get('sha256') or sha256(path)!=row['sha256']):raise ValueError('Missing or mismatched image hash')
    for field in ['patient_id','image_id','sha256']:
        groups=[{r.get(field) for r in rows if r.get(field)} for rows in partitions.values()]
        if groups[0]&groups[1]:raise ValueError(f'Cross-partition overlap: {field}')
    for spec in config['weights'].values():
        if sha256(spec['path'])!=spec['sha256']:raise ValueError('Pretrained weight integrity check failed')
    return {'status':'passed','images':{k:len(v) for k,v in partitions.items()}}
