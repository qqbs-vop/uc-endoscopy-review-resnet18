"""English-only single-model desktop with packaged-application verification."""
import argparse
import html
import json
import queue
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from PIL import Image,ImageTk,ImageGrab

if __package__:
    from .runtime import Predictor,Cancelled,save_json
else:
    from runtime import Predictor,Cancelled,save_json

MODEL_ID='resnet18_H_seed29'
TEXT={
 'en':{
  'title':'Ulcerative Colitis Image Analysis','language':'Language',
  'scope':'Input: static UC endoscopy images. Outputs support research and physician review.',
  'open':'Open image','predict':'Predict + CAM','explain':'Predict + CAM + SHAP','permutations':'SHAP permutations','cancel':'Cancel','output':'Choose export folder',
  'original':'Original image','cam':'Predicted-class Grad-CAM','shap_image':'Regional SHAP (red +, blue −)',
  'result':'Prediction results','top10':'Top 10 SHAP regions','rank':'Rank','class_region':'Class region','class_value':'Class SHAP','boundary_region':'Boundary region','boundary_value':'Boundary SHAP',
  'note':'CAM and SHAP show image regions associated with the classification for physician review.',
  'no_image':'No image selected','ready':'Ready. Select an image.','export_path':'Export folder selected.',
  'working':'Processing. Predictions appear first, followed by CAM and SHAP.','classification':'Prediction complete; explanations are running.',
  'cam_ready':'CAM complete; waiting for remaining explanations.','done':'Complete: images, results and explanations exported.',
  'cancelled':'Cancelled. {detail}','failed':'Processing failed. {detail}','closing':'Cancelling; the window will close after status is saved.',
  'error':'Error','read_error':'Unable to read image','choose_first':'Select a static UC endoscopy image first.',
  'invalid_budget':'SHAP permutations must be an integer from 1 to 128.','endoscopy_files':'Endoscopy images','all_files':'All files',
  'predicted':'Predicted grade','raw':'Raw probabilities','calibrated':'Validation temperature-scaled probabilities','temperature':'Temperature',
  'boundary':'Probability of MES≥2','precision':'Numerical profile','cpu':'CPU FP32','gpu':'GPU TF32 / fixed batch 32',
  'folder':'Result folder','cam_note':'CAM is ready. The boundary heatmap is also saved in the export folder.',
  'no_shap':'SHAP has not been computed for this run.','class_target':'Predicted class','boundary_target':'MES≥2',
  'shap_summary':'Regional SHAP contributions are ready.',
  'region':'Row {row}, column {column}','remaining':'Sum of the other 54 regions','profile_note':'CPU versus historical GPU differences are documented in the acceptance record.',
  'report_title':'Static UC Endoscopy Image Report','probability':'Class probabilities','target':'Explanation target','contribution':'SHAP contribution',
  'startup_failure':'Unable to start. Check that the software folder is complete and review the background error log.',
 }}
assert set(TEXT)=={'en'}

def tr(language,key,**values):return TEXT[language][key].format(**values)

def validate_manifest(manifest):
    data=json.loads(Path(manifest).read_text(encoding='utf-8'))
    if set(data['models'])!={MODEL_ID}:raise ValueError('The model manifest does not match this software package.')
    entry=data['models'][MODEL_ID]
    if (entry['backbone'],entry['variant'],entry['seed'])!=('resnet18','H',29):raise ValueError('The model metadata does not match this software package.')
    return data

def result_text(result,explanation,language,destination=None):
    t=lambda key,**kw:tr(language,key,**kw)
    c=result['raw_probabilities']
    lines=[f"{t('predicted')}: MES {result['predicted_mes']}",t('probability')+':',
           *[f'  MES {k}: {value:.2%}' for k,value in enumerate(c)],
           f"{t('boundary')}: {result['q_raw']:.2%}"]
    if explanation:lines.append(t('shap_summary',count=explanation['permutations'],seconds=explanation['seconds'],residual=explanation['max_abs_residual']))
    else:lines.append(t('no_shap'))
    if 'cam_predicted' in result:lines.append(t('cam_note'))
    if destination:lines.append(f"{t('folder')}: {Path(destination).name}")
    return '\n'.join(lines)

def write_reports(destination,result,explanation):
    for language in TEXT:
        t=lambda key,**kw:tr(language,key,**kw)
        title=t('report_title');text=html.escape(result_text(result,explanation,language))
        panels=''.join(f'<figure><img src="{file}"><figcaption>{html.escape(t(label))}</figcaption></figure>' for file,label in
                       [('original.png','original'),('cam_predicted.png','cam')]+([('shap_regions.png','shap_image')] if explanation else []))
        rows=''
        if explanation:
            for target,key in [('predicted_class','class_target'),('MES>=2','boundary_target')]:
                for r in explanation['top10'][target]:
                    rows+=f"<tr><td>{html.escape(t(key))}</td><td>{html.escape(t('region',row=r['row'],column=r['column']))}</td><td>{r['shap_value']:+.7f}</td></tr>"
                rows+=f"<tr><td>{html.escape(t(key))}</td><td>{html.escape(t('remaining'))}</td><td>{explanation['remaining54_sum'][target]:+.7f}</td></tr>"
        document=f'<!doctype html><html lang="{language}"><meta charset="utf-8"><title>{html.escape(title)}</title><style>body{{font:16px Arial,sans-serif;max-width:1200px;margin:30px auto}}pre{{white-space:pre-wrap;line-height:1.6}}.images{{display:flex;gap:12px}}figure{{margin:0;flex:1}}img{{max-width:100%}}td,th{{padding:6px 14px;border:1px solid #ccc}}table{{border-collapse:collapse}}</style><h1>{html.escape(title)}</h1><pre>{text}</pre><div class="images">{panels}</div><h2>{html.escape(t("top10"))}</h2><table><tr><th>{html.escape(t("target"))}</th><th>{html.escape(t("class_region"))}</th><th>{html.escape(t("contribution"))}</th></tr>{rows}</table><p>{html.escape(t("note"))}</p></html>'
        (Path(destination)/f'report_{language}.html').write_text(document,encoding='utf-8')

class Desktop:
    def __init__(self,root,manifest,language='en'):
        validate_manifest(manifest)
        self.root=root;self.manifest=Path(manifest);self.language=language;self.translated=[]
        self.events=queue.Queue();self.cancel=threading.Event();self.worker=None
        self.image=None;self.output=None;self.result=None;self.explanation=None;self.destination=None
        self.status_key='ready';self.status_values={}
        width=min(1280,root.winfo_screenwidth()-80);height=min(920,root.winfo_screenheight()-100)
        root.geometry(f'{width}x{height}+25+25');root.minsize(1000,720)
        body=ttk.Frame(root,padding=12);body.pack(fill='both',expand=True)
        top=ttk.Frame(body);top.pack(fill='x')
        self.label(top,'title',font=('Segoe UI',13,'bold')).pack(side='left')
        self.label(body,'scope').pack(anchor='w',pady=(6,8))
        actions=ttk.Frame(body);actions.pack(fill='x',pady=5)
        self.open_button=self.button(actions,'open',self.choose)
        self.predict_button=self.button(actions,'predict',lambda:self.start(False))
        self.shap_button=self.button(actions,'explain',lambda:self.start(True))
        self.cancel_button=self.button(actions,'cancel',self.cancel.set)
        self.output_button=self.button(actions,'output',self.choose_output)
        self.path_var=tk.StringVar();ttk.Label(body,textvariable=self.path_var,wraplength=width-40).pack(anchor='w',pady=4)
        self.status=tk.StringVar();ttk.Label(body,textvariable=self.status,wraplength=width-40).pack(anchor='w',pady=4)
        panel=ttk.Frame(body);panel.pack(fill='x',pady=8);self.images=[]
        for key in ['original','cam','shap_image']:
            frame=ttk.LabelFrame(panel,padding=5);self.translated.append((frame,'text',key));frame.pack(side='left',fill='both',expand=True,padx=3)
            image=ttk.Label(frame,anchor='center');image.pack();self.images.append(image)
        bottom=ttk.Frame(body);bottom.pack(fill='both',expand=True)
        left=ttk.LabelFrame(bottom,padding=7);self.translated.append((left,'text','result'));left.pack(side='left',fill='both',expand=True,padx=(0,5))
        self.result_widget=tk.Text(left,wrap='word',width=58,height=15,font=('Segoe UI',10));self.result_widget.pack(fill='both',expand=True)
        right=ttk.LabelFrame(bottom,padding=7);self.translated.append((right,'text','top10'));right.pack(side='left',fill='both',expand=True)
        self.columns=['rank','class_region','class_value','boundary_region','boundary_value']
        self.table=ttk.Treeview(right,columns=self.columns,show='headings',height=10)
        for column in self.columns:self.table.column(column,width=105 if column!='rank' else 38,anchor='center',stretch=True)
        self.table.pack(fill='both',expand=True)
        self.label(body,'note',wraplength=width-40).pack(anchor='w',pady=6)
        self.set_language(language);root.after(100,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def label(self,parent,key,**kw):
        widget=ttk.Label(parent,**kw);self.translated.append((widget,'text',key));return widget
    def button(self,parent,key,command):
        widget=ttk.Button(parent,command=command);self.translated.append((widget,'text',key));widget.pack(side='left',padx=3);return widget
    def t(self,key,**kw):return tr(self.language,key,**kw)
    def set_status(self,key,**values):
        self.status_key=key;self.status_values=values;self.status.set(self.t(key,**values))
    def set_language(self,language):
        if language != 'en':raise ValueError('Only English is supported.')
        self.language='en'
        self.root.title(self.t('title'))
        for widget,option,key in self.translated:widget.configure(**{option:self.t(key)})
        for column in self.columns:self.table.heading(column,text=self.t(column))
        self.path_var.set(self.image.name if self.image else self.t('no_image'))
        self.status.set(self.t(self.status_key,**self.status_values));self.render()
    def render(self):
        self.result_widget.delete('1.0','end');self.table.delete(*self.table.get_children())
        if not self.result:return
        self.result_widget.insert('1.0',result_text(self.result,self.explanation,self.language,self.destination))
        if self.explanation:
            a=self.explanation['top10']['predicted_class'];b=self.explanation['top10']['MES>=2']
            for i,(x,y) in enumerate(zip(a,b)):
                self.table.insert('','end',values=(i+1,f"R{x['row']}C{x['column']}",f"{x['shap_value']:+.5f}",f"R{y['row']}C{y['column']}",f"{y['shap_value']:+.5f}"))
    def show(self,index,path):
        with Image.open(path) as image:
            image=image.convert('RGB');image.thumbnail((350,225));photo=ImageTk.PhotoImage(image)
        self.images[index].configure(image=photo);self.images[index].image=photo
    def busy(self,active):
        for widget in [self.open_button,self.predict_button,self.shap_button,self.output_button]:widget.configure(state='disabled' if active else 'normal')
    def choose(self):
        path=filedialog.askopenfilename(title=self.t('open'),filetypes=[(self.t('endoscopy_files'),'*.png *.jpg *.jpeg *.bmp *.tif *.tiff'),(self.t('all_files'),'*.*')])
        if not path:return
        try:
            with Image.open(path) as image:image.verify()
        except Exception as exc:messagebox.showerror(self.t('read_error'),str(exc));return
        self.image=Path(path);self.path_var.set(self.image.name);self.result=None;self.explanation=None;self.destination=None
        self.render();self.show(0,path)
        for image in self.images[1:]:image.configure(image='')
        self.set_status('ready')
    def choose_output(self):
        path=filedialog.askdirectory(title=self.t('output'))
        if path:self.output=Path(path);self.set_status('export_path',path=path)
    def start(self,explain):
        if self.worker and self.worker.is_alive():return
        if not self.image:messagebox.showinfo(self.t('open'),self.t('choose_first'));return
        if self.output is None:
            self.choose_output()
            if self.output is None:return
        permutations=64
        self.result=None;self.explanation=None;self.render()
        for image in self.images[1:]:image.configure(image='')
        self.cancel.clear();self.busy(True);self.set_status('working')
        source=self.image;destination=self.output/('analysis_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns())[-6:])
        def work():
            try:
                predictor=Predictor(self.manifest,MODEL_ID)
                result=predictor.export(source,destination,explain,permutations,self.cancel,
                                        lambda stage,result:self.events.put((stage,destination,result)))
                explanation=json.loads((destination/'shap.json').read_text(encoding='utf-8')) if explain else None
                write_reports(destination,result,explanation)
                self.events.put(('done',destination,result,explanation))
            except Exception as exc:
                if destination.exists():save_json(destination/'task_error.json',{'error':str(exc),'status':'cancelled' if isinstance(exc,Cancelled) else 'failed'})
                self.events.put(('cancelled' if isinstance(exc,Cancelled) else 'failed',str(exc)))
        self.worker=threading.Thread(target=work,daemon=True);self.worker.start()
    def poll(self):
        try:
            while True:
                event=self.events.get_nowait();stage=event[0]
                if stage=='classification':
                    self.result=event[2];self.destination=event[1];self.render();self.set_status('classification')
                elif stage=='cam':
                    self.result=event[2];self.render();self.show(1,event[1]/'cam_predicted.png');self.set_status('cam_ready')
                elif stage=='done':
                    self.busy(False);self.load_result(event[1]);self.set_status('done')
                else:
                    self.busy(False)
                    self.set_status(stage,detail='')
        except queue.Empty:pass
        self.root.after(100,self.poll)
    def load_result(self,destination):
        self.destination=Path(destination);self.result=json.loads((self.destination/'result.json').read_text(encoding='utf-8'))
        if self.result['model_id']!=MODEL_ID or self.result['status']!='complete':raise ValueError('Invalid H29 result')
        file=self.destination/'shap.json';self.explanation=json.loads(file.read_text(encoding='utf-8')) if file.exists() else None
        self.image=self.destination/'original.png';self.path_var.set(self.image.name)
        self.show(0,self.image);self.show(1,self.destination/'cam_predicted.png')
        if self.explanation:self.show(2,self.destination/'shap_regions.png')
        self.render()
    def close(self):
        self.cancel.set()
        if self.worker and self.worker.is_alive():self.set_status('closing');self.root.after(150,self.close)
        else:self.root.destroy()

def resources():return Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',default=str(resources()/'models/manifest.json'))
    parser.add_argument('--language',choices=['en'],default='en');parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--image');parser.add_argument('--output');parser.add_argument('--explain',action='store_true')
    parser.add_argument('--permutations',type=int,default=64);parser.add_argument('--screenshots')
    args=parser.parse_args();validate_manifest(args.manifest)
    if args.selftest and args.image:
        if not args.output:raise ValueError('--output required')
        result=Predictor(args.manifest,MODEL_ID).export(args.image,args.output,args.explain,args.permutations)
        file=Path(args.output)/'shap.json';explanation=json.loads(file.read_text(encoding='utf-8')) if file.exists() else None
        write_reports(args.output,result,explanation)
    root=tk.Tk()
    if args.selftest and not args.screenshots:root.withdraw()
    app=Desktop(root,args.manifest,args.language)
    if args.selftest:
        language_checks={}
        if args.image:app.load_result(args.output);app.set_status('done')
        for language in ['en']:
            app.set_language(language);root.update()
            def walk_widgets(widget):
                yield widget
                for child in widget.winfo_children():yield from walk_widgets(child)
            assert not any(isinstance(w,ttk.Combobox) for w in walk_widgets(root)), 'Unexpected language selector'
            assert app.language==language and root.title()==tr(language,'title')
            language_checks[language]={'title':root.title(),'top10_rows':len(app.table.get_children()),
                                       'model_id':app.result['model_id'] if app.result else MODEL_ID}
            if args.screenshots:
                screenshot_dir=Path(args.screenshots);screenshot_dir.mkdir(parents=True,exist_ok=True)
                root.deiconify();root.lift();root.attributes('-topmost',True);root.update();time.sleep(.6)
                left,top=root.winfo_rootx(),root.winfo_rooty()
                ImageGrab.grab(bbox=(left,top,left+root.winfo_width(),top+root.winfo_height()),all_screens=True).save(screenshot_dir/f'interface_{language}.png')
        root.destroy()
        if args.output:save_json(Path(args.output)/'selftest.json',{'status':'passed','frozen':bool(getattr(sys,'frozen',False)),
                    'fixed_model_id':MODEL_ID,'languages':language_checks,'requested_permutations':args.permutations})
        return
    root.mainloop()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--selftest' in sys.argv:
            if '--output' in sys.argv:
                folder=Path(sys.argv[sys.argv.index('--output')+1]);folder.mkdir(parents=True,exist_ok=True)
                save_json(folder/'selftest_failure.json',{'status':'failed','error':str(exc)})
        else:
            try:save_json(resources()/'startup_error.json',{'status':'failed','error':str(exc)})
            except OSError:pass
            language='en'
            messagebox.showerror('Startup failed',tr(language,'startup_failure'))
        sys.exit(1)
