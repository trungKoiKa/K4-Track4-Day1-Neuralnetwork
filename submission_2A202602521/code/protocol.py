"""Prospective revision protocol. Run stages from lab.ipynb on Colab T4.

Each experiment starts in an independent Python process. Checkpoints are kept
outside the submission and deleted by the operating system's temporary-folder
lifecycle; only the selected validation checkpoint is used for final scoring.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data import prepare_data
from model import MLP, count_params, activation_stats
from train import DEFAULT_CFG, set_seed, run_experiment, evaluate, final_eval
from results_table import save_result, load_results, to_row, write_xlsx
from plots import plot_run, plot_compare

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'scripts/split_data.py').exists())
OUT = ROOT / 'submission_2A202602521'
PLAN = OUT / 'protocol.json'
PREDICTIONS = {
 'baseline': 'SGD momentum: lr thấp học chậm, lr cao có thể dao động; chọn lr bằng F1 val tại epoch min val loss.',
 'loss': 'CE phù hợp phân lớp hơn MSE logits/one-hot; chỉ so accuracy/F1, không so trị số loss khác thang.',
 'optimizer': 'Adam/AdamW có thể hội tụ nhanh hơn SGD; thử cùng lưới lr cho từng optimizer và so lr tốt nhất.',
 'hyperparameter': 'Batch 2048 giảm số bước/epoch bốn lần so batch 512, có thể làm học chậm với lr giữ nguyên.',
 'dropout': 'p=0.3 có thể giảm train-val gap nhưng giảm F1 nếu mạng chưa overfit.',
 'clipping': 'lr gấp 10 có thể gây dao động; clip ở phân vị 90 gradient baseline có thể giảm bước cực lớn, không chắc cứu được phân kỳ.',
 'mixed_precision': 'FP16/BF16 có thể không nhanh hơn với MLP nhỏ; T4 không có BF16 native, nên tốc độ BF16 chỉ là số đo emulation.',
 'initialization': 'He giữ tín hiệu qua ReLU tốt hơn normal nhỏ; zeros không phá đối xứng và gradient hidden bằng 0.',
 'other': 'Lặp cấu hình đã khoá ở seed 2/3 để đo nhiễu; không dùng các lần lặp để chọn lại cấu hình.'}

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def cfg(exp_id, group, **kwargs):
    return {**DEFAULT_CFG, 'exp_id': exp_id, 'group': group,
            'description': PREDICTIONS[group], **kwargs}

def worker(config, checkpoint_dir):
    assert torch.cuda.is_available(), 'This submission must run on a Colab GPU.'
    torch.set_num_threads(2)
    data = prepare_data('cuda')
    r = run_experiment(config, data)
    save_result(r, str(OUT/'results'))
    plot_run(r, str(OUT/'figures'/f"{config['exp_id']}.png"))
    torch.save(r['best_state'], Path(checkpoint_dir)/f"{config['exp_id']}.pt")
    print(config['exp_id'], r['summary'], flush=True)

def run(config):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    # This immutable record is written BEFORE launching the worker.
    write(OUT/'planned'/f"{config['exp_id']}.json", {
      'recorded_before_run_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'prediction': PREDICTIONS[config['group']], 'cfg': config})
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                    json.dumps(config), '--weights', plan['checkpoint_dir']], check=True, cwd=ROOT)

def health():
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    data = prepare_data('cuda'); set_seed(1)
    model = MLP().cuda(); xb, yb = data['X_tr'][:20],data['y_tr'][:20]
    assert count_params(model)==47879 and model(torch.randn(8,54,device='cuda')).shape==(8,7)
    logits = model(xb); F.cross_entropy(logits,yb).backward()
    gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()}
    assert all(v>0 for v in gradients.values())
    opt=torch.optim.Adam(model.parameters(),lr=.01); losses=[]; accuracy=[]
    for step in range(500):
        opt.zero_grad(); loss=F.cross_entropy(model(xb),yb); loss.backward(); opt.step()
        with torch.no_grad():
            losses.append(float(F.cross_entropy(model(xb),yb)))
            accuracy.append(float((model(xb).argmax(1)==yb).float().mean()))
    assert accuracy[-1]==1 and losses[-1]<.001
    fig,axs=plt.subplots(1,2,figsize=(9,3)); axs[0].plot(losses); axs[1].plot(accuracy)
    axs[0].set(xlabel='Step',ylabel='CE loss',yscale='log'); axs[1].set(xlabel='Step',ylabel='Accuracy')
    fig.suptitle('Health: overfit 20 samples, Adam lr=.01, 500 steps')
    fig.tight_layout(); fig.savefig(OUT/'figures/health_overfit20.png',dpi=130); plt.close(fig)
    initial={}
    for init in ['he','xavier','normal','zeros']:
        set_seed(1); m=MLP(init=init).cuda()
        initial[init]={'relu_std':activation_stats(m,data['X_val'][:2048]),
                       'val_step0_ce':evaluate(m,data['X_val'],data['y_val'])['loss']}
    uniform=float(F.cross_entropy(torch.zeros(2048,7,device='cuda'),data['y_val'][:2048]))
    assert abs(uniform-np.log(7))<1e-5
    majority=int(torch.bincount(data['y_tr']).argmax())
    numerical=data['X_tr'][:,:10]
    assert float(numerical.mean(0).abs().max())<1e-5
    assert float((numerical.std(0,correction=0)-1).abs().max())<1e-5
    h={'parameters':47879,'logits_shape':[20,7], 'gradients':gradients,
       'all_parameter_gradients_nonzero':True,'overfit20_loss':losses[-1],
       'overfit20_accuracy':accuracy[-1],'overfit_history':{'loss':losses,'accuracy':accuracy},
       'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(0),
       'bf16_native':torch.cuda.is_bf16_supported(including_emulation=False),
       'majority_val_accuracy':float((data['y_val']==majority).float().mean()),
       'train_numeric_mean_max_abs':float(numerical.mean(0).abs().max()),
       'train_numeric_std_max_error':float((numerical.std(0,correction=0)-1).abs().max()),
       'class_proportions':{k:torch.bincount(data['y_'+k],minlength=7).div(len(data['y_'+k])).tolist() for k in ['tr','val','eval']},
       'uniform_logits_ce':uniform,'initialization':initial}
    write(OUT/'health_checks.json',h); print(json.dumps(h,ensure_ascii=False)[:2400],flush=True)

def setup():
    if PLAN.exists():
        raise RuntimeError('Use a fresh submission directory for a prospective run; existing protocol will not be overwritten.')
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    write(PLAN,{'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'prior_eval_seen':True,'reason':'Repair rubric compliance, not tune to eval scores.',
      'selection':'maximum validation macro-F1 at minimum validation-loss epoch; tie: exp_id',
      'baseline_lr_grid':[.01,.05,.1], 'adam_lr_grid':[.0003,.001,.003],
      'hypotheses':PREDICTIONS,'checkpoint_dir':tempfile.mkdtemp(prefix='lab_revision_weights_'),
      'seed':1,'validation_seed':42,'epochs':20,'batch':512,
      'clip_rule':'baseline batch-gradient 90th percentile; high lr=10*selected baseline lr',
      'final_seeds':[1,2,3], 'no_eval_until_validation_selection':True})
    print('PROSPECTIVE_PLAN_SAVED',PLAN,flush=True)

def pilots():
    for lr in [.01,.05,.1]: run(cfg(f'sgdm-lr{lr:g}','baseline',lr=lr))
    pilots=[r for r in load_results(OUT/'results') if r['cfg']['exp_id'].startswith('sgdm-lr')]
    best=max(pilots,key=lambda r:r['summary']['val_macro_f1'])
    plan=json.loads(PLAN.read_text(encoding='utf-8')); plan['baseline_lr']=best['cfg']['lr']
    write(PLAN,plan); print('BASELINE_LR_SELECTED_BY_VAL',plan['baseline_lr'])

def baseline():
    p=json.loads(PLAN.read_text(encoding='utf-8'))
    for seed in [1,2,3]: run(cfg(f'base-s{seed}','baseline',lr=p['baseline_lr'],seed=seed))

def experiments():
    p=json.loads(PLAN.read_text(encoding='utf-8')); lr=p['baseline_lr']
    base=next(r for r in load_results(OUT/'results') if r['cfg']['exp_id']=='base-s1')
    threshold=base['summary']['grad_p90']; p['clip_threshold']=threshold; write(PLAN,p)
    configs=[cfg('loss-mse','loss',lr=lr,loss='mse'),
      *[cfg(f'{opt}-lr{rate:g}','optimizer',lr=rate,optimizer=opt) for opt in ['adam','adamw'] for rate in [.0003,.001,.003]],
      cfg('batch2048','hyperparameter',lr=lr,batch=2048),
      cfg('dropout-p03','dropout',lr=lr,dropout=.3),
      cfg('clip-normal','clipping',lr=lr,clip_norm=threshold),
      cfg('highlr-none','clipping',lr=lr*10),
      cfg('highlr-clip','clipping',lr=lr*10,clip_norm=threshold),
      cfg('precision-fp16','mixed_precision',lr=lr,precision='fp16'),
      cfg('precision-bf16','mixed_precision',lr=lr,precision='bf16'),
      *[cfg(f'init-{init}','initialization',lr=lr,init=init) for init in ['zeros','normal','xavier']]]
    for config in configs: run(config)

def select():
    p=json.loads(PLAN.read_text(encoding='utf-8')); rs=load_results(OUT/'results')
    eligible=[r for r in rs if r['cfg']['seed']==1 and not r['summary']['diverged']]
    best=sorted(eligible,key=lambda r:(-r['summary']['val_macro_f1'],r['cfg']['exp_id']))[0]
    p['selected_exp_id']=best['cfg']['exp_id']; p['selected_before_eval_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    write(PLAN,p)
    for seed in [2,3]:
        run({**best['cfg'],'exp_id':f'final-s{seed}','group':'other','seed':seed,
             'description':'Fixed validation-selected config repeated for noise; no retuning.'})
    print('FINAL_CONFIG_FROZEN',best['cfg'],flush=True)

def score_worker(exp_id):
    p=json.loads(PLAN.read_text(encoding='utf-8')); data=prepare_data('cuda')
    r=next(r for r in load_results(OUT/'results') if r['cfg']['exp_id']==exp_id)
    r['best_state']=torch.load(Path(p['checkpoint_dir'])/f'{exp_id}.pt',map_location='cuda',weights_only=True)
    is_final=exp_id==p['selected_exp_id']
    suffix='eval' if is_final else 'baseline' if exp_id=='base-s1' else exp_id
    pred=OUT/f'predictions_{suffix}.csv'; target=OUT/('eval_result.json' if is_final else f'eval_{suffix}.json')
    final_eval(r['cfg'],r,data,str(pred))
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate.py'),'--pred',str(pred),'--out',str(target)],check=True)

def score():
    p=json.loads(PLAN.read_text(encoding='utf-8'))
    for exp_id in dict.fromkeys(['base-s1',p['selected_exp_id'],'final-s2','final-s3']):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--score',exp_id],check=True,cwd=ROOT)
    print('FINAL_EVAL_COMPLETE',flush=True)

def stage(name):
    {'setup':setup,'health':lambda:subprocess.run([sys.executable,str(Path(__file__).resolve()),'--health'],check=True,cwd=ROOT),
     'pilots':pilots,'baseline':baseline,'experiments':experiments,'select':select,'score':score}[name]()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--worker'); parser.add_argument('--weights'); parser.add_argument('--health',action='store_true'); parser.add_argument('--score')
    args=parser.parse_args()
    if args.worker: worker(json.loads(args.worker),args.weights)
    elif args.health: health()
    elif args.score: score_worker(args.score)
