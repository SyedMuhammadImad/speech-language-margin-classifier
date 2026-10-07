"""Three-speaker train/validation/test evaluation; test speakers never select model settings."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from audio_features import discover,load_features,features,measure,speaker_folds
from linear_margin import OneVsRestSVM,LinearSVM

def save_checkpoint(path,model,scaler):
    np.savez(path,weights=np.stack([m.w for m in model.models]),bias=np.array([m.b for m in model.models]),
             classes=model.classes_,mean=scaler.mean_,scale=scaler.scale_)

def checkpoint_scores(path,x):
    with np.load(path,allow_pickle=False) as a:
        weights,bias,classes,mean,scale=[a[k].copy() for k in ['weights','bias','classes','mean','scale']]
    if weights.ndim!=2 or mean.shape!=(weights.shape[1],) or scale.shape!=mean.shape or len(classes)!=len(weights) or bias.shape!=(len(weights),):raise ValueError('Invalid checkpoint shapes')
    if not np.isfinite(np.r_[weights.ravel(),bias,mean,scale]).all() or (scale<=0).any():raise ValueError('Invalid checkpoint values')
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(mean) or not np.isfinite(x).all():raise ValueError('Invalid feature input')
    score=((x-mean)/scale)@weights.T+bias
    return classes[score.argmax(axis=1)],score

def predict_file(checkpoint,wav):
    if not Path(checkpoint).is_file():raise ValueError('Checkpoint file does not exist')
    wav=Path(wav)
    if not wav.is_file() or wav.suffix.lower()!='.wav' or wav.stat().st_size>32*1024*1024:raise ValueError('Expected WAV under 32 MB')
    flat,_=features(wav)
    labels,scores=checkpoint_scores(checkpoint,flat[None,:13])
    return int(labels[0]),scores[0].tolist()

def train(data_dir,cache,output,manifest=None):
    records=discover(data_dir,manifest);flat,_,fingerprint=load_features(records,cache)
    encoder=LabelEncoder().fit([r[1] for r in records]);y=encoder.transform([r[1] for r in records]);groups=np.array([r[2] for r in records])
    raw=flat[:,:13].astype(np.float64);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    names=['manual_linear','sklearn_linear','sklearn_rbf'];pooled={name:np.full(len(y),-1) for name in names};folds=[]
    for i,(ti,vi,ei) in enumerate(speaker_folds(y,groups)):
        scaler=StandardScaler().fit(raw[ti]);x,v,e=[scaler.transform(raw[a]) for a in (ti,vi,ei)]
        candidates=[]
        for regularization in [.001,.01,.1]:
            model=OneVsRestSVM(lambda_param=regularization,n_iters=500).fit(x,y[ti])
            val=measure(y[vi],model.predict(v),len(encoder.classes_))
            candidates.append((val['macro_f1'],regularization,model,val))
        best=max(candidates,key=lambda c:(c[0],c[1]));models={'manual_linear':best[2]};selected={'manual_linear':{'lambda':best[1],'validation':best[3]}}
        for kernel,name in [('linear','sklearn_linear'),('rbf','sklearn_rbf')]:
            variants=[]
            for c in [.1,1.,10.]:
                model=SVC(C=c,kernel=kernel,gamma='scale').fit(x,y[ti]);val=measure(y[vi],model.predict(v),len(encoder.classes_))
                variants.append((val['macro_f1'],-c,model,val))
            chosen=max(variants,key=lambda c:(c[0],c[1]));models[name]=chosen[2];selected[name]={'C':-chosen[1],'validation':chosen[3]}
        fold={'fold':i+1,'train_speakers':np.unique(groups[ti]).tolist(),'validation_speakers':np.unique(groups[vi]).tolist(),
              'test_speakers':np.unique(groups[ei]).tolist(),'train':len(ti),'validation':len(vi),'test':len(ei),'selected':selected,'models':{}}
        for name,model in models.items():
            prediction=model.predict(e);pooled[name][ei]=prediction;fold['models'][name]=measure(y[ei],prediction,len(encoder.classes_))
        path=output/f'fold{i+1}-manual.npz';save_checkpoint(path,models['manual_linear'],scaler)
        prediction,scores=checkpoint_scores(path,raw[ei])
        np.testing.assert_array_equal(prediction,pooled['manual_linear'][ei]);np.testing.assert_allclose(scores,models['manual_linear'].decision_function(e),atol=1e-12)
        actual_label,_=predict_file(path,records[ei[0]][0]);assert actual_label==int(prediction[0])
        fold['checkpoint_and_single_wav_reload_match']=True;folds.append(fold)
        print('Completed speaker fold '+str(i+1),flush=True)
    assert all((p>=0).all() for p in pooled.values())
    result={'recordings':len(y),'speakers':len(np.unique(groups)),'labels':encoder.classes_.tolist(),'feature_fingerprint':fingerprint,
            'feature_setting':'13 mean MFCCs; mono16k, FFT512, hop256, first30sec',
            'selection':'each fold uses its separate validation speaker; test speaker excluded from scaling, training and selection',
            'folds':folds,'pooled_models':{n:measure(y,p,len(encoder.classes_)) for n,p in pooled.items()},
            'limitation':'three voices, possible shared utterances/channels; no reliable broad deployment claim'}
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--data-dir',required=True);t.add_argument('--cache-dir',required=True);t.add_argument('--output-dir',required=True);t.add_argument('--manifest')
    i=sub.add_parser('predict');i.add_argument('--checkpoint',required=True);i.add_argument('--wav',required=True)
    args=p.parse_args()
    try:
        if args.command=='train':train(args.data_dir,args.cache_dir,args.output_dir,args.manifest)
        else:
            label,scores=predict_file(args.checkpoint,args.wav)
            print(json.dumps({'encoded_class':label,'uncalibrated_margins':scores,'label_order':'read labels in training metrics.json'}))
    except (ValueError,OSError,KeyError) as e:p.exit(2,'Failed: '+str(e)+'\n')

if __name__=='__main__':main()
