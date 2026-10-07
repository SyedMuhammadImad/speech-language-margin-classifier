import numpy as np
import pytest
import torch
from linear_margin import *

def test_hinge_weight_and_bias_gradient_against_finite_differences_and_autograd():
    rng=np.random.default_rng(4);x=rng.normal(size=(10,4));y=np.where(np.arange(10)%2,-1.,1.);w=rng.normal(size=4);b=.21
    loss,gradient,bgrad=LinearSVM.loss_gradient(x,y,w,b,.03)
    numerical=np.zeros_like(w)
    for i in range(len(w)):
        plus=w.copy();minus=w.copy();plus[i]+=1e-6;minus[i]-=1e-6
        numerical[i]=(LinearSVM.loss_gradient(x,y,plus,b,.03)[0]-LinearSVM.loss_gradient(x,y,minus,b,.03)[0])/2e-6
    np.testing.assert_allclose(gradient,numerical,atol=1e-8)
    assert abs(bgrad-(LinearSVM.loss_gradient(x,y,w,b+1e-6,.03)[0]-LinearSVM.loss_gradient(x,y,w,b-1e-6,.03)[0])/2e-6)<1e-8
    tw=torch.tensor(w,requires_grad=True);tb=torch.tensor(b,dtype=torch.float64,requires_grad=True)
    objective=.03*tw.square().sum()/2+torch.relu(1-torch.tensor(y)*(torch.tensor(x)@tw+tb)).mean();objective.backward()
    assert abs(objective.item()-loss)<1e-12
    np.testing.assert_allclose(gradient,tw.grad.numpy());assert abs(bgrad-tb.grad.item())<1e-12

def test_binary_label_decoding_zero_margin_and_multiclass_separation():
    rng=np.random.default_rng(8);x=np.r_[rng.normal(-2,.1,(20,2)),rng.normal(2,.1,(20,2))];y=np.r_[np.full(20,-3),np.full(20,50)]
    model=LinearSVM(n_iters=100).fit(x,y);np.testing.assert_array_equal(model.predict(x),y)
    model.w[:]=0;model.b=0;assert np.all(model.predict(x)==-3)
    x=np.r_[rng.normal((-3,-3),.1,(20,2)),rng.normal((3,-3),.1,(20,2)),rng.normal((0,3),.1,(20,2))]
    y=np.repeat(['English','Mixed','Urdu'],20)
    a=OneVsRestSVM(n_iters=200).fit(x,y);b=OneVsRestSVM(n_iters=200).fit(x,y)
    np.testing.assert_array_equal(a.predict(x),y);np.testing.assert_array_equal(a.decision_function(x),b.decision_function(x))

def test_invalid_settings_labels_features_and_unfit_prediction():
    for settings in [{'learning_rate':0},{'lambda_param':-1},{'n_iters':0},{'learning_rate':float('nan')}]:
        with pytest.raises(ValueError):LinearSVM(**settings)
    with pytest.raises(ValueError):LinearSVM().predict(np.ones((2,2)))
    with pytest.raises(ValueError):LinearSVM().fit(np.ones((3,2)),np.array([0,1,2]))
    with pytest.raises(ValueError):LinearSVM().fit(np.full((2,2),np.inf),np.array([0,1]))
    with pytest.raises(ValueError):OneVsRestSVM().fit(np.ones((3,2)),np.array([0,0,0]))
    with pytest.raises(ValueError):OneVsRestSVM().fit(np.ones((3,2)),np.array([0.,1.,float('nan')]))

def test_checkpoint_retains_training_scaler_and_exact_margins(tmp_path):
    from experiment import save_checkpoint,checkpoint_scores
    from sklearn.preprocessing import StandardScaler
    x=np.array([[-9.,1.],[-8.,2.],[8.,6.],[9.,7.],[0.,20.],[1.,21.]])
    y=np.array([0,0,1,1,2,2]);scaler=StandardScaler().fit(x)
    model=OneVsRestSVM(n_iters=200).fit(scaler.transform(x),y)
    file=tmp_path/'model.npz';save_checkpoint(file,model,scaler)
    probe=np.array([[19.,9.],[-12.,5.],[2.,30.]])
    labels,margins=checkpoint_scores(file,probe)
    np.testing.assert_array_equal(labels,model.predict(scaler.transform(probe)))
    np.testing.assert_allclose(margins,model.decision_function(scaler.transform(probe)),atol=1e-12)
    with pytest.raises(ValueError):checkpoint_scores(file,np.ones((2,3)))
    np.savez(file,weights=np.zeros((3,2)),bias=np.zeros(3),classes=np.arange(3),mean=np.zeros(2),scale=np.array([0.,1.]))
    with pytest.raises(ValueError):checkpoint_scores(file,probe)

def test_speaker_partitions_cover_each_recording_once_and_keep_roles_disjoint():
    from audio_features import speaker_folds
    groups=np.repeat(['a','b','c'],9);y=np.tile(np.repeat([0,1,2],3),3)
    covered=[]
    for train,val,test in speaker_folds(y,groups):
        assert not set(groups[train])&set(groups[val])
        assert not set(groups[train])&set(groups[test])
        assert not set(groups[val])&set(groups[test])
        covered.extend(test.tolist())
    assert sorted(covered)==list(range(len(y)))

def test_cli_reports_missing_checkpoint_without_success(tmp_path):
    import subprocess,sys
    from pathlib import Path
    wav=tmp_path/'clip.wav';wav.write_bytes(b'not a wave')
    p=subprocess.run([sys.executable,str(Path(__file__).with_name('experiment.py')),'predict','--checkpoint',str(tmp_path/'missing.npz'),'--wav',str(wav)],capture_output=True,text=True)
    assert p.returncode==2 and 'Failed:' in p.stderr
    with pytest.raises(ValueError,match='WAV'):
        from audio_features import features
        features(wav)
