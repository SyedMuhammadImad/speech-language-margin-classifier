"""Explicit hinge-loss linear classifiers with binary label decoding and one-vs-rest extension."""
import numpy as np

def matrix(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():raise ValueError('Expected finite nonempty feature matrix')
    return x

class LinearSVM:
    def __init__(self,learning_rate=.001,lambda_param=.01,n_iters=500):
        if not np.isfinite([learning_rate,lambda_param]).all() or learning_rate<=0 or lambda_param<0 or type(n_iters) is not int or n_iters<1:
            raise ValueError('Invalid optimization settings')
        self.lr,self.lambda_param,self.n_iters=learning_rate,lambda_param,n_iters
        self.w=None;self.b=0.

    @staticmethod
    def loss_gradient(x,y,w,b,regularization):
        margins=y*(x@w+b);active=margins<1
        loss=regularization*np.dot(w,w)/2+np.maximum(0,1-margins).mean()
        dw=regularization*w-(x[active].T@y[active])/len(x)
        db=-y[active].sum()/len(x)
        return float(loss),dw,float(db)

    def fit(self,x,y):
        x=matrix(x);y=np.asarray(y)
        if y.ndim!=1 or len(y)!=len(x):raise ValueError('Target shape mismatch')
        if y.dtype.kind=='O' and all(isinstance(v,str) for v in y):y=y.astype(str)
        if y.dtype.kind not in 'iuUS':raise ValueError('Labels must be integers or strings')
        self.classes_,encoded=np.unique(y,return_inverse=True)
        if len(self.classes_)!=2:raise ValueError('Binary classifier needs exactly two classes')
        target=np.where(encoded==0,-1.,1.)
        self.w=np.zeros(x.shape[1]);self.b=0.;rng=np.random.default_rng(42)
        self.loss_history=[]
        # Retain the explicit per-example SGD hinge update; shuffle reproducibly.
        for _ in range(self.n_iters):
            for i in rng.permutation(len(x)):
                active=target[i]*(x[i]@self.w+self.b)<1
                grad=self.lambda_param*self.w-(target[i]*x[i] if active else 0)
                self.w-=self.lr*grad
                if active:self.b+=self.lr*target[i]
            loss,_,_=self.loss_gradient(x,target,self.w,self.b,self.lambda_param)
            if not np.isfinite(loss) or not np.isfinite(self.w).all():raise ValueError('Optimization diverged')
            self.loss_history.append(loss)
        return self

    def decision_function(self,x):
        if self.w is None:raise ValueError('Fit before predicting')
        x=matrix(x)
        if x.shape[1]!=len(self.w):raise ValueError('Feature count mismatch')
        result=x@self.w+self.b
        if not np.isfinite(result).all():raise ValueError('Prediction overflow')
        return result

    def predict(self,x):
        margins=self.decision_function(x)
        return self.classes_[(margins>0).astype(int)]

class OneVsRestSVM:
    def __init__(self,**settings):self.settings=settings;self.models=[]
    def fit(self,x,y):
        x=matrix(x);y=np.asarray(y)
        if y.ndim!=1 or len(y)!=len(x):raise ValueError('Target shape mismatch')
        if y.dtype.kind=='O' and all(isinstance(v,str) for v in y):y=y.astype(str)
        if y.dtype.kind not in 'iuUS':raise ValueError('Labels must be integers or strings')
        self.classes_=np.unique(y)
        if len(self.classes_)<2:raise ValueError('Need at least two classes')
        self.models=[LinearSVM(**self.settings).fit(x,(y==label).astype(int)) for label in self.classes_]
        return self
    def decision_function(self,x):
        if not self.models:raise ValueError('Fit before predicting')
        return np.column_stack([model.decision_function(x) for model in self.models])
    def predict(self,x):
        margins=self.decision_function(x)
        return self.classes_[margins.argmax(axis=1)]
