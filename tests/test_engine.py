import numpy as np
import pytest
from scipy import stats
from engine import Model, mean_inference, export_frame
@pytest.mark.parametrize('model,mean,var',[(Model(),30,21),(Model('Poisson'),4,4),(Model('Normal',mu=5,sigma=2),5,4)])
def test_theory(model,mean,var):
 assert np.allclose(model.moments,(mean,var))
 rng=np.random.default_rng(33); a=np.concatenate([model.draw(rng,1000) for _ in range(100)])
 assert abs(a.mean()-mean)<6*np.sqrt(var/len(a))
 assert abs(a.var()-var)/var<.04
@pytest.mark.parametrize('p',[0.,1.])
def test_deterministic(p):
 m=Model(p=p);a=m.draw(np.random.default_rng(1),100)
 assert np.all(a==100*p) and np.isnan(m.zscore(a)).all()
 assert np.all(m.tail_probability(a)==1) and mean_inference(m,a)['pvalue']==1

def test_thresholds():
 m=Model('Normal')
 assert np.isclose(m.tail_probability(2.5),.012419330651552265)
 assert np.isclose(m.tail_probability(stats.norm.ppf(.995)),.01)
@pytest.mark.parametrize('m',[Model(n=15,p=.1),Model('Poisson',lam=.3),Model('Poisson',lam=100)])
def test_false_alarm(m):
 x=np.arange(1000);tails=m.tail_probability(x)
 for alpha in [.1,.05,.01]: assert m.distribution.pmf(x)[tails<alpha].sum()<=alpha+1e-10

def test_reproducibility():
 m=Model();a=m.draw(np.random.default_rng(2026),1000);b=m.draw(np.random.default_rng(2026),1000)
 np.testing.assert_array_equal(a,b)
 assert np.isclose(export_frame(m,a).media_acumulada.iloc[-1],a.mean())

def test_inference():
 r=mean_inference(Model('Normal',sigma=2),[1]*100)
 assert np.isclose(r['pvalue'],2*stats.norm.sf(5))
 assert np.isclose(r['upper']-1,stats.norm.ppf(.975)*.2)
 b=mean_inference(Model(n=10,p=.5),[5]*10)
 assert b['pvalue']==1 and b['lower']<5<b['upper']
 q=mean_inference(Model('Poisson'),[0]*10)
 assert q['lower']==0 and q['upper']>0
@pytest.mark.parametrize('kwargs',[{'p':2},{'sigma':0},{'lam':-1},{'n':1.5},{'kind':'bad'}])
def test_validation(kwargs):
 with pytest.raises(ValueError): Model(**kwargs)
