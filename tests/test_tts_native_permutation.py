import numpy as np
from scripts.experiments.tts_native_permutation import generate_perms,labels_at,gram,curve_metrics


def test_permutation_regions_occurrences_and_prefix():
    n,k=65,3;t=np.arange(n-k)
    region=np.full(n,-1);region[t]=np.where(t<20,0,np.where(t<n-20,1,2))
    phone=np.arange(n)//4
    for mode in ('whole','phone'):
        pi,_=generate_perms(n,t,region,phone,'fixed',mode,64)
        long,_=generate_perms(n,t,region,phone,'fixed',mode,256)
        assert np.array_equal(pi,long[:64])
        assert np.all(region[pi[:,t]]==region[t])
        assert np.all(np.sort(pi[:,20:n-20])==np.arange(20,n-20))
        if mode=='phone':assert np.all(phone[pi[:,t]]==phone[t])


def test_separate_gap_and_silence_intervals():
    spans=[(0,.1,'a'),(.2,.3,''),(.5,.6,'a')]
    labels,vocab=labels_at(np.array([.15,.25,.35,.45,.65]),spans)
    assert len(set(labels))==4
    assert labels[2]==labels[3]
    assert vocab[labels[1]].startswith('occ_1_')


def test_anchor_multiset_invariant_and_curve_metric():
    n,k=65,3;rng=np.random.default_rng(1)
    v=rng.normal(size=(n-k,1024)).astype(np.float32);a=rng.normal(size=v.shape).astype(np.float32)
    d=gram(v,a);i=np.arange(20,n-20);q=i[:,None]+np.arange(-15,16)-k
    region=np.full(n,-1);t=np.arange(n-k);region[t]=np.where(t<20,0,np.where(t<n-20,1,2))
    pi,_=generate_perms(n,t,region,np.arange(n)//5,'anchor','whole',2)
    base=d[i[:,None],q].mean(0)
    for pp in pi:
        moved=d[pp[i,None],pp[q]]
        assert np.array_equal(np.sort(moved[:,k+15]),np.sort(d[i,i]))
        assert abs(curve_metrics(moved.mean(0),k)[0,3]-curve_metrics(base,k)[0,3])<1e-13


def test_gram_epsilon_matches_direct_vectors():
    rng=np.random.default_rng(20)
    v=rng.normal(size=(5,1024)).astype(np.float32);a=rng.normal(size=(6,1024)).astype(np.float32)
    direct=np.linalg.norm(v.astype(float)[:,None]-a.astype(float)[None]+1e-6,axis=2)
    assert np.max(abs(gram(v,a)-direct))<1e-12
