import numpy as np

from scripts.experiments.tts_native_midpoint import coordinates, distances, transform


def pair(n=67):
    rng = np.random.default_rng(17)
    return rng.normal(0,.2,(n,1024)).astype(np.float32), rng.normal(0,.2,(n,1024)).astype(np.float32)


def test_coordinate_extremes_keep_all_scored_rows():
    v,a = pair()
    for k in (-5, 0, 5):
        c = coordinates(v,a,k)
        used = c["i"][:,None]+np.arange(-15,16)-k
        assert used.min() >= c["t"][0] and used.max() <= c["t"][-1]


def test_midpoint_scale_preserves_anchor_residual():
    v,a = pair()
    c = coordinates(v,a,3)
    vv,aa = transform(c,.71,1.)
    i=c["i"]
    assert np.max(abs((aa[i+3]-vv[i])-(a[i+3].astype(float)-v[i]))) < 1e-14
    after = coordinates(vv,aa,3)
    assert abs(after["sigma"]-.71*c["sigma"]) < 1e-13
    assert abs(after["rho"]-c["rho"]) < 1e-13


def test_identity_inverse_and_constant():
    v,a = pair()
    c = coordinates(v,a,-2)
    vv,aa=transform(c,1,1)
    assert np.array_equal(distances(v,a),distances(vv,aa))
    vv,aa=transform(c,.73,1.27)
    invv,inva=transform(coordinates(vv,aa,-2),1/.73,1/1.27)
    assert max(np.max(abs(invv-v)),np.max(abs(inva-a))) < 1e-13
    cv,ca=transform(c,1,1,np.linspace(-.25,.25,1024))
    assert np.max(abs(distances(cv,ca)-distances(v,a))) < 2e-6


def test_quadratic_distance_contains_eps_terms():
    v,a=pair()
    c=coordinates(v,a,1)
    alpha,beta=.75,1.2
    vv,aa=transform(c,alpha,beta)
    i,j=30,24
    dm=c["M"][i]-c["M"][j-1]
    rs=c["R"][i]+c["R"][j-1]
    eps=1e-6
    square=alpha**2*np.dot(dm,dm)+beta**2*np.dot(rs,rs)-2*alpha*beta*np.dot(dm,rs)+2*eps*(alpha*dm.sum()-beta*rs.sum())+len(dm)*eps**2
    assert abs(np.sqrt(square)-np.linalg.norm(vv[i]-aa[j]+eps)) < 1e-13
