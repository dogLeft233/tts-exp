"""Independent direct-vector and squared-distance checks; no scoring module import."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tts_native_midpoint_20260926"
PARENT = ROOT / "runs/tts_native_boundary_audit_20260926"


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):
            h.update(b)
    return h.hexdigest()


def load(d):
    with np.load(d["path"]) as z:
        a=z[d["key"]]
    return a[d.get("start",0):]


errors={"direct_distance":0.,"quadratic_distance":0.,"score":0.,"scales":0.,"statistics":0.}


def close(x,y,name,tol=1e-10):
    e=float(np.max(abs(np.asarray(x)-np.asarray(y))))
    errors[name]=max(errors[name],e)
    assert e<=tol,(name,e)


def boot(values,speakers,result):
    names=sorted(set(speakers))
    y=np.array([np.mean([v for v,s in zip(values,speakers) if s==name]) for name in names])
    close(y.mean(),result["mean"],"statistics")
    assert len(values)==result["n"] and len(names)==result["speakers"]
    for name,value in zip(names,y):
        close(value,result["per_speaker"][name],"statistics")
    ix=np.random.Generator(np.random.PCG64(20260926)).integers(len(names),size=(20000,len(names)))
    weights=np.stack([(ix==j).sum(1) for j in range(len(names))],axis=1)/len(names)
    b=weights@y
    close(np.percentile(b,[.5,99.5]),result["ci99"],"statistics")
    close(np.percentile(b,[2.5,97.5]),result["ci95"],"statistics")


def main():
    protocol=read(OUT/"protocol.json")
    independent_scores={}
    distance_cells=quadratic_cells=0
    files=sorted((OUT/"pairs").glob("*/*/*/*.json"))
    for fi,path in enumerate(files):
        info=read(path)
        group,geom,sid,ta=info["group"],info["geometry"],info["id"],info["T_arm"]
        g=protocol["groups"][group]
        rows={(r["id"],r["arm"]):r for r in g["records"]}
        k=g["k0"]
        coords={}
        for arm in ("N",ta):
            row=rows[sid,arm];n=row["joint_L"]
            v,a=load(row["visual"])[:n],load(row["audio"])[:n]
            if geom=="unit":
                v=(v.astype(float)/np.sqrt(np.sum(v.astype(float)**2,axis=1))[:,None]).astype(np.float32)
                a=(a.astype(float)/np.sqrt(np.sum(a.astype(float)**2,axis=1))[:,None]).astype(np.float32)
            v,a=v.astype(float),a.astype(float)
            domain=np.arange(max(0,-k),min(n,n-k));ix=np.arange(20,n-20)
            m=np.zeros_like(v);r=np.zeros_like(v)
            m[domain]=(v[domain]+a[domain+k])*.5
            r[domain]=(a[domain+k]-v[domain])*.5
            mu=np.sum(m[ix],axis=0)/len(ix)
            sigma=np.sqrt(np.sum((m[ix]-mu)**2)/len(ix))
            rho=np.sqrt(np.sum(r[ix]**2)/len(ix))
            coords[arm]=(v,a,m,r,mu,sigma,rho,domain,ix)
        alpha=coords["N"][5]/coords[ta][5];beta=coords["N"][6]/coords[ta][6]
        close(alpha,info["alpha_T2N"],"scales")
        close(beta,info["beta_T2N"],"scales")
        conditions={"N_base":("N",1.,1.),"T_base":(ta,1.,1.),
                    "T_M":(ta,alpha,1.),"T_R":(ta,1.,beta),"T_both":(ta,alpha,beta),
                    "N_M":("N",1/alpha,1.),"N_R":("N",1.,1/beta),"N_both":("N",1/alpha,1/beta),
                    "N_dose075":("N",.75,1.),"N_dose125":("N",1.25,1.),
                    "T_dose075":(ta,.75,1.),"T_dose125":(ta,1.25,1.)}
        assert sha(path.with_suffix(".npz"))==info["matrix_sha256"]
        calculated={}
        with np.load(path.with_suffix(".npz")) as matrices:
            for label,(arm,aa,bb) in conditions.items():
                v,a,m,r,mu,sigma,rho,domain,ix=coords[arm]
                # Algebraically independent expansion, not M' +/- R' implementation.
                vv=v.copy();av=a.copy()
                vv[domain]=.5*(aa+bb)*v[domain]+.5*(aa-bb)*a[domain+k]+(1-aa)*mu
                av[domain+k]=.5*(aa-bb)*v[domain]+.5*(aa+bb)*a[domain+k]+(1-aa)*mu
                vf,af=vv.astype(np.float32),av.astype(np.float32)
                target=matrices[label]
                for column,s in enumerate(range(-15,16)):
                    d=vf[ix]-af[ix+s]+np.float32(1e-6)
                    direct=np.sqrt(np.sum(d*d,axis=1,dtype=np.float32))
                    close(direct,target[:,column],"direct_distance",2e-5)
                    dm=m[ix]-m[ix+s-k];rs=r[ix]+r[ix+s-k]
                    squared=(aa*aa*np.sum(dm*dm,axis=1)+bb*bb*np.sum(rs*rs,axis=1)
                             -2*aa*bb*np.sum(dm*rs,axis=1)
                             +2e-6*(aa*dm.sum(1)-bb*rs.sum(1))+1024e-12)
                    close(np.sqrt(np.maximum(squared,0)),target[:,column],"quadratic_distance",2e-5)
                distance_cells+=target.size;quadratic_cells+=target.size
                curve=target.sum(axis=0)/len(target)
                j=int(np.argmin(curve));d=float(curve[j]);b=float(np.sort(curve)[15])
                result=dict(C=b-d,B=b,D=d,D_anchor=float(curve[k+15]),best_lag=j-15,offset=15-j,curve=curve.tolist())
                for field,value in result.items():
                    close(value,info["scores"][label][field],"score")
                calculated[label]=result
                mp=(av[ix+k]+vv[ix])/2;rp=(av[ix+k]-vv[ix])/2
                sm=np.sqrt(np.sum((mp-mp.mean(0))**2)/len(ix));sr=np.sqrt(np.sum(rp**2)/len(ix))
                close(sm,info["manipulation"][label]["sigma_after"],"scales")
                close(sr,info["manipulation"][label]["rho_after"],"scales")
        independent_scores[group,geom,sid,ta]=calculated
        if (fi+1)%100==0:
            print("independently checked",fi+1,"pairs",flush=True)
    summary=read(OUT/"summary.json")
    for key,result in summary.items():
        group,geom,support=key.split("/");g=protocol["groups"][group]
        ids=result["ids"];arms=g["arms"][1:]
        rows={r["id"]:r for r in g["records"]};speakers=[rows[sid]["speaker"] for sid in ids]
        def aggregate(fun,expected):
            values=[np.mean([fun(independent_scores[group,geom,sid,arm]) for arm in arms]) for sid in ids]
            boot(values,speakers,expected)
        for label,fields in result["scores"].items():
            for field,expected in fields.items():
                aggregate(lambda ss,l=label,f=field:ss[l][f],expected)
        for eff,fields in result["effects"].items():
            direction,condition=eff.split("/")
            tar=("T_" if direction=="T2N" else "N_")+condition
            base="T_base" if direction=="T2N" else "N_base"
            for field in ("C","B","D","D_anchor"):
                aggregate(lambda ss,t=tar,b=base,f=field:ss[t][f]-ss[b][f],fields[field]["treatment"])
                aggregate(lambda ss,t=tar,di=direction,f=field:ss[t][f]-ss["N_base"][f] if di=="T2N" else ss["T_base"][f]-ss[t][f],fields[field]["residual_TminusN"])
            aggregate(lambda ss,t=tar,b=base:float(ss[t]["best_lag"]!=ss[b]["best_lag"]),fields["lag_changed_fraction"])
        for key,fields in result["dose"].items():
            arm,dose=key.split("/")
            for field,expected in fields.items():
                aggregate(lambda ss,a=arm,d=dose,f=field:ss[a+"_"+d][f]-ss[a+"_base"][f],expected)
    for p,h in read(PARENT/"source_hashes.json").items():
        assert sha(p)==h,p
    result={"status":"PASS","pair_geometries":len(files),"direct_distance_cells":distance_cells,
            "quadratic_distance_cells":quadratic_cells,"max_errors":errors,
            "source_hashes_verified":len(read(PARENT/"source_hashes.json")),
            "method":"Independent linear expanded vector reconstruction, NumPy float32 distances, float64 squared-distance polynomial including epsilon, fixed-row score and independent speaker/paired bootstrap multiplicities"}
    (OUT/"independent_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
