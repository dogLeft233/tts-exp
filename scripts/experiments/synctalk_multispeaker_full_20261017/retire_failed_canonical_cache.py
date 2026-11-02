"""Fresh F failure-package and source video proof before retiring two owned SHM copies."""
import argparse,json,shlex,subprocess
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH,HOST,sha
def run(identity='DebbieDingell'):
    assert identity in ['DebbieDingell','CathyMcMorrisRodgers']
    bundle=ROOT/'geometry_diagnostics'/(identity+'_failed_v2_attempt01')/'full_failure_bundle'
    archived=json.loads((bundle/'verification.json').read_text())
    assert archived['passed'] and archived['archive_integrity_verified'] and archived['geometry_qualified'] is False
    assert sha(bundle/'failure.tar.gz')==archived['archive_sha256'] and sha(bundle/'manifest.json')==archived['manifest_sha256']
    manifest=json.loads((bundle/'manifest.json').read_text());g=next(x for x in manifest['files'] if x['path']=='prepared/'+identity+'/geometry_v2/geometry_qualification.json')
    folder=ROOT/'material'/identity;receipt=json.loads((folder/'material_receipt.json').read_text());files=[]
    for role in ['main','test']:
        p=folder/(role+'.mkv');r=next(r for r in receipt['recordings'] if r['role']==role)
        assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        files.append(dict(role=role,path=str(p),bytes=p.stat().st_size,sha256=r['sha256']))
    proof=dict(passed=True,identity=identity,failed_pair_evidence_independently_archived=True,
        failure_evidence_archive_sha256=archived['archive_sha256'],failed_geometry_sha256=g['sha256'],files=files)
    out=folder/'failed_pair_canonical_F_fresh_verification.json';assert not out.exists();out.write_text(json.dumps(proof,indent=2))
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',str(out),HOST+':'+REMOTE+'/material/'+identity+'/failed_pair_canonical_F_fresh_verification.json'],check=True)
    command=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',REMOTE+'/retire_failed_canonical_remote.py','--root',REMOTE,'--proof',REMOTE+'/material/'+identity+'/failed_pair_canonical_F_fresh_verification.json']
    stdout=subprocess.check_output(SSH+[shlex.join(command)]);result=json.loads(stdout);assert result['passed']
    raw=subprocess.check_output(SSH+['cat '+REMOTE+'/material/'+identity+'/failed_pair_canonical_cache_retirement.json']);assert json.loads(raw)==result
    target=folder/'failed_pair_canonical_cache_retirement.json';assert not target.exists();target.write_bytes(raw)
    remote_sha=subprocess.check_output(SSH+['sha256sum '+REMOTE+'/material/'+identity+'/failed_pair_canonical_cache_retirement.json']).decode().split()[0];assert sha(target)==remote_sha
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',default='DebbieDingell');a=p.parse_args();run(a.identity)
