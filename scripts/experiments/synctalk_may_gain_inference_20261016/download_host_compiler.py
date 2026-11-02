"""Extractable Ubuntu GCC12 packages; no system installation or sudo."""
import hashlib,json
from pathlib import Path
from urllib.parse import quote
import requests
out=Path('runs/synctalk_may_gain_inference_20261016/resources/gcc12');out.mkdir(exist_ok=True)
expected={
'cpp-12':'8767472d367849832cfd3c6ff72189411f46676a90718d23ce3a7950cd3e009c5fb7b97852fd17ff8bc73384f942468d65d9a11eb5ac1e5f93bd9fe9e7a8d83a',
'g++-12':'688ac01da012346467443fe1904676f37ec4271335d4ce5d6b0e75496b17ae6fc580de97e5037e449d0533ab371d223e17ef921583d371ea3fbbd08a87ae8b01',
'gcc-12':'8abbe291a0102e5e5bfcd86ad585f85574c829793708d100b9ccb782766e5dc5bff12094b8780787f81910f785d7d00861e1185cbb06d32dd0276a67143a0ed1',
'libgcc-12-dev':'fa6b797b1d6fb83a32b47d61bb0c227196fee6e2df3cd9dbdcc5d234515353e6e2d73b2e237eab6f45dbbf6faed2473ff428e9912bb9b71228ac2403a95b57d5',
'libstdc++-12-dev':'94d4d84803d23c40911373e7bbf7ac38e825d183f164d6803f1824e649bdc80555ba55254939864084fffbd65d132a74f59d50d3c1f31473936a29ed40275343'}
rows=[]
for package,digest in expected.items():
 name=package+'_12.4.0-2ubuntu1~24.04.1_amd64.deb'
 url='https://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-12/'+quote(name)
 r=requests.get(url,stream=True,timeout=90);r.raise_for_status();p=out/name;h=hashlib.sha512()
 with p.open('wb') as f:
  for b in r.iter_content(1048576):f.write(b);h.update(b)
 assert h.hexdigest()==digest,'Ubuntu apt metadata SHA512 mismatch'
 rows.append(dict(path=str(p.resolve()),source=url,sha512=h.hexdigest(),bytes=p.stat().st_size))
 print(name,p.stat().st_size,flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
