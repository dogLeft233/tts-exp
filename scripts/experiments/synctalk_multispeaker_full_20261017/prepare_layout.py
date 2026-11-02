"""Link immutable stage1 inputs into a separate derived-data workspace."""
import argparse,hashlib,json,os
from pathlib import Path

def run(root,identity):
    source=root/'preprocessed'/identity
    stage=json.loads((source/'stage1_receipt.json').read_text());assert stage['passed']
    target=root/'prepared'/identity;assert not target.exists()
    target.mkdir(parents=True);links=0
    for split,count in [('train',6000),('val',750),('test',1500)]:
        for name in ['ori_imgs','parsing','gt_imgs','torso_imgs']:(target/split/name).mkdir(parents=True)
        for i in range(count):
            for name,suffix in [('ori_imgs','.jpg'),('ori_imgs','.lms'),('parsing','.png'),('parsing','_face.png')]:
                p=source/split/name/f'{i}{suffix}';assert p.is_file()
                os.link(p,target/split/name/p.name);links+=1
    receipt=dict(passed=True,identity=identity,immutable_stage1_inputs=str(source),hardlinked_files=links,
        stage1_receipt_sha256=hashlib.sha256((source/'stage1_receipt.json').read_bytes()).hexdigest(),
        contract_sha256=hashlib.sha256((root/'preprocessing_contract_v1.json').read_bytes()).hexdigest(),training_admitted=False)
    (target/'input_binding.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
