"""Keep original CoGenAV math; represent original <95% face gate as UNMEASURABLE."""
import json,sys,time
from pathlib import Path
from prepare import B,RUN,sha,write



def patch_source(source):
    needle='                raise ValueError(f"Insufficient detected face support: {row[\'id\']}")'
    assert source.count(needle)==1
    source=source.replace(needle,'''                record = _face_unmeasurable(row, detected, len(frames))
                unmeasurable.append(record)
                dump(root / "unmeasurable.json", unmeasurable)
                continue''')
    needle='            landmarks = detector(frames)'
    assert source.count(needle)==1
    source=source.replace(needle,'''            try:
                landmarks = detector(frames)
            except AssertionError as error:
                if "Cannot detect any frames" not in str(error):
                    raise
                record = _face_unmeasurable(row, 0, len(frames))
                unmeasurable.append(record)
                dump(root / "unmeasurable.json", unmeasurable)
                continue''')
    needle='        unmeasurable = eligibility["inputs"]'
    assert source.count(needle)==1
    source=source.replace(needle,'''        unmeasurable = eligibility["inputs"] + _known_face_unmeasurable()''')
    source=source.replace('"status": "UNMEASURABLE_NATIVE30SEC"','"status": "UNMEASURABLE_NATIVE_PREPROCESSING"')
    return source


def main():
    original=B/'legacy_snapshot/cogenav_sync_audit_20261024.py'
    root=RUN/'cogenav';book=root/'unmeasurable_faces03.json'
    before={str(p):sha(p) for p in (root/'embeddings').glob('*.npz')}
    old_scores=json.loads((root/'scores.json').read_text()) if (root/'scores.json').exists() else []
    if not book.exists():write(book,[])
    def known():return json.loads(book.read_text())
    def missing(row,detected,total):
        record=dict(id=row['id'],reason='INSUFFICIENT_DETECTED_FACE_SUPPORT_LT95_PERCENT',detected=detected,
                    decoded_frames=total,ratio=detected/total if total else None,threshold=.95,
                    audio_sha256=row['audio_sha256'],video_sha256=row['video_sha256'],
                    no_model_forward_for_this_input=True)
        existing=known();assert row['id'] not in {x['id'] for x in existing}
        existing.append(record);write(book,existing);return record
    write(root/'support_adapter03_binding.json',dict(original=str(original),original_sha256=sha(original),adapter_sha256=sha(__file__),
        scope='Only availability handling at unchanged95percent/no-detection gate; pretrained weights, cropper, forward, controls, score reductions and protocol/cache hashes unchanged',
        initial_unmeasurable=known(),cached_npz_before=before,existing_score_rows=len(old_scores),time=time.time()))
    ns=dict(__name__='__main__',__file__=str(original),_known_face_unmeasurable=known,_face_unmeasurable=missing)
    sys.argv=[str(original),*sys.argv[1:]]
    exec(compile(patch_source(original.read_text()),str(original),'exec'),ns)
    assert all(sha(p)==value for p,value in before.items())
    current={x['id']:x for x in json.loads((root/'scores.json').read_text())}
    assert all(current[x['id']]==x for x in old_scores)
    write(root/'support_adapter03_audit.json',dict(status='PASS',previous_scores_exactly_preserved=len(old_scores),
        preserved_npz=len(before),unmeasurable=known(),model_code_unchanged=True))
if __name__=='__main__':main()
