"""Preserve selected original WAVs and upstream metadata without changing inference."""
import json
import shutil
from prepare import CLOUD,DATA,RUN,sha,write

def main():
    rows=json.loads((RUN/'inputs.json').read_text())['rows'];records=[]
    for row in rows:
        original=row['original_audio_path'];assert sha(original)==row['original_audio_sha256']
        target=RUN/'original_audio'/(row['id']+'.wav');target.parent.mkdir(exist_ok=True)
        assert not target.exists();shutil.copyfile(original,target)
        assert sha(target)==row['original_audio_sha256']
        records.append(dict(id=row['id'],original_path=original,archived_path=str(target.relative_to(RUN)),sha256=sha(target),scalar_gain=row['scalar_gain']))
    sources=[DATA/'manifest_release.json',DATA/'manifest_release.csv',DATA/'hdtf_gender_matched_20261031/assignments.json',
        CLOUD/'protocol.json',CLOUD/'verification.json',CLOUD/'audio_manifest.json']
    sources += [CLOUD/'records'/(source+'.json') for source in sorted({r['source'] for r in rows})]
    bindings=[]
    for i,path in enumerate(sources):
        target=RUN/'upstream_metadata'/f'{i:03d}_{path.name}';target.parent.mkdir(exist_ok=True)
        assert not target.exists();shutil.copyfile(path,target)
        bindings.append(dict(original_path=str(path),archived_path=str(target.relative_to(RUN)),sha256=sha(target)))
    write(RUN/'upstream_provenance.json',dict(status='PASS',original_audio=records,metadata=bindings,
        scope='Read-only copies of upstream science inputs. No API calls or changes to models/media/scores.'))
    print('UPSTREAM_PROVENANCE_PASS',len(records),len(bindings))

if __name__=='__main__':main()
