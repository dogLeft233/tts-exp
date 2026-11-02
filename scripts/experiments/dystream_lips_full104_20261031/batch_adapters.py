"""Keep audited scorer bodies; load weights once and bind each new input explicitly."""
from pathlib import Path
import argparse, json, sys, traceback, time
from prepare import B, RUN, sha, write

class NativeFaceSupportError(ValueError):
    def __init__(self,detected,total):
        self.detected=detected;self.total=total
        super().__init__(f"Native95percent face support unavailable: {detected}/{total}")

def native_landmarks(detector,frames):
    try:
        marks=detector(frames)
    except AssertionError as error:
        if "Cannot detect any frames" not in str(error):raise
        raise NativeFaceSupportError(0,len(frames)) from error
    detected=sum(m is not None for m in marks)
    if not frames or detected/len(frames)<.95:raise NativeFaceSupportError(detected,len(frames))
    return marks

def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=['avs_u','mtd','vocalist','synchformer','sparsesync']);opt=p.parse_args()
    name=opt.model
    original=B/('score_avs_u.py' if name=='avs_u' else 'score_retrieval.py' if name in ['mtd','vocalist'] else 'score_classifier.py')
    source=original.read_text()
    if name=='avs_u':
        gate='cap.release();marks=detector(frames);assert sum(m is not None for m in marks)/len(frames)>=.95'
        assert source.count(gate)==1
        source=source.replace(gate,'cap.release();marks=_native_landmarks(detector,frames)')
    if name in ['mtd','vocalist']:
        binding="shared.CACHE=B/'smoke/retrieval_fixtures';shared.RUN=B/'smoke'"
        assert source.count(binding)==1
        source=source.replace(binding,"shared.CACHE=EXPERIMENT_RUN/'retrieval_fixtures';shared.RUN=EXPERIMENT_RUN")
    marker=('video=args.video.resolve()' if name=='avs_u' else 'out=a.output or' if name in ['mtd','vocalist'] else 'video=a.video.resolve()')
    assert source.count(marker)==1
    prefix,tail=source.split(marker,1);body=marker+tail
    sys.argv=[str(original),'--output',str(RUN/name)]
    if name!='avs_u':sys.argv+=['--model',name]
    ns={'__name__':'__experiment_adapter__','__file__':str(original),'EXPERIMENT_RUN':RUN,'_native_landmarks':native_landmarks}
    exec(compile(prefix,str(original),'exec'),ns)
    write(RUN/name/'adapter_binding.json',dict(source=str(original),source_sha256=sha(original),
        wrapper_sha256=sha(__file__),scope='Only input bindings and model lifecycle change; audited forward/controls/reductions unchanged'))
    rows=json.loads((RUN/'scoring_inputs.json').read_text())['unique_rows'];results=[]
    for row in rows:
        out=RUN/name/row['id'];receipt=out/'deployment_validation.json'
        if receipt.exists():
            results.append(dict(id=row['id'],status='PASS',receipt=str(receipt),sha256=sha(receipt)));continue
        if name in ['synchformer','sparsesync'] and min(row['samples']//640,row['actual_video_frames'])<125:
            results.append(dict(id=row['id'],status='UNMEASURABLE',reason='FEWER_THAN_125_FULL_FRAMES'));continue
        out.mkdir(parents=True,exist_ok=True)
        args=ns['args' if name=='avs_u' else 'a']
        args.video=Path(row['video_path']);args.audio=Path(row['audio_path']);args.output=out
        if name in ['mtd','vocalist']:args.item_id='dystreamfull104_'+row['id']
        ns['out']=out
        try:
            exec(compile(body,str(original),'exec'),ns)
            results.append(dict(id=row['id'],status='PASS',receipt=str(receipt),sha256=sha(receipt)))
        except NativeFaceSupportError as error:
            evidence=dict(id=row['id'],status='UNMEASURABLE',reason='NATIVE_FACE_SUPPORT_LT95_PERCENT',detected=error.detected,decoded_frames=error.total,
                          threshold=.95,audio_sha256=row['audio_sha256'],video_sha256=row['video_sha256'],no_model_forward=True)
            write(out/'unmeasurable.json',evidence)
            results.append(dict(id=row['id'],status='UNMEASURABLE',reason=evidence['reason'],receipt=str(out/'unmeasurable.json')))
        except Exception as error:
            traceback.print_exc()
            results.append(dict(id=row['id'],status='FAILED',error=repr(error)))
        write(RUN/name/'batch_scores.json',results)
        print('BATCH_ITEM',name,row['id'],results[-1]['status'],flush=True)
    write(RUN/name/'batch_scores.json',results)
    assert len(results)==len(rows)
    if any(r['status']=='FAILED' for r in results):raise RuntimeError(name+' has failed items; inspect batch_scores')

if __name__=='__main__':main()
