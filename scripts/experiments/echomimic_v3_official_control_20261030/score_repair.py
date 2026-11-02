"""One-shot scoring repair: exact presentation-frame prefix, lossless H264."""
from pathlib import Path
import json
import shutil
import subprocess
import control
from control import RUN,B,sha,read,write

ORIGINAL_RUN=subprocess.run


def repaired_command(command):
    cmd=list(map(str,command))
    if cmd[0]==str(B/'tools/bin/ffmpeg') and Path(cmd[-1]).name=='visual.mp4':
        assert '-an' in cmd and '-frames:v' in cmd and cmd[cmd.index('-c:v')+1]=='copy'
        index=cmd.index('-c:v')
        cmd[index:index+2]=['-r','25','-c:v','libx264','-crf','0','-preset','veryfast','-bf','0','-pix_fmt','yuv420p']
    elif cmd[0]==str(B/'tools/bin/ffmpeg') and Path(cmd[-1]).name=='input.mkv':
        cmd[-1:-1]=['-avoid_negative_ts','disabled']
    return cmd


def run(command,*args,**kwargs):
    cmd=repaired_command(command)
    result=ORIGINAL_RUN(cmd,*args,**kwargs)
    if cmd[0]==str(B/'tools/bin/ffmpeg') and Path(cmd[-1]).name=='visual.mp4':
        target=Path(cmd[-1]);source=cmd[cmd.index('-i')+1];frames=cmd[cmd.index('-frames:v')+1]
        def h(p):
            return subprocess.check_output([str(B/'tools/bin/ffmpeg'),'-nostdin','-v','error','-i',str(p),'-map','0:v:0','-frames:v',frames,'-f','hash','-hash','sha256','-']).decode().strip()
        original=h(source);clipped=h(target);assert original==clipped,'Lossless prefix pixel mismatch'
        write(RUN/'prefix_pixel_audits'/(target.parent.name+'.json'),dict(status='PASS',frames=int(frames),
            parent_video=source,parent_sha256=sha(source),prefix_sha256=sha(target),
            parent_decoded_prefix_sha256=original,clipped_decoded_prefix_sha256=clipped,command=cmd))
    return result


def main():
    assert read(RUN/'generation_complete.json')['status']=='PASS'
    assert not (RUN/'execution_repair_attempt02.json').exists()
    assert sha(Path(control.__file__))==read(RUN/'plan.json')['source_sha256']
    bad=RUN/'syncnet';preserved=RUN/'syncnet_preflight_failed_attempt01'
    assert bad.resolve().parent==RUN.resolve() and not preserved.exists()
    assert not list(bad.glob('*/score.json')), 'This repair is for zero scored clips only'
    pins={str(p.relative_to(RUN)):sha(p) for p in bad.rglob('*') if p.is_file()}
    shutil.move(str(bad),str(preserved))
    source=RUN/'code_repair/score_repair.py';source.parent.mkdir();shutil.copyfile(__file__,source)
    write(RUN/'execution_repair_attempt02.json',dict(status='AUTHORIZED_WITHIN_USER_REQUEST',
        reason='Packet-copy truncation of H264 B frames selected PTS3.24 instead of3.20 at frame80; precheck stopped before any SyncNet score. Preserve failed tree. Decode first F frames, lossless x264 CRF0/bf0/r25, prove decoded prefix pixels exact.',
        plan_sha256=sha(RUN/'plan.json'),original_control_sha256=sha(Path(control.__file__)),
        repair_source_sha256=sha(__file__),preserved_failed_files=pins,
        successful_probe=read(RUN/'media_probe_attempt03/result.json'),
        unchanged='All50 model videos, drivers, source selection, 5arm settings, common support, scoring weights and FP32 formula unchanged.'))
    control.subprocess.run=run
    control.score()
    control.analyze()


if __name__=='__main__':main()
