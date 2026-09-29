"""One local-provider runtime probe; never enters scientific evaluation."""
import os
import random
import sys
import time

import numpy as np

from scripts.experiments.tts_independent_visual import ROOT, read, sha, write
from scripts.experiments.tts_native_gain_attribution.common import gpu_compute_pids,gpu_lease

BASE=ROOT/"runs/tts_instance_inventory_20260926"


def main():
    import soundfile as sf
    import torch
    from scripts.tts.faster_qwen3 import FasterQwen3TTSProvider
    row=read(ROOT/"runs/aishell1_qwen_mfa_linear_n100_20260816/00_pairs/cohort.json")["records"][0]
    model=ROOT/"models/Qwen3-TTS-12Hz-0.6B-Base"
    p={"scope":"single runtime probe, not a scientific replicate or native advantage measurement","id":row["sample_id"],"reference":row["audio_path"],"reference_sha256":sha(row["audio_path"]),"text":row["transcript"],"model":str(model),"strict_backend":True,"provider_default_dtype":"bfloat16","max_new_tokens":256,"seed":20260926,"script_sha256":sha(__file__)}
    write(BASE/"smoke_protocol.json",p)
    random.seed(20260926);np.random.seed(20260926);torch.manual_seed(20260926);torch.set_num_threads(2)
    with gpu_lease(gpu_peak_bytes=12<<30,disk_temp_bytes=100<<20,disk_persistent_bytes=100<<20) as gate:
        write(BASE/"smoke_compute_gate.json",gate)
        assert not set(gpu_compute_pids())-{os.getpid()}
        provider=FasterQwen3TTSProvider({"faster_qwen3":{"model_id":str(model),"strict_backend":True,"max_new_tokens":256}})
        start=time.monotonic()
        try:
            provider._ensure_model()
            loaded=time.monotonic()
            result=provider.generate_voice_clone(row["transcript"],__import__('pathlib').Path(row["audio_path"]),row["transcript"],"Chinese")
            elapsed=time.monotonic()-loaded
            audio=np.asarray(result.audio)
            finite=bool(np.isfinite(audio).all())
            dest=BASE/"runtime_smoke.wav"
            sf.write(dest,audio,result.sample_rate,subtype="PCM_16")
            out={"status":"PASS" if finite and len(audio)>100 else "FAIL","load_seconds":loaded-start,"generate_seconds":elapsed,"duration_seconds":len(audio)/result.sample_rate,"sample_rate":result.sample_rate,"finite":finite,"peak":float(np.max(abs(audio))),"gpu_peak_bytes":torch.cuda.max_memory_allocated(),"python":sys.executable,"torch":torch.__version__,"audio_sha256":sha(dest),"no_quality_or_sync_claim":True}
        except Exception as exc:
            out={"status":"FAIL","error_type":type(exc).__name__,"error":str(exc)[:800],"elapsed_seconds":time.monotonic()-start}
        write(BASE/"smoke_result.json",out)
        print(out,flush=True)


if __name__=="__main__":main()
