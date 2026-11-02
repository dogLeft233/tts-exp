"""EARS40 official MTDVocaLiST and original VocaLiST retrieval adapters.

Author Dataset/calc_pdist/eval_model bodies are imported from the audited runners.
Only the input fixture, result capture, and explicitly labelled controls are new.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/home/wjj/tts-audio/tts-exp')
RUN_NAME = 'ears40_w2l_nine_scorers_20261029_attempt01'
RUN = ROOT / 'runs' / RUN_NAME
CACHE = Path('/dev/shm') / RUN_NAME / 'retrieval_preprocessed'
DEPLOY = Path('/home/wjj/av-sync-eval-20261028')
KS = [5, 7, 9, 11, 13, 15]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def shift_pcm(pcm, samples):
    """Physical, non-circular full-waveform perturbation, exact length retained."""
    import numpy as np
    out = np.zeros_like(pcm)
    if samples > 0:
        out[samples:] = pcm[:-samples]
    elif samples < 0:
        out[:samples] = pcm[-samples:]
    else:
        out[:] = pcm
    return out


def independent_reduction(matrix):
    import numpy as np
    z = np.asarray(matrix, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 31 or not len(z):
        raise ValueError('Expected nonempty query by 31 official candidate matrix')
    result = {}
    for k in KS:
        radius = (k - 5) // 2
        smoothed = np.stack([z[max(0, i-radius):min(len(z), i+radius+1)].mean(0)
                             for i in range(len(z))])
        result['K' + str(k)] = float(np.mean(np.abs(15-smoothed.argmax(1)) <= 1))
    return result


def matrix_record(matrix):
    """Native zero-target scores and separate relative-offset diagnostic."""
    import numpy as np
    z = np.asarray(matrix)
    interior = z[20:-20] if len(z) > 40 else z
    return dict(metrics=independent_reduction(z), windows=len(z),
                global_offset_frames=int(15-z.mean(0).argmax()),
                interior_offset_frames=int(15-interior.mean(0).argmax()))


def control_record(native, shifted, samples):
    expected = -samples / 640
    observed = shifted['interior_offset_frames']-native['interior_offset_frames']
    return dict(shift_samples=samples, expected_change_frames=expected,
                observed_change_frames=observed,
                relative_offset_error_frames=abs(observed-expected),
                relative_offset_recovered=abs(observed-expected) <= 1,
                k5_decreased=shifted['metrics']['K5'] < native['metrics']['K5'],
                native_k5=native['metrics']['K5'], shifted_k5=shifted['metrics']['K5'])


def prepare_crop(row, detector):
    """Shared author JPEG fixture; boxes/manifests survive volatile SHM cache."""
    import cv2
    import numpy as np
    folder = CACHE / row['id']
    manifest_path = RUN / 'retrieval_preprocessing' / (row['id'] + '.json')
    if (folder/'complete.json').exists():
        receipt = read(folder/'complete.json')
        assert receipt['video_sha256'] == sha(row['video_path'])
        assert receipt['audio_sha256'] == sha(row['audio_path'])
        assert receipt['frames'] == row['native_frames']
        # A previous interrupted control must never become the native fixture.
        shutil.copyfile(row['audio_path'], folder/'audio.wav')
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(row['video_path'])
    boxes = []
    jpgs = []
    try:
        for i in range(row['native_frames']):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Short native frame decode: ' + row['id'])
            box = detector.get_detections_for_batch(np.asarray([frame]))[0]
            if box is None:
                raise RuntimeError('Face missing: ' + row['id'] + ':' + str(i))
            x1, y1, x2, y2 = map(int, box)
            path = folder / (str(i) + '.jpg')
            assert cv2.imwrite(str(path), frame[y1:y2, x1:x2])
            boxes.append([x1, y1, x2, y2])
            jpgs.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)))
    finally:
        cap.release()
    # Refuse extra old frames if interrupted cache geometry ever changes.
    assert len(list(folder.glob('*.jpg'))) == row['native_frames']
    shutil.copyfile(row['audio_path'], folder/'audio.wav')
    box_path = RUN/'retrieval_preprocessing'/(row['id']+'_boxes.npy')
    box_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(box_path, np.asarray(boxes, dtype=np.int32))
    receipt = dict(id=row['id'], frames=len(jpgs), video_path=row['video_path'],
                   video_sha256=sha(row['video_path']), audio_path=row['audio_path'],
                   audio_sha256=sha(row['audio_path']), boxes_path=str(box_path),
                   boxes_sha256=sha(box_path), jpgs=jpgs, shared_by=['mtd', 'vocalist'],
                   recipe='original resolution per-frame S3FD; default OpenCV JPEG; author Dataset transforms',
                   cache_volatile=True, rebuild_from_canonical_video=True)
    write(manifest_path, receipt)
    write(folder/'complete.json', receipt)
    return folder


def run(model_name):
    import numpy as np
    import soundfile as sf
    import torch
    import mtd_official_audit_20261028 as audit
    if model_name == 'mtd':
        import mtd_official_cloud_rescore_20261028 as official
    else:
        import vocalist_official_cloud_rescore_20261028 as official
    torch.set_num_threads(4)
    torch.manual_seed(20261029)
    torch.backends.cudnn.benchmark = False
    env = official.official_env()
    if model_name == 'mtd':
        from models.student_thin_200 import SyncTransformer
        model = SyncTransformer(d_model=200)
        weight = DEPLOY/'weights/mtdvocalist/pure_MTDVocaLiST.pth'
        source = DEPLOY/'MTDVocaLiST/test_stu.py'
    else:
        from models.model import SyncTransformer
        model = SyncTransformer()
        weight = DEPLOY/'weights/vocalist/vocalist_5f_lrs2.pth'
        source = DEPLOY/'vocalist/test_lrs2.py'
    checkpoint = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model.eval().cuda()
    out = RUN/model_name
    out.mkdir(parents=True, exist_ok=True)
    write(out/'model_load.json', dict(strict=True, parameters=sum(p.numel() for p in model.parameters()),
           weight=str(weight), official_test=str(source), official_test_sha256=sha(source),
           adapter_sha256=sha(__file__), audited_runner_sha256=sha(official.__file__),
           torch=torch.__version__, device=torch.cuda.get_device_name(0),
           primary='official K5 zero target retrieval; no lag calibration',
           secondary=['K7', 'K9', 'K11', 'K13', 'K15'],
           controls='full PCM +/-6400 samples with fixed image frames, zero padding, no circular roll'))
    rows = read(RUN/'inputs.json')
    assert len(rows) == 80 and len({r['id'] for r in rows}) == 80
    assert len({r['sample_id'] for r in rows}) == 40
    assert sum(r['arm'] == 'N' for r in rows) == 40
    assert sum(r['arm'] == 'T' for r in rows) == 40
    scores = read(out/'scores.json') if (out/'scores.json').exists() else {}
    controls = read(out/'controls.json') if (out/'controls.json').exists() else {}
    detector = None
    for row in rows:
        key = row['id']
        if key in scores and (row['arm'] != 'N' or key in controls):
            continue
        start = time.monotonic()
        folder = CACHE/key
        if not (folder/'complete.json').exists() and detector is None:
            sys.path.append(str(ROOT/'third_party/Wav2Lip'))
            import face_detection
            detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                    flip_input=False, device='cuda')
        folder = prepare_crop(row, detector)
        pcm, sr = sf.read(row['audio_path'], dtype='int16')
        assert sr == 16000 and pcm.ndim == 1
        exported, sr2 = sf.read(folder/'audio.wav', dtype='int16')
        assert sr2 == sr and np.array_equal(pcm, exported)
        vid, mel, last = audit.dataset_get(env, folder)
        assert last == row['native_frames']-5
        if model_name == 'vocalist':
            assert 0 <= float(vid.min()) <= float(vid.max()) <= 1
        matrix_dir = out/'matrices'
        matrix_dir.mkdir(exist_ok=True)
        native_path = matrix_dir/(key+'_0.npy')
        if key not in scores:
            result = official.capture_official_reduction(env, model, vid, mel, last, native_path)
            native = matrix_record(np.load(native_path))
            assert native['metrics'] == result, 'Author vs independent metrics mismatch'
            scores[key] = dict(id=key, source=row['sample_id'], arm=row['arm'], **native,
                               pcm_exact=True, author_reduction_exact=True,
                               matrix_path=str(native_path), matrix_sha256=sha(native_path),
                               audio_sha256=sha(row['audio_path']), video_sha256=sha(row['video_path']))
            write(out/'scores.json', scores)
        native = scores[key]
        if row['arm'] == 'N' and key not in controls:
            shifted_results = {}
            try:
                for samples in [-6400, 6400]:
                    shifted = shift_pcm(pcm, samples)
                    sf.write(folder/'audio.wav', shifted, sr, subtype='PCM_16')
                    shifted_back, check_sr = sf.read(folder/'audio.wav', dtype='int16')
                    assert check_sr == sr and np.array_equal(shifted, shifted_back)
                    _, shifted_mel, shifted_last = audit.dataset_get(env, folder)
                    assert shifted_last == last
                    dest = matrix_dir/(key+'_'+str(samples)+'.npy')
                    result = official.capture_official_reduction(env, model, vid, shifted_mel, last, dest)
                    record = matrix_record(np.load(dest))
                    assert record['metrics'] == result
                    shifted_results[str(samples)] = dict(**record, matrix_path=str(dest), matrix_sha256=sha(dest),
                        check=control_record(native, record, samples),
                        pcm_sha256=hashlib.sha256(shifted.astype('<i2').tobytes()).hexdigest())
            finally:
                shutil.copyfile(row['audio_path'], folder/'audio.wav')
            controls[key] = dict(id=key, source=row['sample_id'], shifts=shifted_results)
            write(out/'controls.json', controls)
        write(out/'status.json', dict(status='RUNNING', native_scored=len(scores), native_total=len(rows),
              natural_control_pairs=len(controls), control_total=sum(r['arm']=='N' for r in rows), last=key))
        print(json.dumps(dict(model=model_name, native_scored=len(scores), control_pairs=len(controls),
              id=key, seconds=round(time.monotonic()-start, 2))), flush=True)
    checks = [s['check'] for r in controls.values() for s in r['shifts'].values()]
    assert len(scores) == len(rows) and len(checks) == 2*sum(r['arm']=='N' for r in rows)
    write(out/'control_summary.json', dict(total=len(checks),
          k5_decreased=sum(c['k5_decreased'] for c in checks),
          k5_control_pass=sum(c['k5_decreased'] for c in checks) >= .8*len(checks),
          relative_offset_recovered=sum(c['relative_offset_recovered'] for c in checks),
          relative_offset_control_pass=sum(c['relative_offset_recovered'] for c in checks) >= .8*len(checks),
          relative_offset_max_error_frames=max(c['relative_offset_error_frames'] for c in checks),
          no_lag_correction=True, interpretation='K5 decrease and known offset-change recovery are separate diagnostics'))
    write(out/'status.json', dict(status='COMPLETE', native_scored=len(scores), natural_control_pairs=len(controls)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['mtd', 'vocalist'])
    args = parser.parse_args()
    run(args.model)
