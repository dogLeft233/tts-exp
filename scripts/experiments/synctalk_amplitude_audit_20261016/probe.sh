set -eu
ROOT=/home/wjj/tts-audio/synctalk-amplitude-audit-20261016
find /home/wjj/tts-audio/nas-cch-code -maxdepth 5 -type f \( -name '*.pth' -o -name '*.pt' -o -name '*.pb' \) | head -40
for p in /home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/bin/python /home/wjj/miniconda3/envs/autoavsr/bin/python; do
"$p" - <<'PY'
import sys,importlib.util
print(sys.executable)
for m in ['torch','librosa','soundfile','transformers','scipy','numpy','cv2','python_speech_features']:
 s=importlib.util.find_spec(m); print(m,s.origin if s else None)
PY
done
find /home/wjj/.cache/huggingface -maxdepth 3 -type d -iname '*hubert*' 2>/dev/null
git -C "$ROOT/SyncTalk" rev-parse HEAD || true
curl -IL --connect-timeout 15 --max-time 25 https://github.com/ZiqiaoPeng/SyncTalk 2>&1 | head -12
