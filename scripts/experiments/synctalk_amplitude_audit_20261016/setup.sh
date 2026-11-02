set -eu
ROOT=/home/wjj/tts-audio/synctalk-amplitude-audit-20261016
BASE=/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01
"$BASE/bin/python" -m venv --without-pip --system-site-packages "$ROOT/env"
printf '%s\n' "$BASE/lib/python3.10/site-packages" > "$ROOT/env/lib/python3.10/site-packages/audit_shared_readonly.pth"
"$ROOT/env/bin/python" - <<'PY'
import torch,librosa,soundfile,transformers,json
from pathlib import Path
print(torch.__version__,librosa.__version__,soundfile.__version__,transformers.__version__,torch.cuda.is_available())
p=Path('/home/wjj/tts-audio/tts-exp/data/aishell1_100_zh/manifest.json')
x=json.loads(p.read_text());print(str(x)[:2300])
PY
