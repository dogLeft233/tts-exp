import importlib.util
import json
from pathlib import Path
import tempfile

spec=importlib.util.spec_from_file_location('decoder_plot',Path(__file__).parents[1]/'scripts/experiments/cogenav_decoder_pilot_20261025/plot_results.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_global_axis_keeps_stage_change_explicit_and_uses_raw_teacher_metric():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        trace=[dict(stage='warmup',global_step=2000,teacher_mse=2,loss=4),
               dict(stage='joint',global_step=2500,teacher_mse=1,loss=20)]
        (root/'train_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
        (root/'untrained_monitor.json').write_text(json.dumps(dict(logmel=3)))
        (root/'validation_warmup_02000.json').write_text(json.dumps(dict(logmel=1)))
        (root/'validation_joint_00500.json').write_text(json.dumps(dict(logmel=.5)))
        actual,validation=module.read_curves(root)
        assert actual==trace
        assert [r['global_step'] for r in validation]==[0,2000,2500]
        assert [r['logmel'] for r in validation]==[3,1,.5]
