"""Apply the exact original loading/capacity probes to an isolated dataset namespace."""
import argparse,json
from pathlib import Path
def run(root,stage):
    assert root.name=='DebbieDingell_recording_roles_v2' and stage in ['loading','capacity']
    assert json.loads((root/'prepared/DebbieDingell/geometry_v2/geometry_qualification.json').read_text())['passed']
    import runtime_common
    # Only filesystem namespace changes; parser, network, seeds, rays and probe math are untouched.
    runtime_common.ROOT=root;runtime_common.OFFICIAL=root/'official'
    if stage=='loading':
        import qualify_loading as probe
    else:
        import qualify_capacity as probe
    assert probe.ROOT==root
    probe.run('DebbieDingell')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['loading','capacity'],required=True);a=p.parse_args();run(a.root,a.stage)
