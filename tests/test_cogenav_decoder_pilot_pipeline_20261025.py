import importlib.util
from pathlib import Path
import sys
import torch

folder=Path(__file__).parents[1]/'scripts/experiments/cogenav_decoder_pilot_20261025'
sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location('decoder_pipeline',folder/'pipeline.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_monitor_selection_is_fixed_balanced_and_order_invariant():
    rows=[dict(id=f'S{s}_{i}',speaker=f'S{s}',member=f'validation/S{s}/{i}.wav',split='validation')
          for s in range(4) for i in range(10)]
    rows.append(dict(id='not_val',speaker='S9',member='train/f.wav',split='train'))
    ids=module.monitor_ids(rows)
    assert len(ids)==16 and 'not_val' not in ids
    assert ids==module.monitor_ids(list(reversed(rows)))
    assert all(sum(x.startswith(f'S{s}_') for x in ids)==4 for s in range(4))


def test_inference_trims_or_right_pads_without_stretching():
    condition=torch.zeros(1,3,1024)
    def generator(_):return torch.arange(8.).view(1,1,8)
    assert torch.equal(module.output_wave(generator,condition,4),torch.arange(4.).view(1,4))
    assert torch.equal(module.output_wave(generator,condition,10)[0,:8],torch.arange(8.))
    assert torch.equal(module.output_wave(generator,condition,10)[0,8:],torch.zeros(2))


def test_actual_adapter_vocoder_context_support_and_fp32_roundoff():
    run=module.REPO/'runs/cogenav_decoder_pilot_20261025_attempt01'
    module.configure();module.resource_gate(run,'actual_context_contract',initial=True)
    adapter=module.norm_adapter(run).eval()
    generator=module.load_generator(run,'cuda').eval()
    torch.manual_seed(20261025)
    features=torch.randn(1,100,768,device='cuda')
    results={}
    with torch.inference_mode():
        for precision in ('float32','float64'):
            if precision=='float64':
                adapter.double();generator.double();features=features.double()
            whole=module.reconstruct(adapter,generator,features,100*640)
            positions=[]
            for start in (0,1,16,40,89):
                index=torch.arange(start-module.CONTEXT25,start+11+module.CONTEXT25,device='cuda').clamp(0,99)
                crop=features[:,index]
                part=generator(adapter(crop)).squeeze(1)[:,module.CONTEXT25*640:(module.CONTEXT25+11)*640]
                target=whole[:,start*640:(start+11)*640]
                difference=part-target
                positions.append(dict(start25=start,max_abs=float(difference.abs().max()),
                    relative_l2=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(target))))
            results[precision]=positions
    assert all(r['max_abs']<=1e-10 for r in results['float64'])
    assert all(r['relative_l2']<=1e-5 for r in results['float32'])
    module.dump(run/'actual_context_contract_revision03.json',dict(status='PASS',context25=module.CONTEXT25,results=results))
