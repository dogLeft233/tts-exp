"""CUDA regressions run against both preserved PyTorch environments."""
import copy,hashlib,json,sys,unittest
from pathlib import Path
import torch
SOURCE=Path(__file__).resolve().parents[1]/'scripts/experiments/synctalk_multispeaker_full_20261017'
if SOURCE.exists():sys.path.insert(0,str(SOURCE))
from fp16_preupdate_guard_v1 import PreUpdateGuard,NumericalGuardError

def digest(obj):
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();h.update(str(y.dtype).encode()+str(y.shape).encode()+y.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:add(v)
        else:h.update(repr(x).encode())
    add(obj);return h.hexdigest()

@unittest.skipUnless(torch.cuda.is_available(),'requires CUDA AMP')
class GuardTests(unittest.TestCase):
    def setup_case(self,scale=65536.):
        torch.manual_seed(42)
        m=torch.nn.Linear(4,2).cuda()
        o=torch.optim.AdamW(m.parameters(),lr=.001,betas=(0,.99),eps=1e-8)
        s=torch.cuda.amp.GradScaler(init_scale=scale)
        g=PreUpdateGuard(m,o,s)
        with torch.cuda.amp.autocast():loss=(m(torch.ones(2,4,device='cuda'))*.01).square().mean()
        g.check_loss(loss);s.scale(loss).backward()
        return m,o,s,g

    def reject_unchanged(self,m,o,s,g,reason):
        before=digest([m.state_dict(),o.state_dict(),s.state_dict()])
        with self.assertRaises(NumericalGuardError) as caught:s.step(o)
        self.assertEqual(caught.exception.event['reason'],reason)
        self.assertEqual(g.optimizer_calls,0)
        self.assertEqual(before,digest([m.state_dict(),o.state_dict(),s.state_dict()]))
        g.close()

    def test_inverse_overflow_rejected_before_update(self):
        self.reject_unchanged(*self.setup_case(2.**-128),'nonfinite_inverse_scale_before_unscale')

    def test_finite_inverse_can_create_nonfinite_gradient(self):
        m,o,s,g=self.setup_case(.5)
        for p in m.parameters():p.grad.fill_(torch.finfo(torch.float32).max)
        self.reject_unchanged(m,o,s,g,'nonfinite_created_by_unscale')

    def test_nonfinite_parameter_rejected(self):
        m,o,s,g=self.setup_case()
        with torch.no_grad():next(m.parameters()).flatten()[0]=float('nan')
        self.reject_unchanged(m,o,s,g,'nonfinite_state_before_optimizer')

    def test_nonfinite_buffer_rejected(self):
        m,o,s,g=self.setup_case();m.register_buffer('density_grid',torch.tensor([float('inf')],device='cuda'))
        self.reject_unchanged(m,o,s,g,'nonfinite_state_before_optimizer')

    def test_nonfinite_optimizer_state_rejected(self):
        m,o,s,g=self.setup_case();o.state[next(m.parameters())]['exp_avg']=torch.tensor(float('nan'),device='cuda')
        self.reject_unchanged(m,o,s,g,'nonfinite_state_before_optimizer')

    def test_nonfinite_loss_rejected_before_backward(self):
        m,o,s,g=self.setup_case();before=digest([m.state_dict(),o.state_dict(),s.state_dict()])
        with self.assertRaises(NumericalGuardError) as caught:g.check_loss(torch.tensor(float('nan'),device='cuda'))
        self.assertEqual(caught.exception.event['reason'],'nonfinite_loss_before_backward')
        self.assertEqual(before,digest([m.state_dict(),o.state_dict(),s.state_dict()]));g.close()

    def test_unchecked_loss_rejected(self):
        m,o,s,g=self.setup_case();g.loss_checked=False
        self.reject_unchanged(m,o,s,g,'loss_not_checked_before_backward')

    def test_ordinary_amp_skip_policy_unchanged(self):
        m,o,s,g=self.setup_case();before=digest([m.state_dict(),o.state_dict()]);old=s.get_scale()
        for p in m.parameters():p.grad.fill_(float('inf'))
        s.step(o);s.update()
        self.assertEqual(g.optimizer_calls,0);self.assertEqual(s.get_scale(),old*.5)
        self.assertEqual(before,digest([m.state_dict(),o.state_dict()]));g.close()

    def test_optimizer_entry_cannot_bypass_gradient_guard(self):
        m,o,s,g=self.setup_case()
        for p in m.parameters():p.grad.fill_(float('nan'))
        before=digest([m.state_dict(),o.state_dict()])
        with self.assertRaises(NumericalGuardError):o.step()
        self.assertEqual(g.optimizer_calls,0);self.assertEqual(before,digest([m.state_dict(),o.state_dict()]));g.close()

    def test_finite_path_exactly_matches_original(self):
        torch.manual_seed(8);reference=torch.nn.Linear(4,2).cuda();guarded=copy.deepcopy(reference)
        states=[]
        for m,use_guard in [(reference,False),(guarded,True)]:
            o=torch.optim.AdamW(m.parameters(),lr=.001,betas=(0,.99),eps=1e-8)
            schedule=torch.optim.lr_scheduler.LambdaLR(o,lambda n:.5**(n/100000))
            s=torch.cuda.amp.GradScaler(init_scale=8.);g=PreUpdateGuard(m,o,s) if use_guard else None
            ema=[p.detach().clone() for p in m.parameters()]
            for step in range(4):
                o.zero_grad()
                with torch.cuda.amp.autocast():loss=(m(torch.ones(2,4,device='cuda'))*.01).square().mean()
                if g:g.check_loss(loss)
                s.scale(loss).backward();s.step(o);s.update();schedule.step()
                if step%2==1:
                    for v,p in zip(ema,m.parameters()):v.mul_(.95).add_(p.detach(),alpha=.05)
            states.append(digest([m.state_dict(),o.state_dict(),s.state_dict(),schedule.state_dict(),ema]))
            if g:self.assertEqual(g.optimizer_calls,4);g.close()
        self.assertEqual(*states)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests))
    report=dict(torch=torch.__version__,cuda=torch.version.cuda,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),passed=result.wasSuccessful() and not result.skipped)
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(report,indent=2))
    print(json.dumps(report));sys.exit(0 if report['passed'] else 1)
