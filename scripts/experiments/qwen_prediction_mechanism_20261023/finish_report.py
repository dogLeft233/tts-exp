"""Final evidence interpretation, without changing any frozen analysis."""
from pathlib import Path
import argparse,json,statistics,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.report import main as basic_report

def load(p):return json.loads(Path(p).read_text())
def line_stats(r):
    if r['mean'] is None:return '不可测（共同完整支持为0）'
    return f'{r["mean"]:+.3f}，97.5%CI[{r["CI97_5"][0]:+.3f},{r["CI97_5"][1]:+.3f}]'

def main(run):
    c=load(run/'C_analysis.json');audit=load(run/'C_independent_CPU_audit.json');assert audit['status']=='PASS'
    p=load(run/'C_protocol.json');elig=load(run/'C_policy_eligibility.json');plan=load(run/'C_downstream_plan.json')
    basic_report(run);paragraphs=['','## C最终判读与资格核对','',
        '本次只有细码类通过B门。因此baseline=coarse-only、fine-only=both是精确重复政策；0.6B细码温度0.8138190841302276，1.7B细码温度0.9952570905908943，粗码均0.9。四个逻辑政策不能算四次独立干预；粗码与交互的零值来自政策重复，本轮没有独立测试粗码温度作用。','',
        f'240逻辑单元/120唯一单元；新TTS{audit["C_generation_status_counts"].get("COMPLETE",0)}条，旧0.6B基线复用30条，逻辑alias120条。新视频/官方评分计划{sum(j["count"] for j in plan["jobs"])}个；唯一已评分单元（含复用）{audit["score_units"]}个。实际生成失败另列，不按好分删样。','',
        f'主比较完整来源{len(c["complete_primary_source_ids"])}/10。P1：{line_stats(c["primary_and_gap_contrasts"]["P1_L06_both_minus_baseline"])}；P2：{line_stats(c["primary_and_gap_contrasts"]["P2_gap_shrink"])}。','']
    g=c['numeric_support_gates'];joint=all(g.values())
    paragraphs.append(f'预定数值门：P1 {"通过" if g["P1_L06_both_minus_baseline"] else "未通过"}（点值≥+0.100且97.5%下界>0）；P2 {"通过" if g["P2_gap_shrink"] else "未通过"}（点值≥+0.200且97.5%下界>0）。'+('两个数值门均通过；仍须结合下面的内容/可靠性和实际熵操纵。' if joint else '**没有达到预设的联合救援标准。** 这不是对所有解码机制的否定。'))
    native=c['primary_and_gap_contrasts']['native_model_gap'];ci=native.get('CI95_descriptive')
    if ci is not None and ci[0]<=0<=ci[1]:paragraphs.extend(['','原生模型分差的描述性95%区间跨0：**原生模型分差未在这组确认来源上确认**，不将政策效应写成解释普遍模型优势。'])
    paragraphs.extend(['','| 模型/逻辑政策 | 资格失败draw/30 | 失败来源数 | 平均ASR CER（来源等权） |','|---|---:|---:|---:|'])
    content=c['content_diagnostics']['source_policy_records'] if c['content_diagnostics'] else []
    for model in ['L06','L17']:
        for policy in ['baseline','both']:
            e=next(r for r in elig['policies'] if r['model']==model and r['policy']==policy)
            values=[r['CER_mean_all3'] for r in content if r['model']==model and r['policy']==policy and r['CER_mean_all3'] is not None]
            cer=f'{statistics.fmean(values):.6f} (n={len(values)})' if values else '不可测'
            paragraphs.append(f'| {model}/{policy} | {e["bad_draws"]}/30 | {len(e["bad_sources"])} | {cer} |')
    paragraphs.extend(['','0.6B fine/both的S0009 draw3为11.68秒、EOS=True、时长比3.74239，保留全部原件及主评分；未截尾、未换seed。ASR是无提示、无VAD的Whisper-small代理，数字写法及长尾转写会影响CER，不能据此伪称人工语音质量判断。','',
        '| 模型/目标类 | B冻结目标nat | C自由轨迹实际nat | 误差nat | ≤0.02目标 |','|---|---:|---:|---:|---|'])
    adequate=True
    for r in c['actual_entropy_target_checks']:
        ok=r['within_original_0_02nat_target'] and len(r['source_ids'])==10;adequate=adequate and ok
        paragraphs.append(f'| {r["model"]}/{r["category"]} | {r["B_frozen_target_nat"]:.6f} | {r["C_actual_trajectory_mean_nat"]:.6f} | {r["error_nat"]:+.6f} | {"达到" if ok else "未达到"} |')
    if not adequate:paragraphs.extend(['','**自由生成轨迹未全部实现冻结熵目标，属于操纵不足。** B共同历史上的温度校准通过不保证新自由轨迹仍匹配；本轮没有根据C结果重新调温。没有评分改善不能据此判定熵假说为阴性；即使存在评分变化，也不能把它唯一归因于熵。'])
    strict=c['strict_no_anomaly_draw_sensitivity']
    if strict:paragraphs.extend(['','严格无异常、四臂共同draw敏感性（事后资格条件，非主结果替代）：',f'- P1：{line_stats(strict["P1"])}。',f'- P2：{line_stats(strict["P2"])}。','来源内只用四个主臂同时可测且资格通过的共同seed，支持集逐来源列于C_analysis.json。'])
    paragraphs.extend(['','独立CPU复算通过：全部已评分track的原始31-lag距离曲线按官方公式回算C/Dmin/offset；原生标量仅打印3位小数，允许0.00051舍入/FP32差，未改分。两主比较按独立Python来源聚合与同seed20000 bootstrap回算，与主分析误差≤1e-12。生成参数和新raw音频/codes的SHA也已逐单元核对。','',
        '结论限于本10来源、3draw、同句self-ICL、统一RMS、face3/Wav2Lip GAN与官方自配评分链。温度改变整个选择分布，不能识别熵唯一中介；也不是纯参数量、TFG单独贡献或人类嘴型质量证据。没有训练、没有扩源/seed/温度搜索。','',
        '模型/评估器大权重不重复打包：原持久盘路径与逐文件SHA保留；TAR保存全部新科学产物、旧基线所需音频/codes/视频/官方曲线和小配置/源码。归档SHA与完整成员读回见外部receipt，C盘副本是本机备份，不是百度网盘。',''])
    with (run/'report.md').open('a',encoding='utf8') as f:f.write('\n'.join(paragraphs))
    start=load(run.parent/'qwen_prediction_mechanism_20261023_attempt01/A_L06/started.json')['unix']
    state={'status':'CONCLUDED_FROZEN_PROTOCOL_EXECUTED','scientific_outcome':'JOINT_NUMERIC_GATE_PASSED' if joint else 'JOINT_RESCUE_NOT_ESTABLISHED','actual_entropy_manipulation_adequate':adequate,'interpretation_if_manipulation_inadequate':'not a negative causal test of entropy; temperature policy effects remain narrower','primary_sources':c['complete_primary_source_ids'],'numeric_support_gates':g,'A':'PASS','B_teacher_forced_passes':140,'B':'FINE_ADVANCED_COARSE_NOT_ADVANCED','C_new_TTS_successes':audit['C_generation_status_counts'].get('COMPLETE',0),'C_old_baselines_reused':30,'C_logical_aliases':120,'C_score_units_with_reuse':audit['score_units'],'C_new_scores':sum(j['count'] for j in plan['jobs']),'CPU_A_B_C_audits':'PASS','required_direct_python_tests':10,'pytest_installed':False,'training':False,'expanded_search':False,'elapsed_wall_s':time.time()-start,'conservative_serial_GPU_time_upper_bound_s':time.time()-start,'archive_receipt':'external sidecar; full SHA/member readback follows','unix':time.time()}
    assert state['elapsed_wall_s']<43200 and state['conservative_serial_GPU_time_upper_bound_s']<28800
    with (run/'final_status.json').open('x') as f:json.dump(state,f,indent=2,allow_nan=False)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
