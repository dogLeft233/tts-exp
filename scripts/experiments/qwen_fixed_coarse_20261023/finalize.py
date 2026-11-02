"""Freeze a completed, held technical probe, including failed measurement setup."""
import datetime
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_fixed_coarse_20261023.probe import RUN,read,save,sha


def main():
    assert read(RUN/'independent_readback.json')['status']=='PASS'
    assert read(RUN/'checks/mfa_lexical_repair02/summary.json')['status']=='HOLD'
    snapshot=RUN/'final_sources';snapshot.mkdir(exist_ok=False)
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_qwen_fixed_coarse_20261023.py',
       ROOT/'scripts/experiments/qwen_full_tts_extension_20261022/tts_stage.py',
       ROOT/'scripts/experiments/repeated_take_crossover_20261015/timing_diagnostics.py',
       ROOT/'scripts/experiments/qwen_tts_decline_audit_20261022.py']
    for f in files: (snapshot/f.name).write_bytes(f.read_bytes())
    command=['/home/wjj/.venvs/qwen3-tts-1p7b/bin/python',str(ROOT/'tests/test_qwen_fixed_coarse_20261023.py')]
    test=subprocess.run(command,capture_output=True,text=True)
    save(RUN/'tests_receipt.json',dict(command=command,returncode=test.returncode,stdout=test.stdout,stderr=test.stderr))
    assert test.returncode==0
    save(RUN/'execution_notes.json',dict(
        completed=dict(model_stages=4,synthesis_calls=6,standalone_decoder_replays=2,asr_audios=4,
                       mfa_audios_invalid_first=4,mfa_audios_repaired=4,training_updates=0,tfg_videos=0,syncnet_scores=0),
        exceptions=[
          'First audio_checks prepare used ASR env lacking soundfile; import failure before any files or forward. Preserved source v1. Repaired by lazy imports, stdlib wave ASR reader and Qwen env preprocessing.',
          'First MFA process exit0 but unsegmented full sentence was one OOV/spn, no recognized phones. Original checks/textgrids and summary retained. Repaired in new mfa_lexical_repair02 with dictionary-verified character tokens for all cells, no audio change.',
          'Initial readback reducer could not handle empty phone events (IndexError); no result.json was written. Fixed empty-support handling, then independently recomputed repaired summary.',
        ],
        scope='single source and seed, no expansion; primary content and timing gates retained despite failure',
        measurement_status='HOLD; ASR baseline not error-free; MFA constrained by supplied text, no human listening/annotation'))
    report='''# 固定粗编码、双模型补全：最小技术验证

**结论：实现与精确自重放通过；内容／局部时序保持尚未通过，停止在音频检查，不进入完整评分。**

选旧云端同10人中按ID排序第一条 BAC009S0002W0122、draw1、seed2026102401。句子为“而对楼市成交抑制作用最大的限购”。没有按分数或输出质量挑样。V100、FP32/eager、既有模型/参考/解码参数；两模型共用同SHA codec。旧封存run未修改。

## 实现与重放

在各自官方Talker.generate中添加粗码LogitsProcessor，逐步固定旧粗编码和末尾EOS；其余15码本仍由该模型自己的隐藏状态、参考条件和补全器生成。保留原维度的随机采样调用，不跳过粗码采样消耗；不交换1024/2048维模块。完整reference+new上下文解码后使用原官方裁切。

0.6B、1.7B均通过：①旧完整编码重解码逐样点相同；②原seed不干预生成全部编码与波形相同；③固定自身粗码后全部编码与波形相同。两个控制进程退出0后才执行交叉，两次交叉也退出0。独立读取NPY/WAV重新核对通过；4项测试通过。

| 固定的粗编码来源 | 补全模型 | 帧数／时长 | 相对同粗码原模型的细码改变比例 |
|---|---|---|---|
| 0.6B | 0.6B | 58／4.640秒 | 0（精确重放） |
| 0.6B | 1.7B | 58／4.640秒 | 92.759% |
| 1.7B | 1.7B | 47／3.760秒 | 0（精确重放） |
| 1.7B | 0.6B | 47／3.760秒 | 92.199% |

粗编码与EOS逐项相同、总采样数相同；比例是离散码ID改变率，不是音质变化率。

## 内容与时间检查

四格同一resample_poly转16k、同整体RMS 0.0376838172675746、PCM16，无削波。Whisper-small CPU/int8、无提示文本、关闭VAD与上下文续写。字符编辑距离：0.6原组2→1.7补全7；1.7原组3→0.6补全4。原组也将“抑制”识别成“一直”，所以不能把这些错误直接当人工听辨结果；但没有证据确认交叉后内容保持。

首轮MFA把无空格整句视为OOV，仅spn，测量无效，已保留。修复为词典覆盖核验通过的逐字分词，四格各得到35个对应音素、无unknown；相同标签源于强制给定文字，不能独立证明内容正确。

| 固定粗码后换补全模型 | 音素中心绝对差中位数 | 95分位 | 最大值 | 包络最佳lag |
|---|---:|---:|---:|---:|
| 0.6B粗码：0.6B→1.7B | 25ms | 120ms | 130ms | +10ms |
| 1.7B粗码：1.7B→0.6B | 20ms | 201ms | 245ms | −20ms |

预设技术门：相同总长／音素标签、无unknown，中心差中位数≤20ms、95分位≤40ms、最大≤80ms，包络lag≤10ms。两方向均未通过。边界为MFA估计，受音色、音质及错误转写影响；不能把245ms当已人工确认的真实时间移动。没有声称统计等效或总体结论。

## 研究判断与停止条件

这条接口可用于研究条件补全，但当前不能作为“固定内容与音素时间、只换声学细节”的干净干预。细码和主干隐藏状态是联合生成的；共享codec词表并不保证跨模型条件匹配。原设想中的纯时间／纯音色拆分仍不成立。未跑TFG、SyncNet、训练或云端调用。

后续若继续，应先对本四格中疑似异常的词段做盲听与局部边界复核，区分实际可懂度／时序变化和测量器失效；不要立即扩10人、调阈值或解释评分增益。单句技术失败也不能判定所有固定粗码补全都不可用。

## 证据

- protocol.json：先定样本、模型/源SHA、控制和门限。
- L06_control/、L17_control/：精确重放；L06_cross/、L17_cross/：交叉音频与全部码本。
- checks/asr.json：四格完整转写及ASR模型SHA。
- checks/mfa_lexical_repair02/summary.json：当前有效边界诊断；旧checks/summary.json为无效首轮，不能替代当前结果。
- independent_readback.json、tests_receipt.json、execution_notes.json：读回、测试与失败修复记录。
- final_sources/、artifact_manifest.json：最终源码及完整产物SHA；历史阶段源码同时保留。
'''
    with (RUN/'report.md').open('x',encoding='utf8') as f: f.write(report)
    save(RUN/'completion.json',dict(status='COMPLETED_TECHNICAL_PROBE_HOLD',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_training=True,no_tfg=True,no_syncnet=True))
    entries=[dict(path=str(f.relative_to(RUN)),size=f.stat().st_size,sha256=sha(f))
             for f in sorted(RUN.rglob('*')) if f.is_file()]
    save(RUN/'artifact_manifest.json',dict(files=entries,count=len(entries),bytes=sum(r['size'] for r in entries),
        includes='both failed and repaired alignment evidence; followed input audio symlinks'))
    print('FINAL',len(entries),sum(r['size'] for r in entries),sha(RUN/'artifact_manifest.json'))


if __name__=='__main__':main()
