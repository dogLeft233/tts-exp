"""Assemble the completed audit; never infer pending control results."""
import argparse
import json
from pathlib import Path


def main(out):
    read = lambda name: json.loads((out / name).read_text())
    media = read('media_and_cached_features_audit.json')
    source = read('source_and_weights_audit.json')
    sync = read('syncreward_official_parity.json')
    bf16 = read('syncreward_bf16_controls.json')
    fp32_controls = read('cached_control_reproduction.json')['syncreward']['rows']
    peavs = read('peavs_official_parity.json')
    native = read('peavs_native_default64_reanalysis.json')
    aac = read('peavs_post_aac_timing_audit.json')
    assert media['controls'] == bf16['controls'] == 40
    lower = bf16['lower']
    flat = sum(r['delta'] == 0 for r in bf16['rows'])
    up = 40 - lower - flat
    old_flat = sum(r['delta'] == 0 for r in fp32_controls)
    direction_changes = sum(((r['delta'] > 0) - (r['delta'] < 0)) !=
                            ((r['fp32_delta'] > 0) - (r['fp32_delta'] < 0)) for r in bf16['rows'])
    n = native['subsets']['all20']
    minimum = native['minimal']
    lines = ['# PEAVS与SyncReward官方调用及时移控制审计', '',
        '结论：旧调用不是完全相同的官方默认配置。PEAVS的batch1与默认64存在实质差异，并定位到上游长度掩码错误；SyncReward旧版为FP32，本次官方CUDA BF16已实测可运行。对齐配置后，两模型仍未通过本项目预设的中文唇同步时移控制。', '',
        '这里的32/40是本项目预设资格门，不是论文官方benchmark分数，也不表示模型在所有任务上无效。', '',
        '## 一、媒体、代码和权重', '',
        '- 全40控制逐PCM验证正负6400采样点移位及零填充；视频逐帧像素不变，流起始时间、采样率、帧率一致。',
        '- PEAVS全40控制的视频I3D特征逐值相同，音频VGGish特征确实改变。官方AAC128k中转后，两代表控制的互相关偏移仍准确为-6400/+6400。',
        f"- PEAVS共{len(source['peavs_checked_source_files'])}个发布源码/配置Git blob匹配；仓库6个模型文件Git blob匹配，VGGish/PCA官方发布SHA匹配。SyncReward源码固定{source['syncreward_commit']}，tracked diff为空；Stage2 SHA匹配官方SHA256SUMS。",
        '- 没有发现控制方向、单位、源文件匹配、随机裁窗、错误音轨或分数方向读取错误。', '',
        '## 二、PEAVS：实质批量依赖', '',
        '旧包装器使用--batch_size1；官方run_PEAVS.sh直接运行metric.py，默认64。保留全部发布源码和权重，直接执行原始metric.py，仅拦截其删除缓存的清理命令。',
        f"- 旧batch1输出与原始脚本batch1的60项最大差{peavs['max_cached_head_parity_error']}。",
        f"- 改成默认64，最大单项差{peavs['max_batch1_vs_default64_error']:.7f}。",
        f"- 最小复现：同一视频音频特征{minimum['audio_shape']}、视频特征{minimum['video_shape']}，单独评分{minimum['original_alone']:.6f}，加入一条长视频后{minimum['original_with_companion']:.6f}。",
        '- 发布utils/dataset.py把cur_dur_v写成len(cur_aud)。当音频特征比视频多1步，混合批次中视频补零位置可能被当有效位置使用。生成视频尾部短于音频，加上I3D不满窗舍弃和AAC补齐，会触发此输入。',
        f"- 仅作为诊断，将两侧特征截到等长后，单独/混合差降至{abs(minimum['equalized_diagnostic_alone']-minimum['equalized_diagnostic_with_companion']):.9g}；这不是官方替代分数，未改写原始实验或官方代码。",
        f"- CuPy RawModule兼容核与独立PyTorch定义最大误差{peavs['correlation_cuda_vs_torch_max_error']:.9g}；重新完整提取两项失败样本，分数与旧结果完全一致。", '',
        '|调用|400ms控制下降|本项目门槛|', '|---|---:|---:|',
        '|PEAVS旧batch1|26/40|32/40|', '|PEAVS官方默认64|25/40|32/40|', '',
        '默认64仍然存在同一上游padding错误，其分数依赖输入排序/批次组成，因此不能把“改回64”当作可靠性问题已解决。此处保存了全部80原生输入的排序和批次组成。', '',
        '原生80文件（20N+60cloudT）按默认64复算：', '',
        '|版本|自然均分|云TTS均分|TTS-N|TTS较高|与Sync-C冲突|', '|---|---:|---:|---:|---:|---:|',
        '|旧batch1|2.217287|1.813750|-0.403537|6/60|42/60|',
        f"|默认64，保留上游错误|{n['natural']:.6f}|{n['tts']:.6f}|{n['delta']:+.6f}|{n['tts_higher']}/60|{n['conflicts_vs_syncC']}/60|", '',
        '全部60配对偏好方向未变。严格同参考12来源的默认64分差为'+f"{native['subsets']['same_reference12']['delta']:+.6f}"+'，TTS较高仍3/36、冲突30/36。', '',
        '## 三、SyncReward：FP32与官方BF16', '',
        '- 旧“V100不能使用BF16所以必须FP32”的说法过强。设备无原生BF16，但当前Torch2.6支持包括仿真的BF16，完整模型本次已实际运行成功。',
        '- 旧版使用FP32而非官方score.py的CUDA BF16 autocast；官方支持降低encode_batch_size，旧版使用1代替默认18。本次保留低显存参数1，恢复官方BF16。',
        '- 复用发布的load_av、48k处理器、18x0.5s/stride0.25、共享中心4.75s窗口、编码器及奖励头。严格加载完整Stage2。',
        '- 加载优化保留：按官方配置建立骨干、恢复num_classes1024，再由官方build_reward_model严格加载全部Stage2；未下载随后会被完整覆盖的9GB初始backbone。它与逐字执行原始下载流程不同，但不存在未覆盖随机保留参数。',
        f"- 运行原始score.main的FP32兼容对照，3代表样本与旧包装器最大差{max(r['absolute_error'] for r in sync['parity'])}；分段批量1与2最大差{max(r['error'] for r in sync['encode_batch1_vs2']):.9g}。",
        '- 官方解码/16k->48k重采样后诱发偏移逐采样仍保留，内部区间最大误差0。长代表样本中心从0.565s开始，正负400ms控制的整个4.75s窗口都不含边缘补零，因此该反例不能归因于补零污染。', '',
        '|调用|下降|持平|上升|本项目门槛|', '|---|---:|---:|---:|---:|',
        f'|旧FP32|16/40|{old_flat}/40|{24-old_flat}/40|32/40|',
        f'|官方CUDA BF16|{lower}/40|{flat}/40|{up}/40|32/40|', '',
        f"全60控制媒体的FP32/BF16最大分数差{bf16['fp32_vs_bf16_max_score_difference']:.9f}；按升/平/降三分类，方向改变{direction_changes}项。BF16原始逐项结果和全部控制差值均保存。软件执行不能宣称H20参考硬件上的bit-exact复现。", '',
        '例：BAC009S0002W0122，FP32原生1.450853，提前400ms1.478407、延后400ms1.492406；原始score.main复测逐值一致。BF16代表三项的确切数值见scores.jsonl。', '',
        '## 四、可确认与不能确认的原因', '',
        '- 已确认：存在调用默认设置差异；PEAVS还有可复现的上游padding依赖。已对齐默认批量及官方精度，但不能因此让时移资格门自动通过。',
        '- 已排除：本次输入波形没有真实移位、视频被误改、起始PTS不同、错用音轨、分数方向读反、Stage2权重缺失，以及CuPy兼容层算错相关核。',
        '- 适用范围：PEAVS官方AVS基准明确排除了人脸，论文也说明未评估VoxCeleb/LRS等talking-face数据。I3D/VGGish的0.96s特征尺度也不同于专用唇同步检测。',
        '- 对SyncReward只能确认其在这批中文静态头像视频上的400ms错位排序不可靠，不能仅凭本审计锁定训练域或语义线索为唯一原因，更不能据模型投票宣布Sync-C有偏好。', '',
        '## 五、调用版本与证据', '',
        '- 新PEAVS入口：/home/wjj/av-eval-20261026/score_peavs_official64_20261027.py，默认metric_batch_size64，明确标注上游批次依赖；真实2视频smoke已验证。',
        '- 新SyncReward入口：/home/wjj/av-eval-20261026/score_syncreward_official_20261027.py，默认bf16，显式--precision float32保留历史重放；旧包装器不覆写，以保护冻结协议。',
        '- 旧原生SyncReward均分/胜率仍属于FP32版本；本审计重测控制，不把它们伪称为已重算的BF16原生TTS结果。',
        '- 未恢复用户取消的交叉音轨评测。诊断媒体只对同一音轨施加已知偏移。', '',
        '源与限制：[PEAVS官方仓库](https://github.com/amazon-science/avgen-eval-toolkit)、[PEAVS论文](https://arxiv.org/html/2404.07336v1#S7)、[SyncReward官方仓库](https://github.com/WeChatCV/SyncReward)。', '',
        '详细收据：media_and_cached_features_audit.json、peavs_post_aac_timing_audit.json、source_and_weights_audit.json、peavs_all_repo_weights_audit.json、vggish_weights_audit.json、syncreward_official_parity.json、syncreward_bf16_controls.json、peavs_official_parity.json、peavs_minimal_batch_dependency.json、peavs_native_default64_reanalysis.json。']
    (out / 'official_inference_audit_report_20261027.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    summary = dict(status='AUDIT_COMPLETE_DIFFERENCES_FOUND_CONTROLS_STILL_FAIL',
        peavs_old_lower=26, peavs_default64_lower=25, syncreward_fp32_lower=16,
        syncreward_bf16_lower=lower, syncreward_bf16_ties=flat, controls=40, threshold=32,
        peavs_default64_native=n, peavs_padding_dependency=minimum['batch_dependency'],
        source_or_media_corruption_found=False, original_wrappers_retained=True)
    (out / 'audit_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
