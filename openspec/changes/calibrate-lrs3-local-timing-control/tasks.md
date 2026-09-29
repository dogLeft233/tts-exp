## 1. 读取契约并完成只读审计

- [x] 1.1 阅读本 change 的 proposal、design、delta spec 和 design 列出的历史材料；核对工作区现状，记录本次 run-id 与历史 final/cohort hashes，确认只使用固定 22 个 fit groups。
- [x] 1.2 实现 audit 入口并审计全部 22 records 的四个控制 cell；交付逐项真实文件证据的 `00_audit/audit.json`。验证 wrong-arm、缓存冲突、mux PCM 错误、损坏 hash 和缺失证据的测试分别得到正确审计状态。
- [x] 1.3 根据 audit 固定唯一分支：DEFECT_FOUND 时补最小失败回归并修复，测试通过才锁定 REPAIR_ONLY；NO_DEFECT_FOUND 锁定 SMOOTH_WARP；INCONCLUSIVE 写 BLOCKED final 并停止媒体执行。验证分支选择不读取本次新分数，不能按结果切换。

## 2. 实现最小控制实验

- [x] 2.1 新建显式 run-root 的 runner/protocol，冻结配置和代码绑定；用临时目录测试全新 `--stage all` 按依赖顺序工作、非法 run-id 被拒绝、任何失败写入新 run 而非历史目录。
- [x] 2.2 实现 spec 中的 LOCAL_SWAP 与 LOCAL_WARP_120 变换及 PCM/QC 清单；以 ramp、impulse、PCM 极值、恒零和过短数组测试映射方向、±位移、单调性、端点/长度/舍入、无溢出及失败条件；确认 N/N_REPEAT byte identity。
- [x] 2.3 接入三个独立 render 和四 cell mux/score，复用经审计的基础函数；用 stub 验证两个分支各为 66 videos/88 cells、实际音频参数正确、work/reference 隔离、视频与 PCM/PTS 验证，以及缺失 offset/歧义日志不能默认成功。
- [x] 2.4 实现三个 gate 和终态输出；用合成配对分数覆盖全通过、own-validity 失败但 sensitivity 通过、damage 符号反向、严格阈值相等和 offset/count 边界，确认 eligibility 始终 false。
- [x] 2.5 实现独立 validator 和 exact-identity resume；通过篡改输入/hash/cell/日志/final pass 标志的测试，确认 validator 能复算并拒绝，partial cell 不被覆盖，原 run 文件不变。

## 3. 验证并执行固定实验

- [x] 3.1 运行新测试目录、历史 confirmation 测试及被修改公共函数对应回归；运行受改动 Python 的 Ruff/编译检查与 `openspec validate calibrate-lrs3-local-timing-control --strict`，记录通过结果后才开始新模型调用。
- [x] 3.2 按 design 的 CLI 执行已锁定分支；确认协议/QC 先于新评分，完整产出 66 videos/88 cells，或以准确缺失项输出 BLOCKED；科学失败不触发另一次实验。
- [x] 3.3 运行独立 validator，交付 `05_final/final.json`、`result.md`、`validation.json`；核对历史 hashes 未变，结果说明包含审计证据、所选分支、三个 gate 和解释边界，然后结束本 change 的实验工作。
