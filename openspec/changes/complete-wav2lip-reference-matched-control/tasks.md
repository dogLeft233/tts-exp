# B 实施清单（代码、GPU运行与阶段审计完成）

## 1. 冻结与历史控制

- [x] 1.1 创建独立包/CLI和protocol，核验快照分支B及固定输入绑定；F0/F46固定帧、source/box 绑定和缺输入拒绝测试通过；全量16条/8组运行待执行。
- [x] 1.2 从旧F46 embeddings独立重建legacy/matched两域；physical offset 映射与历史 matched gate 已独立CPU重算并通过（16/16）；fresh GPU 控制仍待执行。

## 2. 当前运行时与候选

- [x] 2.1 历史控制通过后完成 F0 公共控制和 fresh F46_N/repeat/delay；B run `b_20260910_r2` 验证重复像素、两域矩阵、两套 lag/anchor 用途及独立 `control_validation.json` PASS。
- [x] 2.2 全控制通过后生成16个 F46_C 并评分；四 cell 完整、原N音轨身份、36视频/54评分上限均已核对，候选仅在控制 PASS 后生成。
- [x] 2.3 已实现共享 F0_N anchor 的 g0/g1/I 与固定 bootstrap；B 实测 I mean `0.003677`、99% CI `[-0.033373, 0.040314]`、4/8 组正，终态为 `NO_REFERENCE_INTERACTION_ESTABLISHED`，replacement 保持 false。

## 3. 交付

- [x] 3.1 独立重算矩阵/控制/统计并完成聚焦测试；篡改数字/绑定重签拒绝、旧PASS不能代替F46控制、lag/endpoint/PCM 校验和本change OpenSpec strict均通过。
- [x] 3.2 已写 `final.json`/`result.md` 并更新唯一 BM 实体；历史/ fresh F46 均16/16、I 已实测、预算36/54，所有授权保持 false，未执行分支为空。
