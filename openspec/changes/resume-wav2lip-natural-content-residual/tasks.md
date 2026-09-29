## 1. 实现准备与只读导入

- [ ] 1.1 按design阅读原协议和BM；将本续跑笔记更新running，核对P/D文件hash，锁定新revision与run。
- [ ] 1.2 实现小型continuation入口、reuse manifest与独立validator；复验16条/8组、四臂构造、静态face、PCM和全部来源绑定。

## 2. 控制与跨会话验收

- [ ] 2.1 从真实embeddings重算旧13/16、配对16/16及未补偿anchor；保留父历史终态。
- [ ] 2.2 宿主CUDA预检；执行两个固定scorer parity与两个N_REPLAY生成/评分，检查像素、PCM、PTS和数值一致。
- [ ] 2.3 独立controls验收PASS；测试all/candidates/resume均不能绕过；写阶段self-review和BM。

## 3. 完成48个原定候选

- [ ] 3.1 门禁通过后串行生成16×3候选，完整N PCM mux、原31列/U评分；保存媒体、embeddings、矩阵与实际预算。
- [ ] 3.2 按原8组bootstrap计算CORRECT/N、CORRECT/WRONG、CORRECT/SHUFFLE的C/D/A与范数审计，生成唯一科学标签。

## 4. 验收与交付

- [ ] 4.1 通过聚焦回归及受影响父包测试；独立all验收重建全部科学计算、验证父run未变。
- [ ] 4.2 保存真实self-review、最终报告和final绑定；执行OpenSpec strict校验。
- [ ] 4.3 更新并读回同一BM续跑笔记：状态、三组效应/CI、控制数、预算、run与下一步边界；只勾实际完成项。
