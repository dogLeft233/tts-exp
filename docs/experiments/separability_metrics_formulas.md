# Wav2Sem 分离度度量完整公式

---

## 一、帧→音素池化

对每个 token，取 `start_s ~ end_s` 内的帧向量求 **mean pooling**：

$$\text{token\_vec} = \frac{1}{|\text{frames}|} \sum_{t = t_{\text{start}}}^{t_{\text{end}}} \mathbf{h}_t,\quad \mathbf{h}_t \in \mathbb{R}^D$$

输出：pooled `(N_tokens, D)` + labels `(N_tokens,)`

---

## 二、7 度量公式

记 $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ 为池化后的 token 嵌入及其类别标签，$C$ 个类，$\mathbf{\mu}_c$ 为类 $c$ 的质心，$\tilde{\mathbf{\mu}}_c$ 为单位化质心。

### (1) Intra-class Compactness

类内平均成对余弦距离，越低越紧凑。

$$d_{\text{intra}} = \frac{1}{|C|} \sum_{c} \frac{1}{\binom{n_c}{2}} \sum_{\substack{i<j \\ y_i = y_j = c}} \big(1 - \cos(\mathbf{x}_i, \mathbf{x}_j)\big)$$

$n_c < 3$ 的类被跳过。

### (2) Inter-class Separation

类间质心余弦距离均值，越高越分离。

$$\tilde{\mathbf{\mu}}_c = \frac{\mathbf{\mu}_c}{\|\mathbf{\mu}_c\|_2}$$

$$d_{\text{inter}} = \frac{2}{|C|(|C|-1)} \sum_{c < c'} \big(1 - \tilde{\mathbf{\mu}}_c \cdot \tilde{\mathbf{\mu}}_{c'}\big)$$

### (3) Fisher Ratio

按维度平均的类间方差 / 类内方差比。

$$\bar{\mathbf{x}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{x}_i$$

$$SS_{\text{between}}^{(d)} = \sum_{c} n_c \big(\mu_c^{(d)} - \bar{x}^{(d)}\big)^2$$

$$SS_{\text{within}}^{(d)} = \sum_{c} \sum_{y_i=c} \big(x_i^{(d)} - \mu_c^{(d)}\big)^2$$

$$F_d = \frac{SS_{\text{between}}^{(d)} \;/\; (|C|-1)}{SS_{\text{within}}^{(d)} \;/\; (N - |C|) + \varepsilon}$$

$$\text{fisher} = \frac{1}{D} \sum_{d=1}^{D} F_d$$

$|C|<2$ 时返回 NaN。

### (4) Silhouette Score（余弦距离）

对每个样本，$a_i$ 为与其类内其他样本的平均余弦距离，$b_i$ 为到最近异类质心的平均余弦距离。

$$a_i = \frac{1}{n_{y_i} - 1}\sum_{\substack{j \neq i \\ y_j = y_i}} \big(1 - \cos(\mathbf{x}_i, \mathbf{x}_j)\big)$$

$$b_i = \min_{c \neq y_i} \frac{1}{n_c} \sum_{y_j = c} \big(1 - \cos(\mathbf{x}_i, \mathbf{x}_j)\big)$$

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}, \quad s_i \in [-1, 1]$$

$$\text{silh} = \frac{1}{N} \sum_{i=1}^{N} s_i$$

优先调用 `sklearn.metrics.silhouette_score(metric="cosine")`，不可用时用上述直接实现。

### (5) Linear Probe Accuracy

GroupKFold 5 折交叉验证，按 utterance sample_id 分组（保证同一句不跨 train/test），LogisticRegression(max_iter=2000, C=1.0)。

$$\text{acc} = \frac{1}{5}\sum_{k=1}^{5} \frac{1}{|\text{test}_k|} \sum_{i \in \text{test}_k} \mathbb{1}\big[\hat{y}_i = y_i\big]$$

$$\text{F1}_{\text{macro}} = \frac{1}{5}\sum_{k=1}^{5} \frac{1}{|C|}\sum_{c} \text{F1}_c^{(k)}$$

每组数 < CV 折数时自动降折数，<2 折返回 NaN。

### (6) Boundary Sharpness

音素边界处的帧间余弦变化均值，越高表示边界变化越剧烈。

$$B = \{\ \text{token end\_s} \text{ 对应的帧索引} \}$$

$$d_{\text{boundary}} = \frac{1}{|B|} \sum_{b \in B} \big(1 - \cos(\mathbf{h}_{\text{before}(b)},\; \mathbf{h}_{\text{after}(b)})\big)$$

在每句内独立计算后，跨句取均值。

### (7) Segment Stability

非边界的连续帧间余弦变化均值，越低表示段内越稳定。

$$\mathcal{S} = \{1, 2, \ldots, N_{\text{frames}}-1\} \setminus B$$

$$d_{\text{segment}} = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \big(1 - \cos(\mathbf{h}_i,\; \mathbf{h}_{i+1})\big)$$

---

## 三、配对统计检验（Natural vs TTS）

配对样本 $(\text{nat}_i, \text{tts}_i)$ 按 sample_id 匹配，每个度量独立检验。

### Paired Permutation Test（配对置换检验）

**H₀**：两条件下该度量相同

**步骤**：
1. 计算观测均值差 $\Delta_{\text{obs}} = \frac{1}{n}\sum_i (\text{nat}_i - \text{tts}_i)$
2. 随机翻转每对的符号（等价于随机分配 natural/tts 标签）：$\tilde{D}_i = \pm(\text{nat}_i - \text{tts}_i)$
3. 算伪均值差 $\tilde{\Delta} = \frac{1}{n}\sum_i \tilde{D}_i$
4. 重复 2000 次

$$p = \frac{\#\{|\tilde{\Delta}| \ge |\Delta_{\text{obs}}|\} + 1}{2000 + 1}$$

### Bootstrap CI（自助法置信区间）

5000 次有放回重采样配对差 $\Delta_i = \text{nat}_i - \text{tts}_i$，取均值分布的分位数：

$$\text{CI}_{95\%} = [\text{percentile}_{2.5}(\bar{\Delta}^*),\; \text{percentile}_{97.5}(\bar{\Delta}^*)]$$

### Cohen's d（配对效应量）

$$\Delta_i = \text{nat}_i - \text{tts}_i$$

$$d = \frac{\overline{\Delta}}{\text{SD}(\Delta)} = \frac{\frac{1}{n}\sum_i \Delta_i}{\sqrt{\frac{1}{n-1}\sum_i(\Delta_i - \overline{\Delta})^2}}$$

解释：$|d| < 0.2$ 可忽略，$0.2 \le |d| < 0.5$ 小，$0.5 \le |d| < 0.8$ 中，$\ge 0.8$ 大。

### Benjamini-Hochberg FDR 校正

对所有比较（跨模型、层、等级、度量）的 p 值统一校正：

1. 按升序排列全部 $m$ 个 p 值：$p_{(1)} \le p_{(2)} \le \ldots \le p_{(m)}$
2. 找最大 rank $k$ 满足 $p_{(k)} \le \frac{k}{m} \cdot 0.05$
3. 校正后 p 值：$p_{(i)}^{\text{BH}} = \min\left(p_{(i)} \cdot \frac{m}{i},\; 1\right)$
4. $p_{\text{BH}} < 0.05$ 标记为 `significant_fdr = true`

---

## 四、参数配置

| 参数 | 值 |
|---|---|
| MIN_CLASS_SAMPLES | 3（少于 3 样本的类跳过） |
| 帧步长 | 320 samples @ 16kHz = 20ms |
| 池化 | mean over token interval |
| 置换检验迭代 | 2000 |
| Bootstrap 迭代 | 5000 |
| 线性探针 CV 折数 | 5 (GroupKFold by sample_id) |
| 显著性水平 α | 0.05 |
| FDR 校正方法 | Benjamini-Hochberg |

## 五、当前瓶颈

`process_all()` 循环结构导致每 `(condition, variant, level)` 组合都重新 `np.load()` 相同 `.npy` 文件：

| 数据 | 单文件大小 | 每层加载次数 | 总 I/O |
|---|---|---|---|
| XLS-R (13层, 1024维) | ~5MB | 78次 (6组合×13样本) | ~400MB/层 |
| HuBERT (13层, 768维) | ~3MB | 78次 | ~230MB/层 |

优化方向：内存缓存，一次 `np.load` 后按层切片复用。
