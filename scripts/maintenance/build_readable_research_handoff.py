"""Build the approved three-part handoff as drafts; never publish or run models.

The narrative source is manually reviewed prose. Inventory assignment is navigation
only, not a count of independent studies or evidence of model/data use.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

from research_handoff_narratives import REPORTS, TOPICS
from merge_research_handoff import compose

ROOT = Path(__file__).resolve().parents[2]
BM = ROOT / 'basic-memory'
BASE = 'Research/交接/'
LABELS = {
    1: '这个现象能否重复',
    2: '语速停顿和发音时间能否解释差异',
    3: '音量和声音特征能否解释差异',
    4: '模型怎样处理声音',
    5: '能否改善自然录音的结果',
    6: 'SyncNet的高分是否可信',
}
QUESTIONS = {
    1: '换一批录音、语言、TTS模型或视频生成模型后，TTS的分数还会更高吗？',
    2: 'TTS改变了哪些发音和停顿时间？把这些时间安排移到自然录音上，或将TTS视频放回自然时间，能否保留提高？',
    3: '音量、不同频段的能量和局部声音起伏是否影响结果？统一这些因素后，两组还差多少？',
    4: '声音进入模型后，哪些处理消除、保留或放大了差异？内部变化是否真的影响后续画面和评分？',
    5: '能否通过处理自然录音、调整内部信息或改变训练方式，得到可以用于新样本的改善？',
    6: 'SyncNet是否对TTS有偏好？其他评分是否同意？处理错误能解释多少，独立的画面或人工判断又支持什么？',
}
LEGACY_MAP = {'M1': 1, 'M2': 6, 'M3': 2, 'M4': 3, 'M5': 4, 'M6': 5, 'M7': 7}


def file_for(n):
    return BASE + f'第三部分-{n:02d}-{LABELS[n]}.md'


def local(path, label=None):
    p = ROOT / path
    return f'[{label or p.stem}]({quote(str(p), safe="/")})'


def note(path, label):
    return local('basic-memory/' + path, label)


def status_text(value):
    v = value.lower().replace('_', '-')
    if any(x in v for x in ['pause', 'paused']): return '原记录明确暂停，整理文档不会恢复执行'
    if any(x in v for x in ['control-failed']): return '计算有记录，但必要的评分检查未通过'
    if any(x in v for x in ['blocked', 'hold']): return '存在技术或材料阻碍，不能当作科学否定结果'
    if any(x in v for x in ['planned', 'proposal', 'spec', 'candidate', 'frozen']): return '属于方案或候选阶段，不能当作已完成实验'
    if any(x in v for x in ['no-go']): return '当前具体做法停止推进，范围以原记录说明为准'
    if 'partial' in v: return '只完成部分步骤，未完成部分没有结果'
    if any(x in v for x in ['concluded', 'complete', 'done', 'succeeded']): return '原记录登记该项工作完成；完成不表示假设获得支持'
    if any(x in v for x in ['implemented']): return '实现有记录，是否已有合格实验结果须看正文'
    return '原状态字段不能单独确定实验是否完成，请按原文及后续修正判断'


def navigation_module(record):
    """Conservative navigation, not an assertion about an experiment's result."""
    if record['module'] != 'M7':
        return LEGACY_MAP[record['module']]
    title = record['title']
    for n, pattern in [
        (6, r'评分|SyncNet.*(?:监督|偏好)|endpoint|parity|geometry|几何|视觉|ASR.*相关'),
        (5, r'训练|training|adapter|enhance|renderer|distill|objective|重建|residual|control direction'),
        (2, r'时钟|时间|同源参考|replacement|warp|interpol|phone|trajectory|duration|pause|LOCAL_SWAP|lag'),
        (3, r'电平|音量|来源方向|增益|谱形|预算|分配|声学|acoustic|waveform|vocoder'),
        (4, r'音素|编码|特征|表示|表征|separability|phoneme|wav2sem'),
        (1, r'跨数据|测评|复现'),
    ]:
        if re.search(pattern, title, re.I):
            return n
    return 7


def build(inventory, output):
    output.mkdir(parents=True, exist_ok=True)
    inv = json.loads(inventory.read_text())
    records = inv['records']
    numbered = [r for r in records if r['report_number'] is not None]
    cards = collections.defaultdict(list)
    assigned = {}
    used_anchors = set()
    for prefix, module, title, body in REPORTS:
        matches = [r for r in numbered if Path(r['path']).name.startswith(prefix)]
        assert len(matches) == 1, (prefix, matches)
        r = matches[0]
        cards[module].append((title, body, r))
        assigned[r['id']] = module
        used_anchors.add(r['id'])
    assert len(used_anchors) == len(numbered) == 84
    for module, anchor, title, body in TOPICS:
        matches = [r for r in records if Path(r['path']).stem == anchor]
        assert len(matches) == 1, (anchor, len(matches))
        r = matches[0]
        cards[module].append((title, body, r))
        assigned[r['id']] = module
        used_anchors.add(r['id'])
    for r in records:
        assigned.setdefault(r['id'], navigation_module(r))
    # Report titles are source identifiers; readable headings are separately authored.
    documents = []
    def add(path, title, content):
        documents.append({'identifier': path, 'title': title, 'content': content})
        target = output / Path(path).name
        target.write_text(content)

    add(BASE+'第一部分-研究动机.md', '第一部分 研究从哪里开始', '''# 第一部分 研究从哪里开始

## 1. 我们最初发现了什么

我们最先观察到：在一些实验中，将自然录音的文字重新合成为语音，再用它驱动数字人，生成视频的 Sync-C 分数会比直接用自然录音驱动时更高。项目由这个现象出发，之后的工作都在尝试弄清楚原因。

TTS是将文字转换为语音的技术。本项目通常还使用参考录音，让合成声音接近原说话人的声音。数字人视频生成模型根据声音驱动人物嘴部，有的还能生成头部动作。SyncNet是比较声音与嘴唇运动是否匹配的模型，Sync-C是它给出的一个分数，通常越高被解释为匹配越明确。

最初的比较同时改变了两个地方：生成视频时输入的声音变了，评分时使用的声音也变了。因此，TTS分数更高这一观察本身没有告诉我们，改变的是生成效果、评分响应，还是二者都有。

## 2. 为什么这个现象值得解释

一种可能是TTS更容易被生成模型处理，实际产生了与声音更配合的动作。另一种可能是TTS的音量、停顿或声音形状使评分更容易变高，画面的实际配合却没有同样改善。还有一种可能是声音转换、截取和时间处理造成了差异。它们也可能同时存在。

这些都是需要检查的解释。特别是，我们怀疑SyncNet可能更容易给TTS高分，但目前没有证明它存在系统性偏置。评分方法本来就应该对真实的声音与画面匹配差异作出响应，不能把任何声音处理引起的变化都叫偏置。

弄清原因可以帮助我们决定怎样准备生成输入、怎样改进模型，以及怎样评价新方法。如果只追求更高Sync-C，却不知道分数为何改变，可能会把评分上的提高误当成实际改善。

## 3. 我们沿着哪些问题研究

我们首先检查这个现象是否能重复，逐步更换数据、语言、TTS和视频生成模型。随后比较两类声音的语速、停顿、音量及其他差异，尝试分别改变这些因素。

为了检验这些差异是否有用，我们还尝试将TTS中的部分因素移到自然录音上，或在训练中改变声音处理方式。与此同时，我们检查评分过程，使用其他评分方法，并寻找与SyncNet分数相互独立的画面证据。

第二部分把这些工作分成六个模块。介绍顺序是为了让读者理解问题之间的关系，不强行作为每项实验的实际执行时间顺序。

## 4. 目前知道什么，还有什么没有解释

已有结果支持：TTS较高的Sync-C在多个数据和模型条件下出现过，但并非每批都能重复。同一套本地流程在新的数据上也出现过不明确或反向结果，因此不能把所有条件都概括为TTS更好，也不能继续概括为英文一定没有这个现象。

音量及其处理能够影响模型内部信息、生成的画面和部分评分；统一音量缩小过部分队列的TTS分差，但没有解释全部差异。不同声音实例即使文字相同，发音的具体时刻仍不同，互换声音后降分也可能是正常失配。

声音内部信息的差异、某个训练目标的改善和最终视频的改善不是同一件事。我们尚未得到一个在多个生成模型和新说话人上可靠复制TTS收益的通用处理方法。多评分器之间也存在分歧和测量限制，尚不能据此裁定SyncNet必然错误。

这些是按现有记录可写出的判断。更精确的样本、分数、检查失败和后续修正放在第三部分；后续若获得新证据，应更新综合判断，同时保留历史原报告。

## 5. 如何阅读后面的文档

先读第二部分理解六条研究路线，再按问题进入第三部分的具体实验。每项说明交代为什么做、怎样比较、用了什么、观察到什么，以及不能由结果推出什么。模型和数据清单帮助定位具体版本、材料和输出。

“完成”只表示所述工作已经执行并有记录，不表示假设成立。“未确认改善”不是证明没有差异；“检查未通过”也不是方法科学上无效。“仅有方案”“部分完成”和“用户暂停”会单独说明。

## Observations

- [decision] 项目总目的为解释观察到的TTS较高Sync-C，评分偏置是其中一条重要待检验解释。
- [limitation] 当前没有证明SyncNet存在系统性TTS偏置，也没有完成通用收益迁移。

## Relations

- part_of [[研究交接总览]]
- relates_to [[第二部分 六个研究模块]]
''')

    explanations = {
        1: ('它决定我们究竟要解释一个普遍现象，还是某些数据和流程下的结果。', '先固定能保持一致的输入和处理，再更换一项条件；无法只换一项时明确写出其他差异。', '已有多组中文及英文正差，也有未能重复和反向的队列。一个模型显著、另一个不显著，本身不证明两模型的差异显著。', '重复出现提供机制研究对象，重复失败要求我们重新检查来源和条件，不能只挑正结果继续解释。'),
        2: ('同一句文字可以有不同语速、停顿和每个字发生的时刻。把一种声音配给另一种声音驱动的画面，可能正常失配。', '分别标定两种声音的发音时间，检查已知变速能否被恢复，再做时间调整、固定画面换声音或调整画面时间。', '发音时间确实影响评分，部分修复能恢复局部结果；简单拼接或映射未形成普遍迁移方法，同类不同发声也会大幅降分。', '时间对不上时不能把交叉低分直接叫评分偏置。保持总时长相同，也不等于每段发音已经对齐。'),
        3: ('早期TTS和自然录音的音量等属性并不一致，它们可能同时影响生成模型与评分器。', '统一音量或只改变预定声音因素；重新生成时固定评分用的原自然声音，固定画面时则只处理评分声音。', '音量变化能产生实际生成响应并缩小部分TTS分差，其他声音交换依赖条件，尚无单一特征解释全部优势。', '固定评分声音有助于区分生成变化，但分数提高仍不等于实际嘴型更准确；声音操作可能同时改变多种属性。'),
        4: ('外部声音相同程度的变化，可能被不同模型消除或放大。内部表示是模型从声音中提取、供后续使用的信息。', '固定模型权重，检查各处理步骤和不同层的信息；只改变一个步骤，并用能恢复原结果的检查验证。', '已经定位部分标准化和数值步骤如何影响音量响应，也发现表示差异并不稳定预测下游分数。', '说明内部信息怎样变了，不等于说明完整TTS优势的原因。需要与模块三的生成检查及模块五的新样本效果衔接。'),
        5: ('如果确实找到有用因素，应该能在一定条件下转移到自然录音，或用于更合理地训练模型。', '处理自然声音、训练受限制的小模块或改变训练条件，先保留内容与时间，再在未参与选择的说话人上检查。', '做过多种原型、对照和训练，一些局部响应成立，但尚无跨模型稳定推广的方法；低音量训练也存在反向复核。', '技术上训练成功、拟合目标变好和最终交付改善须分开。保持失败、暂停和训练材料缺陷的记录，避免重复尝试已被限制的路线。'),
        6: ('最初现象由Sync-C观察到，因此必须检查分数实际测量了什么，以及是否存在偏向TTS的部分。', '核对公开代码和权重，移动声音制造已知失配，比较不同评分方法的同一批视频，并检查真实视频及独立画面指标。', '不同评分器判断不总一致，部分存在边界、批量或上限问题；Sync-C提高也未必伴随原时刻匹配改善。目前尚未证明系统性TTS偏置。', '不能把九个指标当简单投票。证明偏置还需在真实匹配程度可比较时，发现声音来源带来额外评分差，并有合格独立证据。'),
    }
    body = '# 第二部分 六个研究模块\n\n六个模块都服务于同一个目的：解释为什么某些TTS驱动的视频获得更高Sync-C。它们不是六个互不相关的模型，也不是按部署顺序排列的任务。\n\n'
    for n in LABELS:
        why, how, known, boundary = explanations[n]
        body += f'## 模块{n}：{LABELS[n]}\n\n**要回答的问题：{QUESTIONS[n]}**\n\n### 为什么提出这个问题\n\n{why}\n\n### 怎样比较\n\n{how}\n\n### 目前的认识\n\n{known}\n\n### 限制及与其他模块的关系\n\n{boundary}\n\n具体实验见{note(file_for(n), "第三部分对应实验")}。\n\n'
    body += '## Relations\n\n- part_of [[研究交接总览]]\n- relates_to [[第一部分 研究从哪里开始]]\n'
    add(BASE+'第二部分-研究模块.md', '第二部分 六个研究模块', body)

    for n in LABELS:
        body = f'# 第三部分 模块{n}：{LABELS[n]}\n\n**本模块的问题：{QUESTIONS[n]}**\n\n这里按研究问题合并介绍主要实验，补算、故障恢复和后续修正不另算独立复现。原报告标题可能保留技术用语，正文解释以本页及后续修正为准。没有独立结论的资料仍列在后面的逐条来源中。\n\n'
        for i, (title, prose, r) in enumerate(cards[n], 1):
            body += f'## {n}.{i} {title}\n\n{prose}\n\n资料：{local(r["path"], r["title"])}。\n\n'
        extras = [r for r in records if assigned[r['id']] == n and r['id'] not in used_anchors]
        source_path = BASE + f'来源-模块{n:02d}.md'
        body += f'## 本模块的补充实验、方案与修正记录\n\n另外{len(extras)}条来源记录列在{note(source_path, "本模块的逐条补充来源")}，包含子实验、方案、任务、复核与汇总。它们全部保留链接和状态说明，条目数不是独立实验数。\n\n'
        supplement = f'# 模块{n}补充来源：{LABELS[n]}\n\n下列{len(extras)}条是尚未在主体单独展开的来源记录，包括具体子实验、方案、任务、复核和汇总。同一实验可能有多条记录，不能把条目数当实验数。这里翻译原状态字段以帮助查阅；它不是重新验收，更不能用旧任务状态覆盖主体已核对的最新状态。文中提到某模型或数据，不自动表示实际使用。归类仅用于导航，真正关系以原文为准。\n\n返回{note(file_for(n), "本模块实验说明")}。\n\n'
        for r in extras:
            st = status_text(r['metadata'].get('status', ''))
            supplement += f'- {local(r["path"], r["title"])}。{st}。\n'
        supplement += '\n## Relations\n\n- part_of [[第三部分 实验阅读说明与索引]]\n'
        add(source_path, f'模块{n}补充来源：{LABELS[n]}', supplement)
        body += '\n## Relations\n\n- part_of [[研究交接总览]]\n- relates_to [[第二部分 六个研究模块]]\n- relates_to [[第三部分 实验阅读说明与索引]]\n'
        add(file_for(n), f'第三部分 模块{n}：{LABELS[n]}', body)

    support = [r for r in records if assigned[r['id']] == 7]
    counts = {n: len(cards[n]) for n in LABELS}
    intro = f'''# 第三部分 实验阅读说明与索引

## 1. 怎样区分实验与多次运行

本交接从五个来源目录核对到{len(records)}条研究文档，另保留{len(inv['checkpoint_provenance'])}条恢复快照。文档中有实验报告、方案、任务和复核；它们会重复指向同一次运行。编号报告有84个文件，07为空号，25有两份不同文件，原编号不改。我们不把这些数字称为独立实验数量。

本版用{sum(counts.values())}段较完整的实验或研究工作说明覆盖全部84份编号报告，以及后续专题、全量任务和重要修正。部分说明是调查、数据准备或技术验证，正文明确其性质，不把它们计作已完成的效果实验。其他来源逐条保留在各模块末尾与接手资料中，避免漏掉失败及未执行方案。

同一实验有多次运行时，先说明共同问题，再交代哪次改变了条件、哪次只是恢复故障。子集试跑被后续全量包含时，不算独立复现。跨模块的工作选择一处主要介绍，其他位置通过相关报告查阅。

## 2. 怎样理解数量与统计结果

同一位说话人的多句话、同一句话的多次TTS不能当成互不相关的新人。近期全量比较先在每个人内部平均，再让每个人同等参与总平均；逐句分布只是另一种描述。因此平均分差可能与逐句直接平均略有不同。

分差统一写成“TTS减自然”。Sync-C保留三位小数。百分比类评分的差写成“百分点”，不与Sync-C直接相加。有效样本不足、某一侧没有生成或评分失败时，明确说明每项指标实际用了多少完整配对。

“统计区间跨零”表示目前数据没有排除相反方向，并不证明两组一样。“校正后显著”指同时检查多项结果时已控制偶然找到差异的机会。近期九项比较采用Holm方法一起校正；只有单项区间为正，不能自动称通过九项检验。历史结果没有发布同口径校正值时写未知，不写不显著。

## 3. 每项说明应交代什么

每项实验用句子说明为什么做、比较什么、采用什么模型和数据、实际样本数量、保持哪些条件一致、观察到什么、能够支持什么，以及哪些解释仍未确定。正文不把模型内部信息更接近目标写成视频更准确，也不把自动识别结果当成人工听辨真值。

技术失败、控制失败、不可测、暂停和仅有方案分开记录。样本或字段未核实的地方不凭文件名补全；具体版本、权重、参数、输入和输出可由原报告继续追到运行清单。新写的是解释和导航，不回写冻结的历史报告及凭据。

### 阅读时需要的几个工具名称

ASR是自动把录音识别成文字的工具。识别正确率可以筛查内容异常，但机器转写不是人工确认的真实答案。MFA依据给定文字估计各段发音在声音中的时间；它可以帮助做时间对应，但不能仅凭对齐成功证明实际每个字都读对了。

HuBERT、WavLM、wav2vec2和XLS-R是从声音提取信息的模型，不是本项目中的视频生成模型。不同层可以理解为这些信息经过不同次数处理后的结果。AVE是SyncTalk使用的一条声音处理路径。HiFi-GAN负责把声音信息还原成可播放的声音，通常称为声码器；还原会有误差，不能默认与原录音完全一样。

WORLD和Harvest用于分析、估计和修改声音的音高等属性。RMS用来衡量声音整体幅度；统一RMS是在控制音量的一种技术做法，不保证人的主观听感完全相同。文中“频谱”指声音能量在不同频率上的分布，“局部包络”指声音强弱随时间的起伏。

训练用的数据帮助方法学习；验证用的数据帮助检查或选择方法；留出的测试数据应避免参与这些选择。已经反复查看或调整过的样本，不再当成全新验证。每次随机生成使用的种子是可记录的随机过程设置，多次种子不等于新增说话人。

## 4. 六个模块的实验入口

'''
    for n in LABELS:
        intro += f'- {note(file_for(n), f"模块{n}：{LABELS[n]}")}。本页有{counts[n]}段主体说明，另列相关来源。\n'
    intro += f'''
## 5. 数据、模型与接手资料

查数据角色和规模见[[数据与队列使用清单]]；查生成模型、TTS、语音识别和九个评分方法见[[模型与评分器使用清单]]。参照图来自HDTF不表示声音也来自HDTF。模型部署成功、工厂注册和实际完成实验分别登记。

全部{len(records)}条文档逐一对应到本版模块或接手资料；覆盖检查不代表逐运行去重完成，也不表示已经重新执行全部实验。机器可读对应表见{local(str((output/'readable_coverage.json').relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output/'readable_coverage.json'), '本版覆盖核对表')}。旧的M1至M7仍作逐条来源检索入口，旧编号不再代表新版模块顺序。

接手资料见[[第三部分 研究接手与历史资料]]。旧式原始索引见[[实验台账与核验口径]]，其模型和数据词项仅表示原文提到，不能充当实际使用证据。

## Observations

- [decision] 完整保留来源，按六个问题解释研究工作，记录数量不冒充独立实验数量。
- [limitation] 未在主体逐段展开的资料保留原文入口和状态边界，不凭标题生成科学结论。

## Relations

- part_of [[研究交接总览]]
- relates_to [[数据与队列使用清单]]
- relates_to [[模型与评分器使用清单]]
'''
    add(BASE+'第三部分-阅读说明与索引.md', '第三部分 实验阅读说明与索引', intro)

    handover = '''# 第三部分 研究接手与历史资料

## 1. 已被收窄或修正的旧说法

早期“英文不成立”不再作为总判断。LibriSpeech因来源重合风险不用于独立正负推断，LRS3和EARS后来有英文正差，其他小批负结果保留其条件范围。

早期将自然和TTS交叉换轨后的强烈下降解释为评分器偏好，证据不足。报告44显示同类声音不同发声也会大幅降分；报告45说明局部时间修正可以恢复部分配对，但覆盖不足。相同文字和总时长不能证明具体发音时刻相同。

早期均匀切片得到的部分发音分离或强关联已被真实发音时间分析修正；报告13撤回旧强相关，报告15没有找到稳定逐句分差预测指标。报告41的MuseTalk失败包含输入与观察位置错配，报告42已修正；功能动作改变仍不是同步准确度证明。

MTDVocaLiST旧自定义连续分数已被报告69指出不能当作官方检索指标，正式比较读报告70。PEAVS、SyncReward等的已知控制不足和批量依赖不得因计算完成而省略。

May的TTS训练分数下降必须与报告66的数据时序缺陷一起读。低尺度训练的单次正差也必须与报告67的反向无添加噪声结果一起读，不把局部结果升级为通用训练建议。

## 2. 当前暂停、未完成及仅有方案

HDTF20 TalkLikeYou由用户暂停。已经完成80生成、120个SyncNet评分，但九项及完整归档未闭合，不能恢复评分，也不能形成九项最终结论。

PC-Talk与DyStream的Chinese-LiPS全量生成、九项计算和本机自动核验均已完成，完整视频检查及正式最终报告待补。旧任务记录的运行中状态不表示仍有评分进程。DyStream冻结报告有沿用PC旧结果复用的文字，真实209视频为DyStream新生成；正式报告应纠正文案，冻结原件保留。

PC-Talk运动模板比较尚未执行。现有全量结果来自官方单图、未传运动参考的固定姿态路径；不能写成论文动态模板条件。DyStream用HDTF人物图配Chinese-LiPS声音，也不是已经完成HDTF音频基准。

Patrick同初始化复跑及旧增强路线保留原暂停边界。逐发音差方向仍停在拟合侧内容质量检查，开发和确认数据不动。二百人自然声音准备完成不等于4200个完整合成媒体已经完成。

## 3. 技术失败怎样保留

人脸无法检测、视频太短、输出没有正常停止、训练出现非有限数值、时间网格验证失败和依赖阻塞分别说明。它们不能直接用来证明研究假设错误。恢复执行时保留失败原件、修复内容和输入身份，不能只留下成功运行。

这次文档整理不连接服务器、不生成声音或视频、不重评分、不训练，也不恢复任何暂停任务。后续是否执行按单独任务授权和服务器要求处理。

## 4. 数据与产物从哪里找

当前工作入口是 /Users/wangjiajun/tts-exp。数据位置和用途见[[数据与队列使用清单]]，模型输出按results中的模型和运行目录保存。PC-Talk、DyStream全量产物在本机；HDTF TalkLikeYou的完整80个视频尚未全部下载，不能仅凭本机检查点认为视频可直接打开。

历史Windows盘符与旧Linux路径是当时执行位置。恢复时先查统一网盘根与路径映射，不改历史凭据伪造新位置。密码和API密钥不写入交接文件。

## 5. 继续研究前要核实什么

应先判断所用数据是否具有来源独立性、语音内容是否完整、每项比较是否具有相同有效时间范围，以及评分方法能否识别已知错位。确认这些条件后，再讨论能否区分真实同步变化与声音来源引起的额外评分变化。

目前缺少能够覆盖主要比较的合格独立同步判断。更多评分器或更高分不自动填补这个缺口。后续研究方向可以讨论，但本交接不把新建议混入已做实验。

## 6. 研究设计、工程核查与恢复资料逐条入口

以下资料保留原题名，主要用于追溯方案、材料准备、任务、修正和恢复。有些旧状态可能已被上文或后续正式报告替代；不把它们解释为实时进程状态。

'''
    for r in support:
        handover += f'- {local(r["path"], r["title"])}。{status_text(r["metadata"].get("status", ""))}。\n'
    handover += '\n## Relations\n\n- part_of [[研究交接总览]]\n- relates_to [[历史研究路线与停止边界]]\n- relates_to [[实验台账与核验口径]]\n'
    add(BASE+'第三部分-研究接手与历史资料.md', '第三部分 研究接手与历史资料', handover)

    overview = '''# 研究交接总览

**为什么TTS驱动的数字人视频会获得更高的同步分数？** 本项目首先观察到这一现象，随后尝试重复它、解释它，并探索能否将提高分数的因素用于自然录音。怀疑SyncNet可能偏向TTS是重要研究方向，目前还不是已证实结论。

本版按用户确认的三部分整理，面向没有接触过项目的接手者。研究主线是解释现象，正文使用完整句子；原始报告保留其编号、技术名称和历史结论，新的说明同时指出后续修正。

## 第一部分：研究动机

阅读[[第一部分 研究从哪里开始]]。依次说明最初发现、研究价值、四条主要思路、目前的认识和阅读方式。

## 第二部分：研究模块

阅读[[第二部分 六个研究模块]]。六个模块分别回答：现象能否重复；发音时间能否解释；音量和其他声音特征能否解释；模型内部如何处理差异；能否改善自然录音；SyncNet高分是否可信。

## 第三部分：实验及交接资料

先读[[第三部分 实验阅读说明与索引]]，再按问题进入下列实验。每个模块有完整句子的主体说明及逐条来源，区分观察、解释猜测、技术验证和未完成工作。

'''
    for n in LABELS:
        overview += f'- {note(file_for(n), f"模块{n}：{LABELS[n]}")}。\n'
    overview += f'''
数据与模型见[[数据与队列使用清单]]、[[模型与评分器使用清单]]。修正、暂停、失败与恢复入口见[[第三部分 研究接手与历史资料]]。

## 覆盖范围与证据边界

本版逐段说明全部84份编号报告，并补充后续专题与当前全量任务，共{sum(counts.values())}段实验或研究工作说明；另将{len(records)}条研究文档全部保留为可查来源，{len(inv['checkpoint_provenance'])}条恢复快照单列。段落数、文档数和独立实验数不同：调查、方案、任务、技术核验和重复运行不能计成同等的独立效果实验。

主体说明整理各实验的目的、做法、主要模型数据、观察及限制；未在主体独立展开的子记录保留链接和原状态解释，没有凭标题补写结果。原报告仍是证据出处，后续修正必须与它一起看。数据角色、训练来源重合、统计独立性和未通过检查的限制不会因整理文档而消失。

## 当前接手状态

PC-Talk与DyStream的Chinese-LiPS全量计算及本机自动验收完成，完整媒体检查及正式最终报告待补。HDTF20 TalkLikeYou保持用户暂停，九项没有全部完成。运动模板比较尚未执行；旧训练和增强路线的暂停不变。

## 维护方式

[[研究主线与证据地图]]保留更详细的问题与修正链。旧M1至M7页仅作为旧分类来源索引，新版六模块顺序以上述入口为准。新增或修正先更新正式实验与任务，再维护主体说明、对应清单和研究地图；历史冻结报告不静默改写。

机器清单由build_research_handoff.py生成，主体解释由research_handoff_narratives.py维护，build_readable_research_handoff.py只输出待审稿和覆盖表，不自动发布，不启动实验。原始索引见[[实验台账与核验口径]]。

## Observations

- [decision] 采用动机、模块、实验三部分结构，六个科学问题组织研究工作，历史恢复资料独立保留。
- [decision] 项目从发现TTS较高Sync-C出发，调查原因与尝试迁移；评分偏置是待检验解释。
- [limitation] 没有证明所有条件下TTS均更好，没有证明SyncNet系统性偏置，没有通用增强方法结论。

## Relations

- organizes [[第一部分 研究从哪里开始]]
- organizes [[第二部分 六个研究模块]]
- organizes [[第三部分 实验阅读说明与索引]]
- relates_to [[研究主线与证据地图]]
- relates_to [[数据与队列使用清单]]
- relates_to [[模型与评分器使用清单]]
'''
    add(BASE+'研究交接总览.md', '研究交接总览', overview)
    coverage = {
        'schema_version': 1,
        'purpose': 'Editorial navigation and source coverage, not a unique experiment count',
        'numbered_reports': len(numbered), 'main_descriptions': sum(counts.values()),
        'narrative_counts': counts, 'source_records': len(records),
        'checkpoint_provenance': len(inv['checkpoint_provenance']),
        'sources': [{'id':r['id'], 'path':r['path'], 'sha256':r['sha256'],
                     'new_module':assigned[r['id']], 'main_paragraph_anchor':r['id'] in used_anchors,
                     'destination':file_for(assigned[r['id']]) if assigned[r['id']] != 7 else BASE+'第三部分-研究接手与历史资料.md'} for r in records],
        'uncovered_records': [],
    }
    (output/'readable_coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2)+'\n')
    documents = compose(documents)
    for document in documents:
        (output / Path(document['identifier']).name).write_text(document['content'])
    coverage['presentation'] = 'Short canonical overview; six separate detailed modules with integrated source appendices'
    coverage['canonical_document'] = BASE+'研究交接总览.md'
    (output/'readable_coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2)+'\n')
    (output/'publish.json').write_text(json.dumps(documents, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'documents':len(documents),'descriptions':sum(counts.values()),'counts':counts,
                      'records':len(records),'max_bytes':max(len(d['content'].encode()) for d in documents)},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.inventory, args.output)
