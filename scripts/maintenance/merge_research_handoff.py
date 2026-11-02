"""Compose a short BM overview and six detailed modules; output drafts only."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote
from portable_handoff import finalize

ROOT = Path(__file__).resolve().parents[2]
BM = ROOT / 'basic-memory'
BASE = 'Research/交接/'
MAIN = BASE + '研究交接总览.md'


def link(path, label):
    return f'[{label}]({quote(str(BM / path), safe="/")})'


def clean(text):
    text = re.split(r'^## (?:Observations|Relations)\s*$', text, maxsplit=1, flags=re.M)[0]
    return text.strip()


def compose(documents):
    by = {d['identifier']: d for d in documents}
    sections = []
    def section(path, anchor, heading):
        text = clean(by[path]['content'])
        text = re.sub(r'^# .+\n', '', text, count=1).strip()
        # A single hierarchy under the containing section, preserving subheadings.
        text = re.sub(r'^(#{2,5}) ', lambda m: '#' + m[1] + ' ', text, flags=re.M)
        sections.append((path, anchor, f'<a id="{anchor}"></a>\n\n## {heading}\n\n{text}'))
    section(BASE+'第一部分-研究动机.md', 'motivation', '第一部分：研究动机')
    section(BASE+'第二部分-研究模块.md', 'modules', '第二部分：六个研究模块')
    section(BASE+'第三部分-阅读说明与索引.md', 'reading', '第三部分：实验阅读说明')
    for n in range(1, 7):
        path = next(p for p in by if p.startswith(BASE+f'第三部分-{n:02d}-'))
        section(path, f'experiments-{n}', by[path]['title'])
        section(BASE+f'来源-模块{n:02d}.md', f'sources-{n}', f'模块{n}的补充来源')
    section(BASE+'第三部分-研究接手与历史资料.md', 'handover', '研究接手、修正与历史资料')
    anchors = {path: anchor for path, anchor, _ in sections}
    titles = {by[path]['title']: anchor for path, anchor in anchors.items()}
    def internal(text):
        for path, anchor in anchors.items():
            target = quote(str(BM/path), safe='/')
            text = text.replace(']('+target+')', '](#'+anchor+')')
        text = re.sub(r'\[\[([^\]]+)\]\]', lambda m: f'[{m[1]}](#{titles[m[1]]})' if m[1] in titles else m[0], text)
        return text
    roles = {
        'Research': '研究动机、路线、设计和综合判断。先读本页；具体结论继续核对原实验和后续修正。',
        'Experiments': '按主题命名的实验记录，也包含方案和中途探索；标题出现某模型不代表已实际运行。',
        'docs': '正式报告、部署、结果表、工程决策及参考手册；下表进一步区分。',
        'Tasks': '较早建立的任务及任务板；旧状态可能已经过时，不能当实时进程状态。',
        'codex': '会话交接、任务、恢复快照和小事实；用于恢复执行背景，不替代最终结论。',
        'Papers': '论文阅读笔记和外部方法依据，不是本项目的实验结果。',
        'ToolNotes': '工具安装、连接和存储经验；实际操作仍需核对当前环境。',
        'Pitfalls': '历史故障与解决经验；不能将工程失败写成科学否定。',
        'concepts': '想法与概念解释；是否经过验证须另查实验。',
        'instructions': '记忆库启动、搜索和实验记录要求。接手的助手先读Startup Router。',
        'schemas': '任务、决策和会话笔记的记录格式。',
        '_imported_legacy': '迁移保留的旧资料；存在副本和旧路径，不作为默认写入入口。',
    }
    nav = '# 研究交接总览\n\n**本页是项目研究说明与整个 Basic Memory 文档库的总入口。** 项目首先观察到，部分TTS驱动视频比自然录音驱动视频获得更高Sync-C；研究工作围绕解释这个现象并尝试迁移收益展开。SyncNet可能偏向TTS是待检验解释，目前尚未证明。\n\n动机、六个研究模块、133段实验或研究工作说明及补充来源已合并在本文件中，可以连续阅读。原始报告、任务和结果仍保留独立文件，作为核对出处。旧分篇链接转到本页对应位置，避免维护多份正文。\n\n## 阅读目录\n\n'
    nav += '- [整个文档库怎样使用](#library)\n'
    nav += '\n'.join(f'- [{heading}](#{anchor})' for _, anchor, body in sections for heading in [body.split('## ',1)[1].split('\n',1)[0]])
    nav += '\n- [数据、模型、资产与维护规则](#maintenance)\n\n<a id="library"></a>\n\n## 整个文档库怎样使用\n\n第一次接手时，先读本页的动机与六模块，再按问题阅读实验。要继续执行工作，还应读仓库的CONTEXT.md和HANDOFF.md，以及[Startup Router]('+quote(str(BM/'instructions/Startup Router.md'), safe='/')+')与[实验指令]('+quote(str(BM/'instructions/实验指令.md'), safe='/')+')。要核对某个判断，沿实验后的来源链接读原文，并检查后续修正；本页的概括不能代替运行产物。\n\n下表说明各类资料的用途和阅读入口。\n\n| 目录 | 用途与阅读边界 |\n|---|---|\n'
    for name, role in roles.items():
        nav += f'| {link(name, name+"/")} | {role} |\n'
    nav += '\n### 正式资料怎样区分\n\n| 位置 | 用途 | 直接入口 |\n|---|---|---|\n'
    for directory, role, target in [
        ('experiments','按编号查正式实验报告；历史报告只读，后续修正另查。','experiments/README.md'),
        ('results','查分数汇总，具体条件和有效样本仍以对应报告为准。','results/SYNC-C-SCORES.md'),
        ('deployment','查模型部署环境和命令；部署成功不等于已有实验结论。','deployment'),
        ('reference','查数据、流水线、服务器和资产恢复手册。','reference/dataset-reference.md'),
        ('adr','查为什么作出某项工程或协议决策。','adr'),
        ('agents','查协作、任务和仓库工作约定。','agents'),
        ('superpowers','保留已有设计与计划；是否执行要另查任务和结果。','superpowers'),
    ]:
        nav += f'| {link("docs/"+directory,directory+"/")} | {role} | {link("docs/"+target,"查看")} |\n'
    nav += '\n### 按接手问题找资料\n\n'
    for label, path in [
        ('数据集、说话人数量、输入和参考材料的角色','Research/交接/数据与队列使用清单.md'),
        ('视频模型、TTS、语音识别和九个评分方法','Research/交接/模型与评分器使用清单.md'),
        ('综合判断及历史证据修正链','Research/研究主线与证据地图.md'),
        ('原始来源台账与核验范围','Research/交接/实验台账与核验口径.md'),
        ('停止或暂停过的研究路线','Research/历史研究路线与停止边界.md'),
        ('流水线步骤与产物','docs/reference/pipeline.md'),
        ('服务器访问资料','docs/reference/server-access.md'),
        ('历史本地副本、视频与路径恢复','docs/reference/local-copy-and-video-metrics.md'),
        ('统一网盘位置的决策','docs/adr/049-unified-baidu-project-root.md'),
    ]:
        nav += f'- {link(path,label)}。\n'
    nav += '\n本库按文件角色组织，部分目录来自不同阶段，因此Tasks与codex中的任务不能只凭日期去重。查过往实验时，先在项目内全文搜索，并同时覆盖Experiments和docs/experiments；同一实验的方案、任务、恢复与最终报告应一起看。没有搜到时换中文标题、英文名称和编号重试，不能直接断言未做过。\n\n'
    body = nav + '\n\n'.join(internal(s) for _, _, s in sections)
    body += '\n\n<a id="maintenance"></a>\n\n## 数据、模型、资产与维护规则\n\n'+link('Research/交接/数据与队列使用清单.md','数据与队列清单')+'及'+link('Research/交接/模型与评分器使用清单.md','模型与评分器清单')+'集中维护可复核的使用信息。Markdown库主要保存说明与出处；音频、视频、数组、模型权重和运行凭据通常位于仓库results、runs、数据目录、服务器或归档盘。存在笔记不等于完整媒体已下载，实际恢复要查文件清单、路径映射及校验凭据。历史Windows和Linux路径保留原样，通过映射找到现位置，不能改写冻结凭据。\n\n本页承担连续阅读与导航；原始实验承担证据；任务承担执行状态；HANDOFF承担当前快照。新工作先按Startup Router指定位置记录，再更新本页相关实验、数据模型清单与证据地图。新增结论应区分已观察结果、解释猜测、仅有方案、部分完成和检查失败。暂停状态只有获得对应授权才能改变。\n\n本次合并保留原来的84份编号报告、133段主体说明及全部518条研究来源。旧分篇页仅作兼容跳转，M1至M7是历史来源分类，不是新版六模块编号。本页不是执行计划，不启动生成、评分或训练。\n\n维护工具只生成待审稿。build_readable_research_handoff.py已接入单文件合并；merge_research_handoff.py也可从合并前审核稿重新生成。发布必须审查差异并经BM写入；不能用旧生成稿覆盖后来人工确认的新结论。\n\n## Observations\n\n- [decision] 本页为完整研究正文和整个BM文档库导航的统一入口，分篇页只保留跳转。\n- [limitation] 文档、段落、任务和独立实验数不同，导航覆盖不等于科学验收。\n\n## Relations\n\n- relates_to [[研究主线与证据地图]]\n- relates_to [[数据与队列使用清单]]\n- relates_to [[模型与评分器使用清单]]\n'
    result = [{'identifier':MAIN,'title':'研究交接总览','content':body}]
    for path, anchor in anchors.items():
        title = by[path]['title']
        result.append({'identifier':path,'title':title,'content':f'# {title}\n\n本节正文已合并到'+link(MAIN,'研究交接总览').replace(')',f'#{anchor})')+'。本页仅保留旧链接兼容入口，后续请在合并正文中阅读与维护。\n\n## Relations\n\n- part_of [[研究交接总览]]\n'})
    result.append({'identifier':'README.md','title':'TTS-TFG 文档库导航','content':'# TTS-TFG 文档库导航\n\n本库的统一入口是'+link(MAIN,'研究交接总览：完整研究说明与文档库导航')+'。其中包含研究动机、六个模块、详细实验、补充来源，以及全库目录用途、数据模型和资产恢复入口。\n\n首次接手请从该文档连续阅读；助手开始工作前仍须先读'+link('instructions/Startup Router.md','Startup Router')+'。原始报告与冻结凭据保留在原位置。\n\n## Relations\n\n- organizes [[研究交接总览]]\n'})
    # Keep motivation and the module overview in the entry page. Load detailed
    # experiments only when needed, with each module owning its source appendix.
    main = result[0]['content']
    detailed = sections[2:]
    for path, anchor, section_body in detailed:
        main = main.replace(internal(section_body), '')
        heading = section_body.split('## ', 1)[1].split('\n', 1)[0]
        target = path
        if anchor.startswith('sources-'):
            continue
        main = main.replace(f'[{heading}](#{anchor})', link(target, heading))
    main = re.sub(r'^- \[模块[1-6]的补充来源\]\(#sources-[1-6]\)\n', '', main, flags=re.M)
    for path, anchor in anchors.items():
        if anchor not in ('motivation', 'modules'):
            main = main.replace(f'](#{anchor})', ']('+quote(str(BM/path), safe='/')+')')
    main = main.replace('动机、六个研究模块、133段实验或研究工作说明及补充来源已合并在本文件中，可以连续阅读。原始报告、任务和结果仍保留独立文件，作为核对出处。旧分篇链接转到本页对应位置，避免维护多份正文。', '本页保留研究动机、六模块概览和全库导航。第三部分按六个模块分别保存，133段实验或研究工作说明及补充来源在需要时再打开；阅读说明与接手资料也独立保存。主入口不展开详细实验，减少每次接手时需要读取的内容。')
    main = main.replace('本页承担连续阅读与导航；', '本页承担研究概览与导航；')
    main = main.replace('再更新本页相关实验、数据模型清单与证据地图', '再更新对应模块实验说明、本页概览、数据模型清单与证据地图')
    main = main.replace('本次合并保留原来的84份编号报告、133段主体说明及全部518条研究来源。旧分篇页仅作兼容跳转，', '第三部分六份模块文档及配套接手资料保留原来的84份编号报告对应说明、133段主体说明及全部518条研究来源；各模块的补充来源并入对应模块，原来源页只保留跳转。')
    main = main.replace('build_readable_research_handoff.py已接入单文件合并；merge_research_handoff.py也可从合并前审核稿重新生成', 'build_readable_research_handoff.py输出简短总览与六份详细模块；merge_research_handoff.py也可从完整分篇审核稿重新生成该结构')
    main = main.replace('本页为完整研究正文和整个BM文档库导航的统一入口，分篇页只保留跳转。', '本页为研究概览和整个BM文档库导航的统一入口，详细实验按六模块独立保存。')
    result[0]['content'] = re.sub(r'\n{4,}', '\n\n', main)
    output_by = {d['identifier']:d for d in result}
    for n in range(1,7):
        path = next(p for p in by if p.startswith(BASE+f'第三部分-{n:02d}-'))
        source = BASE+f'来源-模块{n:02d}.md'
        body = clean(by[path]['content'])
        body = body.replace(link(source, '本模块的逐条补充来源'), '[本模块的逐条补充来源](#supplement)')
        appendix = clean(by[source]['content'])
        appendix = re.sub(r'^# .+\n', '', appendix, count=1).strip()
        appendix = re.sub(r'^返回.+?。\n', '', appendix, flags=re.M)
        body += '\n\n<a id="supplement"></a>\n\n## 逐条补充来源\n\n'+appendix+'\n\n返回'+link(MAIN,'研究交接总览')+'。\n\n## Relations\n\n- part_of [[研究交接总览]]\n'
        output_by[path]['content'] = body
        output_by[source]['content'] = '# '+by[source]['title']+'\n\n补充来源已并入'+link(path,'对应模块的详细实验与来源').replace(')', '#supplement)')+'。本页仅保留旧链接入口。\n\n## Relations\n\n- part_of [['+by[path]['title']+']]\n'
    for path in [BASE+'第三部分-阅读说明与索引.md', BASE+'第三部分-研究接手与历史资料.md']:
        output_by[path]['content'] = by[path]['content']
    output_by['README.md']['content'] = output_by['README.md']['content'].replace('研究交接总览：完整研究说明与文档库导航','研究交接总览：研究概览与文档库导航').replace('其中包含研究动机、六个模块、详细实验、补充来源，以及全库目录用途、数据模型和资产恢复入口。','其中保留研究动机、六模块概览和全库目录导航。详细实验与补充来源按六模块独立保存，通过总览中的链接按需阅读。').replace('从该文档连续阅读','先读该总览，再按问题打开对应模块')
    return finalize(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    docs = compose(json.loads(args.input.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'publish.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2)+'\n')
    for doc in docs:
        (args.output/Path(doc['identifier']).name).write_text(doc['content'])
    print(json.dumps({'documents':len(docs),'merged_bytes':len(docs[0]['content'].encode())}))
