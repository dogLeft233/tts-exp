"""Prepare BM library navigation and mechanical link repairs, without publishing.

Frozen experimental reports, checkpoints and imported source records stay intact.
Document status is quoted from metadata; it is never interpreted as live state.
"""
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote

from portable_handoff import BM, ROOT, MAIN, HANDOVER

OUT = ROOT/'results/memory_library_20261102'
OLD = {'研究主线与证据地图': '研究交接总览', 'Startup Router':'研究交接总览'}


def rel(source, target):
    return quote(os.path.relpath(BM/target, (BM/source).parent), safe='/')


def body(raw):
    return raw.split('---', 2)[2].lstrip() if raw.startswith('---\n') else raw


def title(raw, path):
    m = re.search(r'^title:\s*(.+)$', raw, re.M)
    return m[1].strip('"\'') if m else path.stem


def frozen(p):
    s = str(p.relative_to(BM))
    return (s.startswith('_imported_legacy/') or s.startswith('docs/superpowers/archive/')
            or (s.startswith('docs/experiments/') and p.name != 'README.md')
            or p.name.startswith('Codex checkpoint')
            or 'Experiments/assets/' in s)


def repair(text, source):
    def fix(m):
        value = unquote(m[1])
        path, sep, anchor = value.partition('#')
        if path.endswith('Startup Router.md'):
            return ']('+rel(source, MAIN)+')'
        if path.endswith('研究主线与证据地图.md'):
            return ']('+rel(source, HANDOVER)+'#evidence-history)'
        for prefix in (str(ROOT), 'E:/Documents/tts-exp', 'E:/Documents/tts-audio/tts-exp'):
            if path.startswith(prefix+'/') or path == prefix:
                target = ROOT/path[len(prefix):].lstrip('/')
                return ']('+quote(os.path.relpath(target,(BM/source).parent),safe='/')+(sep+anchor if sep else '')+')'
        return m[0]
    text = re.sub(r'\]\(([^)]+)\)', fix, text)
    for old, new in OLD.items():
        text = text.replace('[['+old+']]', '[['+new+']]')
    for old in ['memory://instructions/startup-router', 'memory://tts-exp/instructions/startup-router', 'memory://tts-exp/research/研究主线与证据地图']:
        text = text.replace(old, 'memory://tts-exp/research/交接/研究交接总览')
    text = text.replace(str(ROOT)+'/', './').replace(str(ROOT), '项目仓库根目录')
    text = re.sub(r'E:[/\\]Documents[/\\](?:tts-audio[/\\])?tts-exp(?![A-Za-z0-9_-])', '.', text, flags=re.I)
    text = re.sub(r'C:(?:/|\\)+Users(?:/|\\)+[^/\\\s`"<>]+', '${USER_HOME}', text, flags=re.I)
    return text


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(BM.rglob('*.md'))
    originals = {str(p.relative_to(BM)):p.read_text() for p in paths}
    docs = {}
    def add(path, heading, content):
        docs[path] = {'identifier':path,'title':heading,'content':content}
    for p in paths:
        if frozen(p):continue
        name=str(p.relative_to(BM)); raw=originals[name]; new=repair(body(raw), name)
        if new != body(raw):add(name,title(raw,p),new)

    categories = {
        'Research': ('研究设计资料索引', '这里保留研究设计、路线边界和综合分析。阅读主线统一从研究交接总览进入，方案是否执行必须继续查实验和任务。'),
        'Experiments': ('专题实验记录索引', '这些记录包含实验、设计和修正，并非全部已完成。按六个研究模块读解释；需要查原始专题时从这里进入。'),
        'Tasks': ('历史任务索引', '任务板汇总已知暂停和未结边界，单条任务中的旧状态不代表当前正在运行。这里与codex任务都保留，不能仅凭相似标题合并实验。'),
        'codex': ('会话与恢复资料索引', '任务、会话恢复和小事实分开导航。恢复快照是历史现场，不承担实时状态或最终科学结论。'),
        'codex/tts-exp': ('项目任务与恢复记录索引', '先读HANDOFF确认接手快照，再按问题查具体任务。以Codex checkpoint开头的记录只用于恢复历史现场，原件不回写。'),
        'codex/remember': ('项目小事实索引', '这里只保存小事实；完整研究结论和执行状态分别查实验与任务。'),
        'Papers': ('论文阅读索引', '外部论文和方法笔记用于理解依据，不代表本项目已实现或获得同样结果。'),
        'ToolNotes': ('工具经验索引', '工具连接、安装和存储经验保留执行日期，实际使用时核对当前环境。'),
        'Pitfalls': ('故障排查索引', '历史错误和处理经验按问题查找，工程失败不解释为研究假设不成立。'),
        'concepts': ('概念与想法索引', '这里的解释和想法不自动具有实验验证；验证情况继续追查实验来源。'),
        'instructions': ('专门记录要求索引', '统一启动入口是研究交接总览，这里只放具体任务类型的记录要求。'),
        'schemas': ('笔记格式索引', '这些格式供任务、会话和决策记录使用，不是研究结论。'),
        'docs': ('正式资料分类索引', '实验报告、分数、部署、决策和参考手册按用途分开。研究导航统一从总览进入。'),
        'docs/adr': ('工程决策索引', '按编号查工程与协议选择及其理由。历史决定不等于当前运行环境，应结合后续替代记录。'),
        'docs/deployment': ('模型部署索引', '环境、代码版本、权重和命令在每个模型条目中；部署完成不表示已完成科学比较。'),
        'docs/results': ('评分汇总索引', '分数必须连同原协议、有效样本、后续修正和校正方法阅读，不混合不同批次排名。'),
        'docs/reference': ('数据与操作参考索引', '查数据角色、流水线、服务器、工具和资产恢复。包含历史执行位置的原记录不直接代表接收者的本机路径。'),
        'docs/agents': ('协作约定索引', '仓库任务、问题跟踪和协作约定；启动规则以研究交接总览为准。'),
        'docs/superpowers': ('旧设计与交接资料索引', '保留已有设计、封存方案和旧交接材料；它们不替代当前唯一总览，也不代表已执行。'),
    }
    for folder,(heading,explain) in categories.items():
        source=folder+'/README.md'
        entries=[p for p in paths if p.parent==BM/folder and p.name!='README.md']
        nested=sorted({str(p.relative_to(BM)).split('/')[len(Path(folder).parts)] for p in paths if (BM/folder) in p.parents and p.parent!=BM/folder} - {'导航'})
        content=f'# {heading}\n\n{explain}\n\n返回[研究交接总览]({rel(source,MAIN)})。\n\n'
        if nested:
            content+='## 子目录\n\n'
            for sub in nested:
                dest=folder+'/'+sub+'/README.md'
                # A directory link is valid when the historical nested tree has no index.
                if dest not in [k+'/README.md' for k in categories] and not (BM/dest).exists():dest=folder+'/'+sub
                content+=f'- [{sub}]({rel(source,dest)})。\n'
        regular=[p for p in entries if not p.name.startswith('Codex checkpoint')]
        checkpoints=[p for p in entries if p.name.startswith('Codex checkpoint')]
        groups=[('资料',regular),('历史恢复快照',checkpoints)]
        for group, items in groups:
            if not items:continue
            content+=f'\n## {group}\n\n'
            batches=[items[i:i+50] for i in range(0,len(items),50)]
            for i,batch in enumerate(batches,1):
                target=source if len(batches)==1 else folder+f'/导航/{group}-{i:02d}.md'
                listing=''
                for p in batch:
                    name=str(p.relative_to(BM));t=title(originals[name],p).replace('[','（').replace(']','）')
                    listing+=f'- [{t}]({rel(target,name)})。\n'
                if target==source:content+=listing
                else:
                    add(target,f'{heading}：{group}（{i}）',f'# {heading}：{group}（{i}）\n\n原记录按文件名排序，仅作查找，不代表执行先后或结果优劣。返回[分类索引]({rel(target,source)})。\n\n'+listing+'\n## Relations\n\n- part_of [['+heading+']]\n')
                    content+=f'- [{group}分册{i}]({rel(source,target)})。\n'
        content+='\n## Relations\n\n- part_of [[研究交接总览]]\n'
        if source in originals and title(originals[source],Path(source)) != heading:
            # Preserve a pre-existing specialized index and only repair its links.
            content=repair(body(originals[source]),source)
        add(source,heading if source not in originals else title(originals[source],Path(source)),content)

    source=MAIN
    main=docs.get(source,{'content':body(originals[source])})['content']
    for folder in categories:
        # The root table links now lead to readable indexes rather than OS folders.
        target=rel(source,folder)
        main=main.replace(']('+target+')', ']('+rel(source,folder+'/README.md')+')')
    policy='''\n### 文档与路径维护\n\n各分类索引只负责查找，研究路线与启动规则仍集中在本页。专题实验与任务索引按需打开，历史恢复快照独立分册，避免一次读取整个库。分类依据文件用途，不根据标题推断已使用模型、实验是否成功或当前是否运行。\n\n可维护说明中的仓库位置使用相对路径；命令若用相对位置，应从项目仓库根执行。接收者自行配置数据盘、模型盘、归档挂载和服务器目录，不把某位成员的个人电脑路径作为通用配置。历史编号报告、恢复快照、迁移原件及封存设计中记录的原执行位置属于来源凭据，保留其原样；恢复时另查路径映射，不改写历史来伪装迁移后的执行位置。\n\n遇到同名记录时先核对用途和来源：方案、任务、恢复与报告可能讲述同一次实验，不能简单删成一份。逐字重复的原件也只登记来源关系，不删除唯一备份。原实验中的输入、分数、统计与暂停状态不因文档整理而改变。\n'''
    if '### 文档与路径维护' not in main:
        main=main.replace('\n<a id="motivation"></a>',policy+'\n<a id="motivation"></a>')
    add(source,'研究交接总览',main)
    root=body(originals['README.md'])
    if '各类资料的用途和分类索引统一见上述总览' not in root:
        root+='\n各类资料的用途和分类索引统一见上述总览；本页仅保留入口。\n'
    add('README.md',title(originals['README.md'],Path('README.md')),root)
    # Preserve byte-accurate sources outside BM; this file is not a progress note.
    changed={p:originals[p] for p in docs if p in originals}
    (OUT/'before.json').write_text(json.dumps(changed,ensure_ascii=False))
    (OUT/'publish.json').write_text(json.dumps(list(docs.values()),ensure_ascii=False))
    audit={'frozen':[], 'remaining_local_paths':[], 'identical_body_groups':[]}
    groups=collections.defaultdict(list)
    for p in paths:
        name=str(p.relative_to(BM));raw=originals[name]
        if frozen(p):audit['frozen'].append({'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        current=docs.get(name,{'content':raw})['content']
        if re.search(r'/Users/|[A-Z]:[/\\]',current):audit['remaining_local_paths'].append({'path':name,'protected':frozen(p)})
        groups[hashlib.sha256(body(raw).encode()).hexdigest()].append(name)
    audit['identical_body_groups']=[v for v in groups.values() if len(v)>1]
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({'drafts':len(docs),'existing_edits':len(changed),'new_indexes':len(docs)-len(changed),'frozen':len(audit['frozen']),'remaining_path_notes':len(audit['remaining_local_paths']),'exact_duplicate_groups':len(audit['identical_body_groups'])}))


if __name__=='__main__':build()
