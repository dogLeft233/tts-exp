"""Portable Markdown links and a single navigation entry; no disk mutations."""
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[2]
BM = ROOT / 'basic-memory'
MAIN = 'Research/交接/研究交接总览.md'
HANDOVER = 'Research/交接/第三部分-研究接手与历史资料.md'


def portable(text, identifier):
    parent = (BM / identifier).parent
    def target(match):
        value = unquote(match[1])
        path, sep, anchor = value.partition('#')
        if path.startswith(str(ROOT)):
            if path == str(BM/'instructions/Startup Router.md'):
                path = str(BM/MAIN)
            elif path == str(BM/'Research/研究主线与证据地图.md'):
                path, sep, anchor = str(BM/HANDOVER), '#', 'evidence-history'
            return '](' + quote(os.path.relpath(path, parent), safe='/') + (sep+anchor if sep else '') + ')'
        return match[0]
    text = re.sub(r'\]\(([^)]+)\)', target, text)
    text = text.replace('[[研究主线与证据地图]]', '[[第三部分 研究接手与历史资料]]')
    text = text.replace('[[Startup Router]]', '[[研究交接总览]]')
    text = text.replace(str(ROOT)+'/', '').replace(str(ROOT), '项目仓库根目录')
    text = text.replace('先按Startup Router指定位置记录', '先按本页的接手与维护规则记录')
    text = text.replace('接手的助手先读Startup Router。', '接手时先读本总览。')
    text = text.replace('数据模型清单与证据地图', '数据模型清单与历史证据补充')
    return text


def finalize(documents):
    """Keep the reviewed startup rules and evidence appendix in rebuilt drafts."""
    for doc in documents:
        identifier = doc['identifier']
        text = doc['content']
        if identifier == MAIN:
            current = (BM / MAIN).read_text()
            rules = current.split('<a id="start"></a>', 1)[1].split('<a id="library"></a>', 1)[0]
            text = text.replace('<a id="library"></a>', '<a id="start"></a>'+rules+'<a id="library"></a>')
            text = re.sub(r'以及\[Startup Router\]\([^)]+\)与', '以及', text)
            text = text.replace('- [整个文档库怎样使用](#library)', '- [接手与维护规则](#start)\n- [整个文档库怎样使用](#library)')
        elif identifier == HANDOVER:
            current = (BM / HANDOVER).read_text()
            appendix = current.split('<a id="evidence-history"></a>', 1)[1].split('\n## Relations', 1)[0]
            at = text.find('\n## Relations')
            extra = '\n\n<a id="evidence-history"></a>'+appendix
            text = text[:at]+extra+text[at:] if at >= 0 else text+extra
        elif identifier == 'README.md':
            text = re.sub(r'首次接手.+?。原始报告', '首次接手先读该总览，再按问题打开对应模块。原始报告', text)
        doc['content'] = portable(text, identifier)
    return documents
