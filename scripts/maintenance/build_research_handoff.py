"""Inventory research documents without inferring experimental use from mentions.

Writes review drafts outside Basic Memory; publish Markdown with BM tools.
Only paths, hashes, explicit metadata and entity hit locations are collected.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote
from portable_handoff import portable

ROOT = Path(__file__).resolve().parents[2]
BM = ROOT / 'basic-memory'
SOURCES = ['docs/experiments', 'Experiments', 'Research', 'Tasks', 'codex/tts-exp']
MODULES = {
    'M1': ('原生效应与跨模型复现', '哪些数据、TTS 和生成模型下存在原生评分优势？'),
    'M2': ('评分协议与多评分器有效性', '本旧分类用于查阅SyncNet评分偏置调查、其他评分器比较及评分流程核查。项目总目的仍是解释观察到的TTS较高Sync-C；评分偏置是重要待检验解释，尚非结论。新版对应模块六，先读研究交接总览。'),
    'M3': ('时间组织与音轨迁移', '时长、停顿、局部时钟和不同发声的匹配如何影响结果？'),
    'M4': ('电平频谱与生成端机制', '固定评分音轨时，电平、谱形和声学干预产生什么响应？'),
    'M5': ('编码器与音素表征', '输入前处理、编码方向、尺度和音素表征如何变化？'),
    'M6': ('训练效应与增强方法', '上述效应能否迁移到训练或形成可靠的自然音频增强？'),
    'M7': ('研究设计历史与恢复', '哪些方案尚未执行，哪些停止边界和恢复凭据必须保留？'),
}
ENTITIES = {
    'datasets': {
        'AISHELL-1': r'AISHELL|aishell|BAC009', 'Chinese-LiPS': r'Chinese[-_ ]?LiPS|chinese_lips|lips_full104|lips21',
        'HDTF': r'HDTF|hdtf', 'EARS': r'EARS|ears40', 'LRS3': r'LRS3|lrs3',
        'LibriSpeech': r'LibriSpeech|librispeech', 'MEAD': r'\bMEAD\b|\bmead\b',
        'VFHQ': r'VFHQ|vfhq', 'TalkVid': r'TalkVid|talkvid', 'GRID': r'\bGRID\b|grid16|geometry8cal',
        'MDC/Effect AI': r'\bMDC\b|mdc_tts|Effect AI', 'RAMC': r'RAMC|ramc',
        'AliMeeting': r'AliMeeting|alimeeting', 'TED-LIUM3': r'TED[-_ ]?LIUM|tedlium',
        'SyncTalk identities': r'\bMay\b|\bPatrick\b|\bChris\b|\bCathy\b',
        'CSS10/CML-TTS': r'CSS10|CML[-_]?TTS',
    },
    'tfg': {n: p for n, p in [
        ('Ditto',r'Ditto|ditto'),('Wav2Lip',r'Wav2Lip|wav2lip|\bW2L\b'),('MuseTalk',r'MuseTalk|musetalk'),
        ('TalkLikeYou',r'TalkLikeYou|talklikeyou'),('PC-Talk',r'PC[-_ ]?Talk|pctalk'),('DyStream',r'DyStream|dystream'),
        ('FlashHead',r'FlashHead|flashhead'),('EchoMimicV3',r'EchoMimic\s?V3|echomimic[-_]v3'),
        ('EchoMimicV2',r'EchoMimic\s?V2|echomimic[-_]v2'),('IMTalker',r'IMTalker|imtalker'),
        ('V-Express',r'V[-_]?Express'),('JoyVASA',r'JoyVASA|joyvasa'),('LatentSync',r'LatentSync|latentsync'),
        ('Hallo2',r'Hallo2|hallo2'),('Hallo3',r'Hallo3|hallo3'),('LeapTalk',r'LeapTalk|leaptalk'),
        ('AVTR-1',r'AVTR[-_]1'),('AvatarForcing',r'AvatarForcing|avatarforcing'),('StableAvatar',r'StableAvatar|stableavatar'),
        ('Seedance',r'Seedance|seedance'),('SyncTalk',r'SyncTalk|synctalk'),('InfiniteTalk',r'InfiniteTalk|infinitetalk'),
        ('MultiTalk',r'MultiTalk|multitalk'),('Sonic',r'\bSonic\b'),('AptAvatar',r'AptAvatar'),('LipForcing',r'LipForcing'),
    ]},
    'tts': {
        'Qwen local/faster_qwen3': r'faster_qwen3|Qwen3[-_ ]?TTS|Qwen3[-_ ]?VC|Qwen.*0\.6B|Qwen.*1\.7B',
        'Qwen cloud Audio 3.1': r'qwen[-_ ]audio|Qwen Audio|云端.*Qwen|云Qwen|云.*Plus|Qwen.*3\.1',
        'IndexTTS2': r'IndexTTS[-_ ]?2|index[-_]tts[-_]2', 'GPT-SoVITS': r'GPT[-_ ]?SoVITS|\bGSV\b',
        'F5-TTS': r'F5[-_ ]?TTS|F5TTS', 'CosyVoice': r'CosyVoice|\bCosy\b',
        'Fish Audio': r'Fish[-_ ]?Audio|fish_audio', 'Higgs': r'Higgs|higgs',
        'ChatTTS': r'ChatTTS|chattts', 'SiliconFlow': r'SiliconFlow|siliconflow',
    },
    'scorers': {n: p for n,p in [
        ('SyncNet',r'SyncNet|syncnet|Sync[-_ ]?C'),('CoGenAV',r'CoGenAV|cogenav'),
        ('PEAVS',r'PEAVS|peavs'),('SyncReward',r'SyncReward|syncreward'),('AVS_u',r'AVS_u|AVS_u|AVS u'),
        ('MTDVocaLiST',r'MTDVocaLiST|MTD K5|\bMTD\b'),('VocaLiST',r'(?<!MTD)VocaLiST|vocalist'),
        ('Synchformer',r'Synchformer|synchformer'),('SparseSync',r'SparseSync|sparsesync'),
    ]},
    'encoders_tools': {n: p for n,p in [
        ('HuBERT',r'HuBERT|hubert'),('XLS-R',r'XLS[-_]?R'),('wav2vec2',r'wav2vec2|Wav2Vec2'),
        ('WavLM',r'WavLM|wavlm'),('Whisper',r'Whisper|whisper'),('MFA',r'\bMFA\b|mfa_'),
        ('WORLD',r'WORLD|Harvest'),('HiFi-GAN',r'HiFi[-_]?GAN'),('DAC',r'\bDAC\b'),
        ('W2vBERT',r'W2vBERT|w2v[-_]?bert'),('LivePortrait',r'LivePortrait|liveportrait'),
    ]},
}


def module_for(path, title):
    m = re.match(r'(\d+)-', path.name)
    if path.parent.name == 'experiments' and m:
        n = int(m[1])
        if path.name == '25-tts-tfg-mechanism.md': return 'M4'
        if n in [1,2,4,5,8,15,75,76,77,79,80,83,84]: return 'M1'
        if n in [6,10,17]: return 'M7'
        if n in [9,12,13,14,19,20,24,49,52,53,57,58,59,60,61,62,63,65]: return 'M5'
        if n in [16,31,32,33,34,35,36,37,44,45]: return 'M3'
        if n in [43,51,68,69,70,71,72,73,74,78,81,82]: return 'M2'
        if n in [21,22,23,25,26,27,28,29,30,50,54,55,56,64,66,67]: return 'M6'
        return 'M4'
    for key, pattern in [
        ('M2',r'九评分|评分器|官方.*评分|控制失败|score|scorer|parity|音轨审计|原生四指标|评分边界|协议差异|评分验证|偏好探针|批量掩码|VSR|视觉.*审计|geometry8cal|GRID16eval'),
        ('M6',r'训练|增强头|train|enhance|adapter|decoder|renderer|模型感知|路由|微调|重建质量|cross-attention|控制链路|control direction'),
        ('M3',r'MFA-linear|时钟|时间|重定时|retim|warp|交叉失配|实例匹配|实例错配|重复发音|停顿|节奏|对齐|replacement|Bridge|时滞|reference|ROI|shift|delay|temporal|phone core|interpolation|重合成|变时|间隙|发声中心|语速|同长交叉|音轨|lag|边界机制|视听.*置换'),
        ('M5',r'编码|音素|encoder|phoneme|特征|HuBERT|表征|可分|codec|编解码|DAC|标签关系|错配标签|Qwen双规模'),
        ('M4',r'电平|音量|响度|谱形|频谱|声学|增益|幅度|WORLD|Harvest|F0|音高|pitch|原生因子|构音|包络|source mechanism|来源位移|mel|原型残差|长期谱|native-axis|原生输入三因素|共同域平移|waveform gain'),
        ('M1',r'跨数据集|多语言|原生.*优势|复现|复核|对照|测评|多.*模型|TTS.*比较|native.*comparison|multiset|confirmation|specificity|native mechanism|TTS.*下降|跨生成器'),
    ]:
        if re.search(pattern,title,re.I): return key
    return 'M7'


def build(out):
    out.mkdir(parents=True,exist_ok=True)
    records=[]; checkpoints=[]
    for folder in SOURCES:
        for p in sorted((BM/folder).rglob('*.md')):
            if any(part in {'assets', '交接'} for part in p.relative_to(BM/folder).parts[:-1]):
                continue
            if p.name.startswith('Codex checkpoint'):
                checkpoints.append({'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
                continue
            text=p.read_text(); title=p.stem; meta={}
            if text.startswith('---\n'):
                fm=text.split('---',2)[1]
                for k in ['title','status','type','updated','current_step']:
                    m=re.search(r'^'+k+r':\s*(.+)$',fm,re.M)
                    if m: meta[k]=m[1].strip("'\"")
                title=meta.get('title',title)
            heading=re.search(r'^# (.+)',text,re.M)
            if heading: title=heading[1]
            r={'id':hashlib.sha256(str(p.relative_to(BM)).encode()).hexdigest()[:12],
               'path':str(p.relative_to(ROOT)),'title':title,'source_group':folder,
               'module':module_for(p,title),'metadata':meta,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
               'status_interpretation':'SOURCE_METADATA_ONLY; not independently re-audited',
               'mentions':{},'mention_semantics':'document mentions, NOT proof of experimental use'}
            numbered = re.match(r'(\d+)-', p.name) if folder == 'docs/experiments' else None
            r['report_number'] = int(numbered[1]) if numbered else None
            r['artifact_path_mentions'] = sorted(set(
                x for x in re.findall(r'`([^`\n]*(?:runs/|results/)[^`\n]*)`', text)
                if len(x) < 400 and not re.search(r'key|password|token|secret', x, re.I)
            ))
            for category, patterns in ENTITIES.items():
                r['mentions'][category]={}
                for name, pattern in patterns.items():
                    hits=[i for i,l in enumerate(text.splitlines(),1) if re.search(pattern,l,re.I)]
                    if hits:r['mentions'][category][name]=hits
            records.append(r)
    payload={'schema_version':1,'scope':'Recursive Markdown in five canonical folders; assets, generated handoff notes and legacy imports excluded',
             'not_an_experiment_count':True,'records':records,'checkpoint_provenance':checkpoints,
             'source_counts':dict(collections.Counter(r['source_group'] for r in records)),
             'module_counts':dict(collections.Counter(r['module'] for r in records))}
    (out/'inventory.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for code,(label,question) in MODULES.items():
        rows=[r for r in records if r['module']==code]
        body=f'# {code} {label}\n\n> 这是旧七分类的来源索引，保留供追溯；新版请从 [[研究交接总览]] 按三部分、六个问题阅读。旧M编号不对应新版模块编号，本页词项表不是面向新读者的实验结论。\n\n{question}\n\n自动导航归类，允许跨模块关联；以原报告和总地图的修正链为准。以下 {len(rows)} 条是文档记录，不是独立实验数。数据／模型栏是全文词项命中，可能来自背景、对照、未执行方案或引用，**不能当作实际使用表**；逐条命中行和文件 SHA 见实验台账 inventory.json。正式编号报告的原始结论未改。\n\n'
        for group in SOURCES:
            group_rows=[r for r in rows if r['source_group']==group]
            if not group_rows:continue
            body+=f'## {group}（{len(group_rows)} 条）\n\n|记录入口|源状态字段|提及数据|提及生成／TTS模型|\n|---|---|---|---|\n'
            for r in group_rows:
                link=quote(str(ROOT/r['path']),safe='/')
                title=r['title'].replace('|','／').replace('\n',' ')
                datasets='、'.join(r['mentions']['datasets']) or '未识别，查原文'
                models='、'.join([*r['mentions']['tfg'],*r['mentions']['tts']]) or '未识别，查原文'
                status=r['metadata'].get('status','未设字段；查原文')
                body+=f'|[{title}]({link})|{status}|{datasets}|{models}|\n'
            body+='\n'
        body+='## Relations\n\n- part_of [[研究交接总览]]\n- relates_to [[研究主线与证据地图]]\n'
        (out/f'{code}.md').write_text(portable(body, f'Research/交接/{code}.md'))
    print(json.dumps({'records':len(records),'checkpoint_provenance':len(checkpoints),
                     'source_counts':payload['source_counts'],'module_counts':payload['module_counts'],
                     'module_bytes':{c:(out/f'{c}.md').stat().st_size for c in MODULES}},ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    build(ap.parse_args().output)
