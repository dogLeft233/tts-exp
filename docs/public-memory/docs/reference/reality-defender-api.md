---
title: Reality Defender API
type: note
permalink: tts-exp/docs/reference/reality-defender-api
---

# Reality Defender API 官方文档调查

> 调查日期：2026-09-09
> 来源范围：仅使用 Reality Defender 官方文档站点（`docs.realitydefender.com`）及其文档索引。
> 安全边界：本文没有调用真实 API，没有上传文件，也没有记录任何 API key。

## 结论摘要

| 关注点 | 官方文档结论 |
|---|---|
| 主要检测对象 | 本地图片、音频、视频、文本，以及 Facebook、Instagram、Twitter/X、YouTube、TikTok、Threads 社交媒体 URL。SDK 和 REST API 都支持这些入口。[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart)、[Social Media URL Upload](https://docs.realitydefender.com/api-reference/endpoint/social) |
| 调用模型 | REST API；官方更推荐大多数集成使用 SDK。媒体文件走“申请预签名 URL → PUT 二进制文件 → 用 `requestId` 取结果”，社交 URL 可直接 POST。[Introduction](https://docs.realitydefender.com/introduction)、[API Quickstart](https://docs.realitydefender.com/api-reference/quickstart) |
| 处理模式 | 结果可能异步生成：需要轮询，或使用 webhook；SDK 还提供事件式/轮询式以及同步/异步编程模型。[API Quickstart](https://docs.realitydefender.com/api-reference/quickstart)、[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart) |
| 文件限制 | 文本 900 KB、图片 50 MB、音频 20 MB、视频 250 MB；扩展名和文件名规则见下文。[Presigned URL](https://docs.realitydefender.com/api-reference/endpoint/presigned) |
| 本地硬件 | 当前公开文档没有提出本地 GPU、显存或 RAM 最低要求，也没有要求下载/运行本地检测模型；文档只描述通过 API/SDK 提交到平台。此处是“文档未说明”的结论，不代表所有私有部署形态都没有额外要求。[Introduction](https://docs.realitydefender.com/introduction)、[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart) |
| 速率/配额 | 当前公开文档没有给出 requests/min、并发上限、每日额度或配额查询接口。SDK 提到的“可配置并发限制”是客户端能力，不应当当作服务端配额。免费层明确限制的是可上传媒体类型：仅音频和图片。[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart) |
| 最小客户端 | 原始 REST 测试可用 `curl`；官方 Python 示例使用 `requests`。文档没有在门户页给出 SDK 包名/版本；SDK 安装细节被留给各语言 README。[API Quickstart](https://docs.realitydefender.com/api-reference/quickstart)、[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart) |

## 1. 主要功能与检测类型

### 1.1 检测入口

| 类型 | 官方列出的格式/入口 | 主要结果能力 |
|---|---|---|
| 图片 | `.jpg`、`.jpeg`、`.png`、`.gif`、`.webp` | 图片检测；人工/伪造模型可提供 heatmap 预签名 PNG URL。 |
| 音频 | `.mp3`、`.wav`、`.m4a`、`.aac`、`.ogg`、`.flac`、`.alac` | 音频检测；返回模型结果、语言/音频分块等聚合数据。 |
| 视频 | `.mp4`、`.mov` | 视频检测；可有缩略图、场景/时间线等聚合数据；文档还描述了视频中的音频结果字段。 |
| 文本 | `.txt` | 文本检测；可返回 HTML explainability 预签名 URL。 |
| 社交媒体 URL | Facebook、Instagram、Twitter/X、YouTube、TikTok、Threads | 提交链接后异步分析；返回 `requestId`，再按同一结果接口查询。 |

格式列表来自 [SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart) 和 [AWS Presigned URL](https://docs.realitydefender.com/api-reference/endpoint/presigned)；社交平台列表来自 [Social Media URL Upload](https://docs.realitydefender.com/api-reference/endpoint/social) 和 [API Quickstart](https://docs.realitydefender.com/api-reference/quickstart)。

### 文档首页列出的产品形态

首页除 SDK 和 Platform REST API 外，还列出 Web Conferencing、Contact Center、Access Security & Fraud Prevention、Brand Safety 等产品入口；本文只调查可直接从公开 API/SDK 文档确认的媒体检测集成，不对这些解决方案页面之外的部署细节做推断。[Introduction](https://docs.realitydefender.com/introduction)

### 1.2 结果与模型

- 典型总体状态包括 `AUTHENTIC`、`FAKE`、`SUSPICIOUS`、`NOT_APPLICABLE`、`UNABLE_TO_EVALUATE`；处理中可能是 `ANALYZING`。[Media Detail](https://docs.realitydefender.com/api-reference/endpoint/get_media_detail)
- 响应包含 `resultsSummary` 和 `models[]`，可看到每个模型的状态、分数以及模型特定数据。官方明确建议以 ensemble 结果为主要判断依据，不要把单个模型结果当作唯一事实。[API Quickstart](https://docs.realitydefender.com/api-reference/quickstart)、[All Media](https://docs.realitydefender.com/api-reference/endpoint/get_all_media)
- 文档给出的模型名只是示例，而且会随版本、计划、机构设置以及模型增删而变化；客户端不应硬编码模型 slug。[Media Detail](https://docs.realitydefender.com/api-reference/endpoint/get_media_detail)
- 对图片，非 ensemble 且判定为人工/伪造的模型可能有 heatmap；视频、音频和文本的 `heatmaps` 通常为 `null`。heatmap、缩略图、原文件、聚合元数据等预签名 URL 的有效期为 15 分钟，失效后应重新获取 media detail。[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart)、[Media Detail](https://docs.realitydefender.com/api-reference/endpoint/get_media_detail)
- 可选的用户反馈接口允许针对已完成结果提交标签（`REAL`、`SYNTHETIC`、`MANIPULATED`、`UNKNOWN`）和反馈类别（`FALSE_POSITIVE`、`FALSE_NEGATIVE`、`CONFIRMATION`、`OTHER`）。[Create User Feedback](https://docs.realitydefender.com/api-reference/endpoint/create_user_feedback)

### 1.3 “不可适用”不等于接口错误

文档将某些输入特征单独标为 `NOT_APPLICABLE`，例如：

- 图片没有检测到人脸，或人脸太小；
- 音频短于 1.5 秒；
- 音频包含拨号音/音乐、检测到多个说话人、噪声过大，或语言不适合当前处理；
- 文件处理超时或发生处理错误时，可能是 `UNABLE_TO_EVALUATE`，文档建议重试或上传更小文件。

这些是内容适用性/处理结果，不是对扩展名或字节大小限制的替代说明。[Media Detail](https://docs.realitydefender.com/api-reference/endpoint/get_media_detail)

## 2. API 调用方式

API 示例使用的服务基址是 `https://api.prd.realitydefender.xyz`。每个受保护 API 请求都要在 `X-API-KEY` header 中提供 key；本文只记录 header 名称，不记录 key 值。[API Quickstart](https://docs.realitydefender.com/api-reference/quickstart)

### 2.1 本地文件：预签名 URL 流程

| 步骤 | 方法与路径 | 请求/结果 |
|---|---|---|
| 1. 申请上传地址 | `POST /api/files/aws-presigned` | JSON 参数为 `fileName`（含扩展名），并带 `X-API-KEY`；响应提供 `signedUrl`，流程同时使用返回的 `requestId`。 |
| 2. 上传文件 | `PUT <signedUrl>` | 将文件原始二进制直接写入预签名 URL；官方示例说明不需要额外 headers 或 metadata。 |
| 3. 取检测结果 | `GET /api/media/users/{requestId}` | 带 `X-API-KEY` 查询具体结果；在状态不再是 `ANALYZING` 后读取 `resultsSummary`/ensemble 结果。 |

以上是 [API Quickstart](https://docs.realitydefender.com/api-reference/quickstart) 的完整调用顺序；字段和文件规则见 [AWS Presigned URL](https://docs.realitydefender.com/api-reference/endpoint/presigned) 与 [Media Detail](https://docs.realitydefender.com/api-reference/endpoint/get_media_detail)。

### 2.2 社交媒体 URL

- `POST /api/files/social`，JSON 参数为 `socialLink`，带 `X-API-KEY`；成功后取得 `requestId`。
- 使用同一个 `GET /api/media/users/{requestId}` 查询结果。
- 结果可能包含 `socialLinkDownloaded` 和 `socialLinkDownloadFailed`，因此客户端应处理下载失败状态。

详见 [Social Media URL Upload](https://docs.realitydefender.com/api-reference/endpoint/social) 和 [Media Detail](https://docs.realitydefender.com/api-reference/endpoint/get_media_detail)。

### 2.3 其他 API 能力

- `GET /api/v2/media/users`：查看当前用户提交的文件和社交 URL；还支持按文件名、日期和分页筛选。此端点默认针对用户自己的上传，不等于组织级全量列表。[All Media](https://docs.realitydefender.com/api-reference/endpoint/get_all_media)
- `POST /api/v2/user-feedback`：对已完成结果提交用户反馈；要求 `requestId`、标签和反馈类别，具有访问控制。[Create User Feedback](https://docs.realitydefender.com/api-reference/endpoint/create_user_feedback)

### 2.4 SDK

官方 SDK 页面列出 TypeScript/JavaScript、Python、Go、Rust、Java 五种实现。每种 SDK 都遵循“提交本地文件或社交 URL → 得到 `requestId` → 获取分析结果”的流程，并支持事件式或轮询式结果处理；SDK 页面没有给出统一的安装命令、包版本或跨语言依赖清单，而是指向各语言 README。[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart)

## 3. 输入文件、请求与格式限制

### 3.1 本地文件限制

| 输入 | 扩展名 | 单文件上限 |
|---|---|---:|
| 文本 | `.txt` | 900 KB |
| 图片 | `.jpg`、`.jpeg`、`.png`、`.gif`、`.webp` | 50 MB |
| 音频 | `.mp3`、`.wav`、`.m4a`、`.aac`、`.ogg`、`.flac`、`.alac` | 20 MB |
| 视频 | `.mp4`、`.mov` | 250 MB |

表中上限和扩展名来自 [AWS Presigned URL](https://docs.realitydefender.com/api-reference/endpoint/presigned)；SDK 页面给出同一组限制，并额外说明免费层只支持音频和图片上传。[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart)

### 3.2 文件名与 URL

- 申请预签名 URL 时必须在 `fileName` 中包含扩展名。
- 文件名最长 200 个字符；特殊字符会被转换为下划线；大小写不敏感。[AWS Presigned URL](https://docs.realitydefender.com/api-reference/endpoint/presigned)
- 预签名 URL、heatmap、缩略图、原文件位置和聚合元数据 URL 有 15 分钟有效期；过期应重新 GET media detail，而不是缓存旧 URL。[Media Detail](https://docs.realitydefender.com/api-reference/endpoint/get_media_detail)

### 3.3 请求体形式

- 申请预签名 URL、提交社交 URL、查询反馈：JSON 请求体，`Content-Type: application/json`，并按接口要求带 API key。
- 上传到预签名 URL：`PUT` 原始文件字节；官方明确说不需要再附加 headers 或 metadata。
- 本调查未执行这些请求；示例中的 key 均被刻意省略。

## 4. 是否需要本地 GPU 或大内存

从官方公开页面能确认的是：平台通过 REST API/SDK 接收文件或 URL，客户端负责 HTTP 通信、上传以及获取结果；文档没有要求安装本地模型、配置 CUDA/GPU、指定显存或 RAM。[Introduction](https://docs.realitydefender.com/introduction)、[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart)

因此，做最小化客户端测试时，主要依赖是网络、HTTP 客户端和运行时注入的 API key，而不是本地推理硬件。这里的结论严格限于当前公开文档：文档没有覆盖的私有化/特殊部署方案，不能从本调查推导硬件需求。

## 5. 速率、配额与异步要求

### 5.1 速率与配额

在官方文档索引列出的 Introduction、SDK Quickstart、API Quickstart 和端点页面中，没有找到数值化的请求速率、每分钟/每日额度、单 key 并发上限、重试-after 规则或配额查询 API。官方明确写出的相关限制只有：

- SDK 可以配置客户端并发限制；
- 免费层仅支持音频和图片上传；
- 各媒体类型的单文件大小上限见上表。

不要把 SDK 的可配置 concurrency 当作 Reality Defender 服务端承诺的配额；如果上线前需要具体限流或套餐额度，应向 Reality Defender 确认。[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart)、[官方文档索引](https://docs.realitydefender.com/llms.txt)

### 5.2 异步/轮询

- API Quickstart 警告较大文件可能需要更长分析时间；客户端需要持续轮询结果，或设置 webhook。
- SDK 页面支持 event-based 或 polling 处理，并支持同步/异步编程模型。
- 公开文档列出了“可以使用 webhook”，但当前 API 文档索引没有给出 webhook URL、签名、重试策略或事件 schema；这些细节不应自行猜测。

因此，REST 最小客户端应把上传/提交和取结果视为两个阶段，能处理 `ANALYZING`，并对超时/`UNABLE_TO_EVALUATE` 留出重试路径。[API Quickstart](https://docs.realitydefender.com/api-reference/quickstart)、[Media Detail](https://docs.realitydefender.com/api-reference/endpoint/get_media_detail)、[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart)

## 6. 最小化测试所需客户端依赖

### 6.1 最小 REST 测试

按官方示例，最小客户端可以是：

1. `curl`：无需安装 Reality Defender SDK；具备 HTTPS 网络访问，并在运行时提供 API key。
2. Python：使用官方示例中的 `requests`；读文件使用 Python 标准库即可。官方文档没有指定 `requests` 版本。

实际测试还需要一份符合上文扩展名/大小限制的本地样本，或一个受支持的社交媒体 URL；本文没有上传样本或调用接口。参考 [API Quickstart](https://docs.realitydefender.com/api-reference/quickstart) 的 curl/Python 示例和 [AWS Presigned URL](https://docs.realitydefender.com/api-reference/endpoint/presigned) 的 PUT 说明。

### 6.2 使用官方 SDK

可选语言为 TypeScript/JavaScript、Python、Go、Rust、Java。官方门户只说明“按语言 README 安装和使用”，没有在页面上固定包名或版本，因此在没有进一步读取对应官方仓库 README 前，不应在项目依赖文件中臆填版本。[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart)

对于一次性 smoke test，REST + `curl` 是文档中依赖最少的路径；对于生产集成或批量并发，官方倾向使用 SDK，并利用其资源管理、并发以及事件/轮询封装。[Introduction](https://docs.realitydefender.com/introduction)、[SDK Quickstart](https://docs.realitydefender.com/sdks/quickstart)

## 官方来源

- [Introduction](https://docs.realitydefender.com/introduction)
- [Reality Defender SDK](https://docs.realitydefender.com/sdks/quickstart)
- [API Quickstart](https://docs.realitydefender.com/api-reference/quickstart)
- [AWS Presigned URL](https://docs.realitydefender.com/api-reference/endpoint/presigned)
- [Social Media URL Upload](https://docs.realitydefender.com/api-reference/endpoint/social)
- [Media Detail](https://docs.realitydefender.com/api-reference/endpoint/get_media_detail)
- [All Media](https://docs.realitydefender.com/api-reference/endpoint/get_all_media)
- [Create User Feedback](https://docs.realitydefender.com/api-reference/endpoint/create_user_feedback)
- [官方文档索引](https://docs.realitydefender.com/llms.txt)
