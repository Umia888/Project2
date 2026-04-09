"""
向量切片与 RAG 分块策略（按文件类型选用不同策略）。

知识点总览
----------
1) 向量切片（embedding-based chunking）
   - 将文本切成「检索单元」后，用 text-embedding 模型把每块映射为稠密向量；
   - 检索时可与 query 向量算相似度（本项目中 LLM 读块摘要做排序，语义块本身更内聚，便于命中正确段落）。

2) 递归字符分块（Recursive Character Text Splitting）
   - 思想：按分隔符**优先级**依次尝试切分（如 \\n\\n → \\n → 句号 → 空格），大块再递归细分，直到每块 ≤ chunk_size；
   - chunk_overlap：见文件内常量 RECURSIVE_CHUNK_OVERLAP（与 RECURSIVE_CHUNK_SIZE 配套），相邻块共享一段尾部/首部文本；
   - 适用：**Markdown / 纯文本**、结构以换行与标点为主；**中英混排**合同、说明文；
   - 优点：实现稳、成本低（无额外嵌入调用）；缺点：无法感知「语义突变」边界。

3) 语义分块（Semantic Chunking，本实现为「段落级嵌入 + 相似度合并」）
   - 思想：先按段落切开，对每段求 embedding；相邻段若向量相似（同一话题）则合并为一块，直到达到长度上限；
   - 块与块之间再追加 SEMANTIC_INTER_CHUNK_OVERLAP 字的前块尾部，衔接话题切换边界（见常量注释）；
   - 适用：**DOCX** 等常出现长段叙述、条款编号不总换行的法律稿；合并减少「半句话一块」的噪声；
   - 优点：块内主题更一致；缺点：每文件多轮嵌入 API，成本与时延高于递归分块。

4) 类型路由
   - .md / .txt → 递归字符分块（强调结构与标点边界）；
   - .docx → 语义分块（强调长段内的主题连续性）；
   - 其他扩展名 → 与 txt 相同，递归分块；
   - 模型生成的「类案语料池」等无扩展名逻辑 → 在调用处传入 force_strategy 或默认递归。
"""

from __future__ import annotations

import math
import re
from typing import List

from openai import OpenAI

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _LangChainRecursiveSplitter
except ModuleNotFoundError:  # 未 pip install 时仍可用纯 Python 分块
    _LangChainRecursiveSplitter = None  # type: ignore[misc, assignment]


def _fallback_recursive_split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List[str]:
    """
    与 RecursiveCharacterTextSplitter 行为接近的纯 Python 实现：在窗口内按分隔符优先级从后向前找断点，再带 overlap 滑动。
    不引入 langchain-text-splitters 依赖，避免环境不一致导致 Home.py 无法启动。
    """
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= chunk_size:
        return [t]

    seps = [s for s in separators if s != ""]
    step_min = max(1, chunk_size - chunk_overlap)
    out: List[str] = []
    start = 0
    n = len(t)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = t[start:end]
            cut_rel = -1
            for sep in seps:
                if not sep:
                    continue
                pos = window.rfind(sep)
                if pos >= 0:
                    cand = pos + len(sep)
                    if cand >= max(40, chunk_size // 10) and cand > cut_rel:
                        cut_rel = cand
            if cut_rel > 0:
                end = start + cut_rel
        piece = t[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + step_min
        start = min(next_start, n)

    return out if out else [t[:chunk_size]]

# 与 OpenAI 控制台常见命名一致；小模型成本低，适合分块阶段批量嵌入
_EMBED_MODEL = "text-embedding-3-small"

# --- 切块重叠 overlap（均按 Python 字符 len 计数，中英混排一字一算）---
# RECURSIVE_CHUNK_SIZE：单块目标上限；与 overlap 搭配后单块实际可能略超，由分隔符优先保证不在词/句正中硬切。
RECURSIVE_CHUNK_SIZE = 900
# RECURSIVE_CHUNK_OVERLAP = 180
# 理由：约为 chunk_size 的 20%。合同/法条里「前款」「上述」「该义务」等跨句指代多，边界若落在两句之间，
#       无 overlap 时容易出现两块各缺一半主谓宾；180 字量级约覆盖数句中文或半段英文从句，使相邻块共享「过渡带」，
#       检索命中任一块时仍带上下文脉，减轻断章取义。若 <100 则衔接偏弱；若 >280 则块间重复多、排序 prompt 冗长。
RECURSIVE_CHUNK_OVERLAP = 180

# SEMANTIC_INTER_CHUNK_OVERLAP = 130
# 理由：语义合并块内部已按段落主题聚类，块边界多为话题切换点；再在「下一块」首部拼接「上一块末尾 130 字」，
#       形成与递归分块类似的滑动窗口效果，但略小于 180，因语义块平均更长、避免单块膨胀过快。
#       130 仍足以覆盖跨段衔接语（如「此外」「同时」引导的一句），又不与前块大段重复。
SEMANTIC_INTER_CHUNK_OVERLAP = 130


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _mean_vec(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    n = float(len(vectors))
    return [x / n for x in acc]


def _embed_batch(texts: List[str], api_key: str) -> List[List[float]]:
    """批量嵌入；OpenAI 保证返回与 input 顺序一致（按 index 排序）。"""
    client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]


def recursive_character_chunks(
    text: str,
    filename: str,
    *,
    chunk_size: int = RECURSIVE_CHUNK_SIZE,
    chunk_overlap: int = RECURSIVE_CHUNK_OVERLAP,
) -> List[dict]:
    # 知识点：分隔符顺序体现「先大段、后小段」— 尽量不在词中间断开。
    # overlap：LangChain 版由 splitter 处理；无依赖时用 _fallback_recursive_split_text 滑动窗口（见文件头常量注释）。
    if not (text or "").strip():
        return []
    separators = [
        "\n\n",
        "\n",
        "。",
        "；",
        "．",
        ". ",
        "; ",
        "！",
        "？",
        "! ",
        "? ",
        " ",
        "",
    ]
    _body = text.strip()
    if _LangChainRecursiveSplitter is not None:
        splitter = _LangChainRecursiveSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )
        parts = splitter.split_text(_body)
    else:
        parts = _fallback_recursive_split_text(
            _body,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
    out: List[dict] = []
    for i, chunk in enumerate(parts):
        if not chunk.strip():
            continue
        out.append(
            {
                "filename": filename,
                "text": chunk.strip(),
                "chunk_id": i,
                "strategy": "recursive_character",
            }
        )
    # 全文不足一块时无「邻块」，不造 overlap；仅截取 chunk_size 防极端长单行。
    return out if out else [{"filename": filename, "text": text.strip()[:chunk_size], "chunk_id": 0, "strategy": "recursive_character"}]


def _apply_semantic_inter_chunk_overlap(chunks: List[dict], overlap_chars: int) -> List[dict]:
    # 在语义策略产出的相邻块之间，把「前一块末尾 overlap_chars 字」接到「后一块开头」，与递归分块内置 overlap 目的一致：防断章取义。
    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks
    out: List[dict] = []
    for i, ch in enumerate(chunks):
        t = ch["text"]
        if i > 0:
            prev_txt = chunks[i - 1]["text"]
            tail = prev_txt[-overlap_chars:] if len(prev_txt) >= overlap_chars else prev_txt
            tail = tail.strip()
            if tail and not t.startswith(tail):
                t = f"{tail}\n{t}"
        out.append({**ch, "text": t.strip()})
    return out


def _paragraphs(text: str) -> List[str]:
    raw = re.split(r"\n\s*\n+", (text or "").strip())
    return [p.strip() for p in raw if p.strip()]


def semantic_chunks_by_paragraph_merge(
    text: str,
    filename: str,
    api_key: str,
    *,
    max_chunk_chars: int = 1400,
    similarity_threshold: float = 0.58,
    batch_size: int = 16,
) -> List[dict]:
    # 知识点：相邻段 embedding 与当前块均值向量比相似度 — 高则合并，表示话题连续；低则切块 — 表示语义转折。
    # 若段落极多，成本 ∝ 段数；失败时由上层回退到递归分块。
    paras = _paragraphs(text)
    if not paras:
        return []
    total_len = sum(len(p) for p in paras)
    if len(paras) == 1 or total_len <= max_chunk_chars:
        return [
            {
                "filename": filename,
                "text": "\n\n".join(paras),
                "chunk_id": 0,
                "strategy": "semantic_paragraph_merge",
            }
        ]

    all_embs: List[List[float]] = []
    for i in range(0, len(paras), batch_size):
        batch = paras[i : i + batch_size]
        all_embs.extend(_embed_batch(batch, api_key))

    chunks_text: List[str] = []
    current_parts: List[str] = []
    current_embs: List[List[float]] = []

    for p, emb in zip(paras, all_embs):
        if not current_parts:
            current_parts = [p]
            current_embs = [emb]
            if sum(len(x) for x in current_parts) >= max_chunk_chars:
                chunks_text.append("\n\n".join(current_parts))
                current_parts = []
                current_embs = []
            continue

        merged_len = sum(len(x) for x in current_parts) + len(p) + 2
        mean_e = _mean_vec(current_embs)
        sim = _cosine_similarity(emb, mean_e)

        if merged_len <= max_chunk_chars and sim >= similarity_threshold:
            current_parts.append(p)
            current_embs.append(emb)
        else:
            chunks_text.append("\n\n".join(current_parts))
            current_parts = [p]
            current_embs = [emb]

        if sum(len(x) for x in current_parts) >= max_chunk_chars:
            chunks_text.append("\n\n".join(current_parts))
            current_parts = []
            current_embs = []

    if current_parts:
        chunks_text.append("\n\n".join(current_parts))

    out: List[dict] = []
    for i, c in enumerate(chunks_text):
        c = (c or "").strip()
        if c:
            out.append(
                {
                    "filename": filename,
                    "text": c,
                    "chunk_id": i,
                    "strategy": "semantic_paragraph_merge",
                }
            )
    if not out:
        return recursive_character_chunks(text, filename)
    # 语义块边界处补 SEMANTIC_INTER_CHUNK_OVERLAP 字重叠（见文件头常量注释）；递归分块由 splitter 已处理 overlap，勿重复套一层。
    return _apply_semantic_inter_chunk_overlap(out, SEMANTIC_INTER_CHUNK_OVERLAP)


def choose_chunk_strategy_for_filename(filename: str) -> str:
    """返回 'semantic_paragraph_merge' 或 'recursive_character'。"""
    fn = (filename or "").lower()
    if fn.endswith(".docx"):
        return "semantic_paragraph_merge"
    if fn.endswith((".md", ".txt")):
        return "recursive_character"
    return "recursive_character"


def documents_to_vector_chunks(
    documents: List[dict],
    api_key: str,
    *,
    force_recursive: bool = False,
) -> List[dict]:
    """
    将 [{'filename','text'}, ...] 转为 chunk 列表，每项含 strategy / chunk_id。
    force_recursive=True 时全部用递归分块（如无 Key、或合成语料希望省嵌入）。
    """
    if not documents:
        return []
    if not (api_key or "").strip():
        force_recursive = True

    all_chunks: List[dict] = []
    global_idx = 0

    for doc in documents:
        fn = doc.get("filename") or "unknown"
        body = doc.get("text") or ""
        if not body.strip():
            continue

        strat = choose_chunk_strategy_for_filename(fn) if not force_recursive else "recursive_character"
        try:
            if strat == "semantic_paragraph_merge" and not force_recursive:
                sub = semantic_chunks_by_paragraph_merge(body, fn, api_key)
            else:
                sub = recursive_character_chunks(body, fn)
        except Exception:
            sub = recursive_character_chunks(body, fn)

        for ch in sub:
            row = {
                "filename": ch["filename"],
                "text": ch["text"],
                "chunk_id": global_idx,
                "strategy": ch.get("strategy", "recursive_character"),
                "source_chunk_index": ch.get("chunk_id", 0),
            }
            global_idx += 1
            all_chunks.append(row)

    return all_chunks


def chunk_label(item: dict) -> str:
    """用于 UI / 排序 prompt 展示：文件名 + 策略 + 源内序号。"""
    fn = item.get("filename", "")
    st = item.get("strategy") or "legacy_document"
    sci = item.get("source_chunk_index", item.get("chunk_id", 0))
    return f"{fn} [{st} #{sci}]"
