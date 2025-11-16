### FILE: scripts/build_jsonl_from_data.py
"""
Standalone ingestion utility.

It scans ../data (relative to this script) for readable files, converts them
into a consolidated plain-text corpus, and emits prompt/completion style JSONL
records ready for tiny-language-model fine-tuning.

Usage:
    python scripts/build_jsonl_from_data.py
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PREFERENCE = [
    (PROJECT_ROOT.parent / "data").resolve(),  # ../data from project root
    (PROJECT_ROOT / "data").resolve(),         # fallback to repo-local data
]
OUTPUT_DIR = PROJECT_ROOT / "data"
RAW_CORPUS_PATH = OUTPUT_DIR / "raw_corpus.txt"
TRAIN_JSONL_PATH = OUTPUT_DIR / "QUAD_v1.jsonl"

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".csv", ".jsonl", ".json"}


def find_source_dir() -> Path:
    """Pick the first existing ../data-style directory."""
    for candidate in SOURCE_PREFERENCE:
        if candidate.exists():
            return candidate
    # If nothing exists yet, still return the canonical ../data path.
    default = SOURCE_PREFERENCE[0]
    default.mkdir(parents=True, exist_ok=True)
    return default


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] Failed to read {path}: {exc}")
        return ""


def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] Failed to parse PDF {path}: {exc}")
        return ""


def read_csv(path: Path) -> str:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[warn] Failed to parse CSV {path}: {exc}")
        return ""
    rows = []
    for _, row in df.iterrows():
        values = [str(val) for val in row.values if pd.notna(val)]
        if values:
            rows.append(" ".join(values))
    return "\n\n".join(rows)


def normalize_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return paragraphs


def detect_squad_articles(payload) -> Optional[List[Dict]]:
    """Return a list of SQuAD-like articles if the payload matches that schema."""

    def is_article(obj) -> bool:
        return isinstance(obj, dict) and isinstance(obj.get("paragraphs"), list)

    if isinstance(payload, dict):
        data_block = payload.get("data")
        if isinstance(data_block, list) and data_block and is_article(data_block[0]):
            return data_block
        for value in payload.values():
            if isinstance(value, list) and value and is_article(value[0]):
                return value
    elif isinstance(payload, list) and payload and is_article(payload[0]):
        return payload
    return None


def squad_pairs_from_articles(articles: List[Dict]) -> List[Dict[str, str]]:
    """Convert QUAD/CUAD style JSON into prompt/completion pairs."""
    pairs: List[Dict[str, str]] = []

    for article in articles:
        title = str(article.get("title") or "").strip()
        for paragraph in article.get("paragraphs") or []:
            context = str(paragraph.get("context") or "").strip()
            if not context:
                continue
            for qa in paragraph.get("qas") or []:
                if qa.get("is_impossible"):
                    continue
                question = str(qa.get("question") or "").strip()
                if not question:
                    continue
                answers = qa.get("answers") or []
                answer_text = ""
                for answer in answers:
                    candidate = str(answer.get("text") or "").strip()
                    if candidate:
                        answer_text = candidate
                        break
                if not answer_text:
                    continue
                prompt_lines = [
                    "You have been trained on a legal corpus. Use the provided passage to answer the question.",
                ]
                if title:
                    prompt_lines.append(f"Document: {title}")
                prompt_lines.extend(
                    [
                        f"Question: {question}",
                        f"Passage:\n{context}",
                        "Answer:",
                    ]
                )
                prompt = "\n\n".join(prompt_lines)
                pairs.append({"prompt": prompt, "completion": " " + answer_text})
    return pairs


def parse_jsonlike_block(raw: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (paragraphs, prompt_completion_pairs) from a single JSON/JSONL chunk."""
    paragraphs: List[str] = []
    pairs: List[Dict[str, str]] = []
    data = json.loads(raw)

    squad_articles = detect_squad_articles(data)
    if squad_articles:
        pairs.extend(squad_pairs_from_articles(squad_articles))
        return paragraphs, pairs

    if isinstance(data, list):
        iterable: Iterable = data
    else:
        iterable = [data]
    for item in iterable:
        if isinstance(item, dict):
            prompt = item.get("prompt")
            completion = item.get("completion")
            if prompt is not None and completion is not None:
                pairs.append({"prompt": str(prompt), "completion": str(completion)})
            else:
                values = " ".join(str(v) for v in item.values())
                paragraphs.extend(normalize_paragraphs(values))
        else:
            paragraphs.extend(normalize_paragraphs(str(item)))
    return paragraphs, pairs


def collect_corpus(source_dir: Path) -> Tuple[List[str], List[Dict[str, str]], Counter]:
    paragraphs: List[str] = []
    prompt_completion_pairs: List[Dict[str, str]] = []
    counts: Counter = Counter()

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        counts["files"] += 1
        text_blob = ""

        if suffix in {".txt", ".md"}:
            text_blob = safe_read_text(path)
        elif suffix == ".pdf":
            text_blob = read_pdf(path)
        elif suffix == ".csv":
            text_blob = read_csv(path)
        elif suffix in {".json", ".jsonl"}:
            raw_lines = (
                path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if suffix == ".jsonl"
                else [path.read_text(encoding="utf-8", errors="ignore")]
            )
            for line in raw_lines:
                if not line.strip():
                    continue
                try:
                    extra_paragraphs, pc_pairs = parse_jsonlike_block(line)
                except json.JSONDecodeError:
                    extra_paragraphs = normalize_paragraphs(line)
                    pc_pairs = []
                paragraphs.extend(extra_paragraphs)
                prompt_completion_pairs.extend(pc_pairs)
            counts[f"{suffix}_lines"] += len(raw_lines)
            continue  # already handled per-line ingestion

        if text_blob:
            new_paragraphs = normalize_paragraphs(text_blob)
            paragraphs.extend(new_paragraphs)
            counts[f"{suffix}_paragraphs"] += len(new_paragraphs)

    return paragraphs, prompt_completion_pairs, counts


def build_prompt_completion(paragraph: str) -> Dict[str, str]:
    prompt = (
        "You have been trained on a legal corpus. Use the provided passage to "
        "answer questions or restate key clauses.\n\n"
        f"Passage:\n{paragraph}\n\nAnswer:"
    )
    completion = " " + paragraph
    return {"prompt": prompt, "completion": completion}


def write_raw_corpus(paragraphs: List[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RAW_CORPUS_PATH.open("w", encoding="utf-8") as handle:
        for para in paragraphs:
            handle.write(para.strip() + "\n\n")


def write_train_jsonl(pairs: List[Dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN_JSONL_PATH.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


def build_training_files(force: bool = True) -> Dict[str, int]:
    source_dir = find_source_dir()
    print(f"[info] Reading source corpus from {source_dir}")
    paragraphs, prompt_completion_pairs, stats = collect_corpus(source_dir)

    if not paragraphs and not prompt_completion_pairs:
        print("[warn] Source corpus is empty. Add .txt/.md/.pdf/.csv/.json/.jsonl files to ../data.")
        return {"paragraphs": 0, "jsonl_examples": 0}

    if paragraphs:
        write_raw_corpus(paragraphs)
        print(f"[info] Wrote raw corpus ({len(paragraphs)} paragraphs) -> {RAW_CORPUS_PATH}")

    training_pairs = list(prompt_completion_pairs)
    if not training_pairs:
        training_pairs = [build_prompt_completion(p) for p in paragraphs]

    if not training_pairs:
        print("[warn] No data available to populate train.jsonl")
        return {"paragraphs": len(paragraphs), "jsonl_examples": 0}

    write_train_jsonl(training_pairs)
    print(f"[info] Wrote {len(training_pairs)} prompt/completion pairs -> {TRAIN_JSONL_PATH}")
    print(f"[info] Processed file stats: {dict(stats)}")
    return {"paragraphs": len(paragraphs), "jsonl_examples": len(training_pairs)}


def main() -> None:
    summary = build_training_files(force=True)
    if summary["paragraphs"] == 0 and summary["jsonl_examples"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
