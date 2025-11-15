# build_jsonl_from_data.py
import os, json, csv, glob
from pathlib import Path
from pypdf import PdfReader   # pypdf2 or pypdf
import pandas as pd

DATA_DIR = Path("../data")
OUT_JSONL = Path("data/train.jsonl")
OUT_CORPUS = Path("data/raw_corpus.txt")

os.makedirs("data", exist_ok=True)

def read_pdf(path):
    text = []
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            text.append(p.extract_text() or "")
    except Exception as e:
        print("PDF read error:", path, e)
    return "\n".join(text).strip()

def collect_texts():
    texts = []
    for ext in ("*.jsonl", "*.txt", "*.md", "*.pdf", "*.csv"):
        for p in DATA_DIR.glob(ext):
            p = p.resolve()
            try:
                if p.suffix.lower() == ".jsonl":
                    # take each line if it has prompt/completion
                    with open(p, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line=line.strip()
                            if not line: continue
                            try:
                                obj = json.loads(line)
                                # if it's already prompt/completion, write to out jsonl directly
                                if "prompt" in obj and "completion" in obj:
                                    with open(OUT_JSONL, "a", encoding="utf-8") as out:
                                        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                    continue
                            except:
                                pass
                            texts.append(line)
                elif p.suffix.lower() in (".txt", ".md"):
                    with open(p, "r", encoding="utf-8") as fh:
                        texts.append(fh.read())
                elif p.suffix.lower() == ".pdf":
                    texts.append(read_pdf(str(p)))
                elif p.suffix.lower() == ".csv":
                    try:
                        df = pd.read_csv(p)
                        # join textual columns heuristically
                        for _, row in df.iterrows():
                            row_text = " ".join([str(v) for v in row.dropna().values])
                            texts.append(row_text)
                    except Exception as e:
                        print("CSV read failed:", p, e)
            except Exception as e:
                print("Error reading", p, e)
    return [t for t in texts if t and len(t.strip())>20]

if __name__ == "__main__":
    texts = collect_texts()
    print(f"Collected {len(texts)} documents from {DATA_DIR}")
    # write corpus (for RAG)
    with open(OUT_CORPUS, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n\n")
    print("Wrote corpus to", OUT_CORPUS)
    # If OUT_JSONL already has content (from included jsonl), we keep it. Otherwise create a few prompt/completion examples
    if OUT_JSONL.exists():
        print("train.jsonl existed or was appended to — check data/train.jsonl")
    else:
        # create simple QA pairs from each document: first 200 chars -> prompt, rest -> completion (quick heuristic)
        with open(OUT_JSONL, "w", encoding="utf-8") as out:
            for d in texts:
                prompt = "Extract key points from the following document:\n\n" + d[:500] + "\n\n###\n"
                completion = "Key points: " + (d[500:1200].strip() or " (summary omitted) ")
                out.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
        print("Wrote synthetic train.jsonl with", len(texts), "examples")

