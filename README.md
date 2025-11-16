### FILE: README.md
## Tiny Legal LoRA Project

1. Create and activate a virtual environment:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   ```
2. Install Python dependencies (use the official CPU PyTorch wheel if pip cannot find one automatically):
   ```
   pip install -r requirements.txt
   ```
3. Populate `../data` with legal documents (`.txt`, `.md`, `.pdf`, `.csv`, `.json`, `.jsonl`).  
4. Build the raw corpus + prompt/completion dataset:
   ```
   python scripts/build_jsonl_from_data.py
   ```
5. Fine-tune adapters on CPU (slow but reproducible):
   ```
   python scripts/tune.py
   ```
6. Launch the interactive REPL with the adapted model:
   ```
   python scripts/model.py
   ```
7. Start the REST backend (loads the adapters once and serves `/api/ask`):
   ```
   python backend/api.py
   ```
8. Point your existing frontend (e.g., the React app in `frontend/`) at `http://localhost:5000/api/ask` to issue questions.

CPU-only runs on older MacBook Pros will be slow and memory-constrained; switch to a GPU-backed machine if you need quicker iterations or larger base models.
