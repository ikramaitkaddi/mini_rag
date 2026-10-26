# LocalAI_Lab

A local course assistant built with Flask, Ollama, and keyword-based retrieval over the files in `documents/`.

## Requirements

- Python 3.10 or newer
- Ollama running at http://127.0.0.1:11434
- The qwen3.5 model for the app; gemma3 for the comparison
- Python packages from `requirements.txt`

From this directory, install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
cd source
python app.py
```

Open http://127.0.0.1:5001. The server listens on localhost. The app searches the local documents, sends relevant context and per-session conversation history to Ollama, supports structured JSON answers, and displays sources and response time.

The console chatbot and benchmark are also in `source/`:

```powershell
python chatbot.py
python benchmark.py qwen3.5 gemma3
```

## Results and evidence

`report.pdf` contains the English answers to Q1-Q15, the measured two-model comparison, a benchmark table, and the limits of the single-run measurements. Both generated prime-number functions passed tests from -5 to 2000; the report explains their difference on a very large integer.

All six requested PNG captures are in `screenshots/`. The `04_python.png` capture shows some Windows character-encoding artifacts in the model's formatted text, while the Python command and returned answer remain visible.

