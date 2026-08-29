# Running the resume-safe pipeline

Install dependencies once:

```powershell
py -m pip install -r requirements.txt
```

For the first run—and every normal restart—run:

```powershell
py .\analyze_semantic_saturation.py --corpus "C:\Users\anshi\.convokit\saved-corpora\reddit-corpus-small" --output results
```

The script selects CUDA automatically when an NVIDIA CUDA-capable GPU is available; otherwise it reports and uses CPU. Its startup output always includes `Selected device: CUDA` or `Selected device: CPU`. Override it with `--device cpu` or `--device cuda`.

Embeddings are saved in `results\checkpoints\embeddings.npy`. `results\checkpoints\embedding_manifest.json` records the exact embedding-row, clean-comment-row, and utterance-ID mapping plus completed chunks. A completed chunk is never recomputed during a normal restart.

Stop safely with Ctrl+C. At most the currently embedding chunk needs to be redone. Thread results are committed incrementally to `results\thread_metrics.csv` and `results\thread_curves.csv`, with completed IDs in `results\checkpoints\thread_progress.csv`; restarting proceeds with the next unfinished thread.

For a smaller machine, reduce the batch or checkpoint chunk size:

```powershell
py .\analyze_semantic_saturation.py --corpus "C:\Users\anshi\.convokit\saved-corpora\reddit-corpus-small" --output results --batch-size 16 --chunk-size 1000
```

To intentionally discard this output directory's saved checkpoints and recompute, add `--force-recompute`:

```powershell
py .\analyze_semantic_saturation.py --corpus "C:\Users\anshi\.convokit\saved-corpora\reddit-corpus-small" --output results --force-recompute
```
