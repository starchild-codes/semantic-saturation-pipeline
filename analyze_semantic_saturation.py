"""Resume-safe frozen semantic-saturation analysis pipeline.

The scientific protocol is fixed in FROZEN_PROTOCOL.md. This module adds only
durable checkpoints, progress reporting, and hardware selection.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MIN_COMMENTS = 20
GRID_POINTS = 101
MANIFEST_VERSION = 1


def clean_text(text):
    if text is None:
        return None
    text = str(text).strip()
    return None if not text or text.lower() in {"[deleted]", "[removed]"} else text


def atomic_write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


def atomic_write_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    atomic_replace(temporary, path)


def atomic_replace(temporary, path, attempts=20, retry_seconds=0.25):
    """Retry Windows replace calls briefly when OneDrive has a transient lock."""
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(retry_seconds)


def clean_comment_fingerprint(comments):
    """Fingerprint the exact order represented in the embedding array."""
    digest = hashlib.sha256()
    columns = ["clean_comment_row", "utterance_id", "conversation_id"]
    for row in comments[columns].itertuples(index=False):
        digest.update(f"{row[0]}\t{row[1]}\t{row[2]}\n".encode("utf-8"))
    return digest.hexdigest()


def load_convokit_corpus(corpus_dir):
    """Load comments only; all scientific filtering remains intentionally fixed."""
    from convokit import Corpus

    corpus = Corpus(filename=str(corpus_dir))
    rows = []
    for utt in corpus.iter_utterances():
        if utt.reply_to is None:  # root submissions/posts are excluded
            continue
        text = clean_text(utt.text)
        if text is not None:
            rows.append({
                "conversation_id": utt.conversation_id,
                "utterance_id": utt.id,
                "timestamp": utt.timestamp,
                "text": text,
            })

    comments = pd.DataFrame(rows)
    if comments.empty:
        raise ValueError("No usable comments were found.")
    comments["timestamp"] = pd.to_numeric(comments["timestamp"], errors="coerce")
    comments = comments.dropna(subset=["timestamp"])
    comments = comments.sort_values(
        ["conversation_id", "timestamp", "utterance_id"], kind="mergesort"
    ).reset_index(drop=True)
    comments["clean_comment_row"] = np.arange(len(comments), dtype=np.int64)
    if not comments["utterance_id"].is_unique:
        raise AssertionError("Utterance IDs must uniquely identify clean comment rows.")
    return comments


def eligible_comments(comments):
    sizes = comments.groupby("conversation_id", sort=False).size()
    eligible_ids = sizes.index[sizes >= MIN_COMMENTS]
    eligible = comments[comments["conversation_id"].isin(eligible_ids)].copy()
    eligible = eligible.reset_index(drop=True)
    eligible["embedding_row"] = np.arange(len(eligible), dtype=np.int64)
    counts = eligible.groupby("conversation_id").size()
    assert not counts.empty and (counts >= MIN_COMMENTS).all()
    assert eligible["embedding_row"].is_unique
    return eligible


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return requested


def embedding_manifest(eligible):
    return {
        "version": MANIFEST_VERSION,
        "model": MODEL_NAME,
        "n_embeddings": int(len(eligible)),
        "row_fingerprint": clean_comment_fingerprint(eligible),
        # Explicit mappings make every embedding row independently traceable.
        "rows": [
            {
                "embedding_row": int(row.embedding_row),
                "clean_comment_row": int(row.clean_comment_row),
                "utterance_id": str(row.utterance_id),
                "conversation_id": str(row.conversation_id),
            }
            for row in eligible[
                ["embedding_row", "clean_comment_row", "utterance_id", "conversation_id"]
            ].itertuples(index=False)
        ],
        "completed_chunks": [],
    }


def manifest_matches(manifest, eligible):
    expected = embedding_manifest(eligible)
    for key in ("version", "model", "n_embeddings", "row_fingerprint", "rows"):
        if manifest.get(key) != expected[key]:
            return False
    return True


def prepare_embedding_checkpoint(checkpoint_dir, eligible, force):
    manifest_path = checkpoint_dir / "embedding_manifest.json"
    embeddings_path = checkpoint_dir / "embeddings.npy"
    if force:
        for item in (manifest_path, embeddings_path):
            if item.exists():
                item.unlink()

    # An interruption immediately after the initial manifest write leaves no
    # completed work, so it can safely resume as a fresh embedding allocation.
    if manifest_path.exists() and not embeddings_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            partial_manifest = json.load(handle)
        if partial_manifest.get("completed_chunks"):
            raise RuntimeError(
                "Embedding checkpoint is incomplete. Re-run with --force-recompute "
                "to rebuild it safely."
            )
        manifest_path.unlink()
    elif embeddings_path.exists() and not manifest_path.exists():
        raise RuntimeError(
            "Embedding checkpoint is incomplete. Re-run with --force-recompute "
            "to rebuild it safely."
        )
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not manifest_matches(manifest, eligible):
            raise RuntimeError(
                "Saved embeddings do not match the current eligible comments. "
                "Use --force-recompute to start a new checkpoint."
            )
        return manifest, embeddings_path

    manifest = embedding_manifest(eligible)
    atomic_write_json(manifest, manifest_path)
    return manifest, embeddings_path


def embed_comments(eligible, checkpoint_dir, model, batch_size, chunk_size, force):
    manifest, embeddings_path = prepare_embedding_checkpoint(checkpoint_dir, eligible, force)
    manifest_path = checkpoint_dir / "embedding_manifest.json"
    n_comments = len(eligible)
    completed = {int(chunk["start"]) for chunk in manifest.get("completed_chunks", [])}
    chunks = [(start, min(start + chunk_size, n_comments))
              for start in range(0, n_comments, chunk_size)]

    if embeddings_path.exists():
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
        if embeddings.shape[0] != n_comments:
            raise AssertionError("Embedding rows no longer align with the eligible-comment manifest.")
    else:
        embeddings = None

    completed_rows = sum(end - start for start, end in chunks if start in completed)
    progress = tqdm(total=n_comments, initial=completed_rows, unit="comment",
                    desc="Embedding comments")
    try:
        for start, end in chunks:
            if start in completed:
                continue
            texts = eligible.iloc[start:end]["text"].tolist()
            with torch.inference_mode():
                values = model.encode(
                    texts, batch_size=batch_size, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                ).astype(np.float32, copy=False)
            if embeddings is None:
                embeddings = np.lib.format.open_memmap(
                    embeddings_path, mode="w+", dtype=np.float32,
                    shape=(n_comments, values.shape[1]),
                )
            if values.shape != (end - start, embeddings.shape[1]):
                raise AssertionError("Unexpected embedding shape; checkpoint was not updated.")
            embeddings[start:end] = values
            embeddings.flush()
            manifest["embedding_dimension"] = int(embeddings.shape[1])
            manifest["completed_chunks"].append({"start": start, "end": end})
            atomic_write_json(manifest, manifest_path)
            completed.add(start)
            progress.update(end - start)
    finally:
        progress.close()
        if embeddings is not None:
            embeddings.flush()

    if embeddings is None:  # all chunks were already complete at startup
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r")
    if len(completed) != len(chunks):
        raise AssertionError("Not all embedding chunks were completed.")
    assert embeddings.shape[0] == len(eligible)
    return embeddings


def coverage_curve(embeddings):
    similarities = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    coverage = np.maximum.accumulate(similarities, axis=0).mean(axis=1)
    if coverage[-1] == 0:
        raise AssertionError("Final semantic coverage cannot be normalized.")
    coverage = np.clip(coverage / coverage[-1], -1.0, 1.0)
    assert np.isclose(coverage[-1], 1.0, atol=1e-6)
    return coverage


def saturation_point(coverage, threshold):
    index = int(np.argmax(coverage >= threshold))
    return (index + 1) / len(coverage) if coverage[index] >= threshold else np.nan


def interpolate_curve(coverage, grid):
    n = len(coverage)
    positions = np.arange(1, n + 1) / n
    return np.interp(grid, np.concatenate([[0.0], positions]),
                     np.concatenate([[0.0], coverage]))


def load_thread_checkpoint(out, eligible, force):
    progress_path = out / "checkpoints" / "thread_progress.csv"
    metrics_path, curves_path = out / "thread_metrics.csv", out / "thread_curves.csv"
    if force:
        for item in (progress_path, metrics_path, curves_path):
            if item.exists():
                item.unlink()
    if not progress_path.exists():
        return pd.DataFrame(columns=["conversation_id"]), pd.DataFrame(), pd.DataFrame()

    progress = pd.read_csv(progress_path, dtype={"conversation_id": str})
    if progress["conversation_id"].duplicated().any():
        raise AssertionError("A thread appears twice in the progress checkpoint.")
    completed = set(progress["conversation_id"])
    metrics = pd.read_csv(metrics_path, dtype={"conversation_id": str}) if metrics_path.exists() else pd.DataFrame()
    curves = pd.read_csv(curves_path, dtype={"conversation_id": str}) if curves_path.exists() else pd.DataFrame()
    # Ignore result rows written just before an interruption but not committed.
    metrics = metrics[metrics["conversation_id"].isin(completed)].copy()
    curves = curves[curves["conversation_id"].isin(completed)].copy()
    if set(metrics.get("conversation_id", [])) != completed or set(curves.get("conversation_id", [])) != completed:
        raise RuntimeError("Thread result checkpoint is incomplete; use --force-recompute to rebuild it.")
    eligible_ids = set(eligible["conversation_id"].astype(str))
    if not completed.issubset(eligible_ids):
        raise RuntimeError("Thread progress does not match current eligible comments; use --force-recompute.")
    return progress, metrics, curves


def analyze_threads(eligible, embeddings, out, force):
    progress, metrics, curves = load_thread_checkpoint(out, eligible, force)
    progress_path = out / "checkpoints" / "thread_progress.csv"
    metrics_path, curves_path = out / "thread_metrics.csv", out / "thread_curves.csv"
    completed = set(progress.get("conversation_id", pd.Series(dtype=str)).astype(str))
    groups = [(str(cid), group) for cid, group in eligible.groupby("conversation_id", sort=False)]
    grid = np.linspace(0, 1, GRID_POINTS)
    bar = tqdm(total=len(groups), initial=len(completed), unit="thread", desc="Processing threads")
    try:
        for conversation_id, group in groups:
            if conversation_id in completed:
                continue
            assert len(group) >= MIN_COMMENTS
            indices = group["embedding_row"].to_numpy(dtype=np.int64)
            assert np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
            coverage = coverage_curve(np.asarray(embeddings[indices]))
            s80, s90 = saturation_point(coverage, 0.80), saturation_point(coverage, 0.90)
            assert s80 <= s90
            metric_row = pd.DataFrame([{
                "conversation_id": conversation_id, "n_comments": len(group),
                "s80": s80, "s90": s90,
            }])
            curve_rows = pd.DataFrame({
                "conversation_id": conversation_id, "thread_position": grid,
                "semantic_coverage": interpolate_curve(coverage, grid),
            })
            # Results are atomically replaced before progress is committed. On an
            # interruption, uncommitted rows are filtered out on restart.
            metrics = pd.concat([metrics, metric_row], ignore_index=True)
            curves = pd.concat([curves, curve_rows], ignore_index=True)
            atomic_write_csv(metrics, metrics_path)
            atomic_write_csv(curves, curves_path)
            progress = pd.concat([progress, pd.DataFrame([{"conversation_id": conversation_id}])],
                                 ignore_index=True)
            atomic_write_csv(progress, progress_path)
            completed.add(conversation_id)
            bar.update(1)
    finally:
        bar.close()
    if metrics["conversation_id"].duplicated().any():
        raise AssertionError("No thread may be processed twice.")
    return metrics, curves


def summarize(thread_df, curve_df):
    rho, p_value = spearmanr(thread_df["n_comments"], thread_df["s80"])
    summary = {
        "model": MODEL_NAME,
        "minimum_usable_comments_per_thread": MIN_COMMENTS,
        "n_eligible_threads": int(len(thread_df)),
        "s80": {
            "median": float(thread_df["s80"].median()),
            "mean": float(thread_df["s80"].mean()),
            "q25": float(thread_df["s80"].quantile(0.25)),
            "q75": float(thread_df["s80"].quantile(0.75)),
        },
        "s90": {
            "median": float(thread_df["s90"].median()),
            "mean": float(thread_df["s90"].mean()),
            "q25": float(thread_df["s90"].quantile(0.25)),
            "q75": float(thread_df["s90"].quantile(0.75)),
        },
        "spearman_thread_length_vs_s80": {"rho": float(rho), "p_value": float(p_value)},
    }
    # CSV round-tripping can create tiny representation variants of the shared
    # 101-point grid; normalize only the output coordinate before aggregation.
    curve_df = curve_df.copy()
    curve_df["thread_position"] = curve_df["thread_position"].round(12)
    aggregate = curve_df.groupby("thread_position")["semantic_coverage"].agg(
        ["mean", "median", "count"]
    ).reset_index()
    return summary, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Local ConvoKit reddit-corpus-small directory")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32)")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Checkpoint chunk size (default: 2000)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-recompute", action="store_true", help="Discard saved checkpoints and recompute")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.chunk_size <= 0:
        parser.error("--batch-size and --chunk-size must be positive.")

    # Delay heavyweight imports so --help is instant and the model is still
    # loaded only once for an actual run.
    global torch, SentenceTransformer
    import torch
    from sentence_transformers import SentenceTransformer

    out = Path(args.output)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.force_recompute:
        for item in (out / "aggregate_curve.csv", out / "summary.json"):
            if item.exists():
                item.unlink()

    print("Loading corpus...")
    comments = load_convokit_corpus(args.corpus)
    atomic_write_csv(comments, out / "clean_comments.csv")
    eligible = eligible_comments(comments)
    print(f"Usable comments after cleaning: {len(comments)}")
    print(f"Eligible comments to embed: {len(eligible)} across {eligible['conversation_id'].nunique()} threads")

    device = choose_device(args.device)
    print(f"Selected device: {device.upper()}")
    print(f"Loading embedding model once: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = embed_comments(eligible, checkpoint_dir, model, args.batch_size,
                                args.chunk_size, args.force_recompute)
    metrics, curves = analyze_threads(eligible, embeddings, out, args.force_recompute)
    if len(metrics) != eligible["conversation_id"].nunique():
        raise AssertionError("Final outputs must contain every eligible thread.")
    summary, aggregate = summarize(metrics, curves)
    atomic_write_csv(aggregate, out / "aggregate_curve.csv")
    atomic_write_json(summary, out / "summary.json")
    print("\n=== FROZEN PRIMARY RESULTS ===")
    print(json.dumps(summary, indent=2))
    # Avoid a display-only UnicodeEncodeError in legacy Windows consoles when
    # the absolute path contains non-ASCII characters.
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()
