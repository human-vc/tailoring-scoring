# tailoring-scoring

Full-corpus scoring of the federal-solicitation tailoring measure (Qwen2.5-7B-Instruct, vLLM). Gates on the human-coded audit set before scoring the full corpus.

## Run on a GPU box (≥24GB: A10G / L40S / A100)

```bash
pip install -r requirements.txt
python score_brev.py
```

The gate prints `[GATE] tailored-vs-generic AUC` — expect ~0.9 (validated 0.915). If it drops below 0.85 the run aborts before the full pass.

Output: `data/e_full_scores.parquet` (`NoticeId`, `rscore`).
