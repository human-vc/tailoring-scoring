import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import pandas as pd, numpy as np, re, time
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GATE_MIN_AUC = 0.85
RUBRIC = """You score how RESTRICTIVE/TAILORED a US federal solicitation specification is — how much the text appears written so only ONE specific firm could plausibly win. Consider these signals:
1. Names a specific brand/make/model as REQUIRED (not "or equal").
2. Sole-source / "only one responsible source" / "intent to award to [firm]" language.
3. Requires narrow certifications/qualifications few vendors hold.
4. Proprietary features, exact part numbers, or one-of-a-kind specs.
5. Experience/past-performance requirements that favor an incumbent.
6. Overall reads as copied from one vendor's product sheet.
Judge ONLY the specification; ignore FAR/boilerplate clauses, dollar amounts, and document length.
Think briefly, then end with exactly: SCORE: <integer 0-100> where 0=fully generic/open, 100=written-for-one."""

def prm(t):
    return f"{RUBRIC}\n\nSPECIFICATION:\n{t}\n\nAssessment:"

llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, max_tokens=160)

def build_prompts(texts, mode):
    p = [prm(str(t)[:1600]) for t in texts]
    if mode == "chat":
        p = [tok.apply_chat_template([{"role": "user", "content": x}],
                                     tokenize=False, add_generation_prompt=True) for x in p]
    return p

def parse(txt):
    m = re.findall(r"SCORE:\s*(\d{1,3})", txt) or re.findall(r"\b(\d{1,3})\b", txt)
    return min(int(m[-1]), 100) if m else np.nan

def score_texts(texts, mode, return_raw=False):
    outs = llm.generate(build_prompts(texts, mode), sp)
    raw = [o.outputs[0].text for o in outs]
    res = np.array([parse(t) for t in raw], dtype=float)
    return (res, raw) if return_raw else res

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return (np.greater.outer(pos, neg).sum() + 0.5*np.equal.outer(pos, neg).sum()) / (len(pos)*len(neg))

def spearman(x, y):
    rx = pd.Series(np.asarray(x, float)).rank().values
    ry = pd.Series(np.asarray(y, float)).rank().values
    return np.corrcoef(rx, ry)[0, 1]

g = pd.read_parquet("data/audit_gate.parquet")
results = {}
for mode in ("raw", "chat"):
    s, raw = score_texts(g.text_clean.tolist(), mode, return_raw=True)
    gv = pd.DataFrame({"s": s, "h": g.h.values}).dropna(subset=["s"])
    a = auc(gv[gv.h == 2].s.values, gv[gv.h == 0].s.values)
    rho = spearman(gv.s.values, gv.h.values)
    results[mode] = (a, s)
    print(f"[GATE:{mode}] AUC = {a:.3f} | Spearman = {rho:.3f} | mean {gv.s.mean():.1f} median {gv.s.median():.0f} valid {gv.s.notna().mean():.2f}")
    print(f"[GATE:{mode}] sample output:\n    {raw[0][:280].strip()}\n")

best = max(results, key=lambda m: results[m][0])
best_auc = results[best][0]
print(f"[GATE] best format = {best} (AUC {best_auc:.3f})")
if best_auc < GATE_MIN_AUC:
    raise SystemExit(f"[GATE FAILED] best AUC {best_auc:.3f} < {GATE_MIN_AUC}. Neither format reproduces — STOP, ping before trusting.")
print(f"[GATE PASSED] using {best} format for full corpus.\n")

inp = pd.read_parquet("data/e_score_inputs.parquet")
t0 = time.time()
inp["rscore"] = score_texts(inp.text_clean.tolist(), best)
inp[["NoticeId", "rscore"]].to_parquet("data/e_full_scores.parquet", index=False)
print(f"[FULL] scored {len(inp)} in {(time.time()-t0)/60:.1f}min | mean {inp.rscore.mean():.1f} "
      f"median {inp.rscore.median():.0f} valid {inp.rscore.notna().mean():.3f}")
print("[FULL] wrote data/e_full_scores.parquet")
