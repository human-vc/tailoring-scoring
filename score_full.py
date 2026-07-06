import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import pandas as pd, numpy as np, re
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GATE_MIN_RHO = 0.45

RUBRIC = """You rate how TAILORED a US federal solicitation specification is — how much it reads as written so only ONE specific firm can win. Use this exact 3-point scale:
0 = OPEN. Generic requirement; many qualified firms could compete. No brand lock-in, no sole-source language, ordinary experience asks. (A brand or NSN cited routinely, or named with genuine "or equal", is still 0.)
1 = NARROWED. Somewhat restrictive; real but surmountable barriers (a fairly specific cert, moderately narrow experience, some proprietary lean) that thin the field without clearly naming one winner.
2 = WIRED. Reads written-for-one: brand/make/model required with no genuine "or equal", sole-source / "intent to award to [firm]", or incumbent-only experience only one vendor plausibly meets.
Judge ONLY the specification; ignore FAR boilerplate, dollar amounts, and length.
Think in one sentence, then end with exactly: RATING: <0, 1, or 2>"""

def prm(t):
    return f"{RUBRIC}\n\nSPECIFICATION:\n{str(t)[:1600]}\n\nAssessment:"

def parse(txt):
    m = re.findall(r"RATING:\s*([012])", txt) or re.findall(r"\b([012])\b", txt)
    return int(m[-1]) if m else np.nan

llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)
sp = SamplingParams(temperature=0.0, max_tokens=140)

def score(texts):
    outs = llm.generate([prm(t) for t in texts], sp)
    return np.array([parse(o.outputs[0].text) for o in outs], float)

def spearman(x, y):
    m = ~(pd.isna(x) | pd.isna(y)); x, y = np.asarray(x)[m], np.asarray(y)[m]
    return np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]

g = pd.read_parquet("data/audit_gate.parquet").dropna(subset=["text_clean"])
gs = score(g.text_clean.tolist())
rho = spearman(gs, g.h.values)
print(f"[GATE] ordinal rho vs human = {rho:.3f} (n={len(g)})")
if rho < GATE_MIN_RHO:
    raise SystemExit(f"[GATE FAILED] rho {rho:.3f} < {GATE_MIN_RHO} — base scorer drifted, STOP.")
print(f"[GATE PASSED] proceeding to full corpus.\n")

inp = pd.read_parquet("data/e_score_inputs.parquet")
inp["ord"] = score(inp.text_clean.tolist())
inp[["NoticeId", "ord"]].to_parquet("data/e_full_ord.parquet", index=False)
print(f"[FULL] scored {len(inp)} | valid {inp.ord.notna().mean():.3f} | dist {inp.ord.value_counts(dropna=False).to_dict()}")
print("[FULL] wrote data/e_full_ord.parquet — sync back for local fusion")
