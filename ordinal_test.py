import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import pandas as pd, numpy as np, re
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
RUBRIC = """You rate how TAILORED a US federal solicitation specification is — how much it reads as written so only ONE specific firm can win. Use this exact 3-point scale:
0 = OPEN. Generic requirement; many qualified firms could compete. No brand lock-in, no sole-source language, ordinary experience asks.
1 = NARROWED. Somewhat restrictive; real but surmountable barriers (a fairly specific cert, moderately narrow experience, some proprietary lean) that thin the field without clearly naming one winner.
2 = WIRED. Reads written-for-one: brand/make/model required (not "or equal"), sole-source / "intent to award to [firm]", exact part numbers, or incumbent-only experience only one vendor plausibly meets.
Judge ONLY the specification; ignore FAR boilerplate, dollar amounts, and length.
Think in one sentence, then end with exactly: RATING: <0, 1, or 2>"""

def prm(t):
    return f"{RUBRIC}\n\nSPECIFICATION:\n{str(t)[:1600]}\n\nAssessment:"

def parse(txt):
    m = re.findall(r"RATING:\s*([012])", txt)
    if m:
        return int(m[-1])
    m2 = re.findall(r"\b([012])\b", txt)
    return int(m2[-1]) if m2 else np.nan

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]

llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, max_tokens=140)

d = pd.read_parquet("data/audit_gate.parquet").dropna(subset=["text_clean"]).reset_index(drop=True)
outs = llm.generate([prm(t) for t in d.text_clean], sp)
d["ord"] = [parse(o.outputs[0].text) for o in outs]
dd = d.dropna(subset=["ord"])

rng = np.random.default_rng(11)
idx = rng.permutation(len(dd)); dev, te = idx[:len(dd)//2], idx[len(dd)//2:]
full = spearman(dd["ord"], dd.h)
devr = spearman(dd["ord"].values[dev], dd.h.values[dev])
test = spearman(dd["ord"].values[te], dd.h.values[te])
print(f"\nordinal 7B scored {len(dd)} audit items | valid {dd['ord'].notna().mean():.2f}")
print(f"  prior 0-100 3B rho (full) = 0.416")
print(f"  ordinal 7B rho: dev={devr:.3f}  TEST(locked)={test:.3f}  full={full:.3f}")
print("  confusion (human rows / model cols):")
print(pd.crosstab(dd.h, dd["ord"]).to_string())
