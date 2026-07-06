import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import pandas as pd, numpy as np, re
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"

RUBRIC = """You rate how TAILORED a US federal solicitation specification is — how much it reads as written so only ONE specific firm can win. Use this exact 3-point scale:
0 = OPEN. Generic requirement; many qualified firms could compete. No brand lock-in, no sole-source language, ordinary experience asks. (A brand or NSN cited routinely, or named with genuine "or equal", is still 0.)
1 = NARROWED. Somewhat restrictive; real but surmountable barriers (a fairly specific cert, moderately narrow experience, some proprietary lean) that thin the field without clearly naming one winner.
2 = WIRED. Reads written-for-one: brand/make/model required with no genuine "or equal", sole-source / "intent to award to [firm]", or incumbent-only experience only one vendor plausibly meets.
Judge ONLY the specification; ignore FAR boilerplate, dollar amounts, and length.
Think in one sentence, then end with exactly: RATING: <0, 1, or 2>"""

NSN = re.compile(r"\b\d{4}-\d{2}-\d{3}-\d{4}\b")
PARTNO = re.compile(r"\b(?:P/?N|part\s*(?:number|no\.?)|NSN|national\s+stock\s+number)\b\s*[:#]?\s*[A-Z0-9]*\d[A-Z0-9\-]{2,}", re.I)
BRANDMODEL = re.compile(r"\b(?:model|make/model|manufacturer(?:'s)?\s+(?:name|part))\b\s*[:#]?\s*[A-Z0-9][A-Z0-9\-]{2,}")
OR_EQUAL = re.compile(r"\bor\s*equal\b|\bsalient\s*characteristic", re.I)
SOLE = re.compile(r"\b(sole\s*source|only\s*one\s*responsible\s*source|intent\s*to\s*(?:sole.?source|award)|no\s*substitut|single\s*source|brand\s*name\s*only|no\s*equal\b)", re.I)

def features(t):
    t = str(t)
    return {"nsn": int(bool(NSN.search(t))), "partno": int(bool(PARTNO.search(t))),
            "brandmodel": int(bool(BRANDMODEL.search(t))), "or_equal": int(bool(OR_EQUAL.search(t))),
            "sole": int(bool(SOLE.search(t)))}

def p1(t):
    return f"{RUBRIC}\n\nSPECIFICATION:\n{str(t)[:1600]}\n\nAssessment:"

llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)
sp_cot = SamplingParams(temperature=0.0, max_tokens=140)
sp_ev = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

def score(texts):
    prompts = [p1(t) for t in texts]
    r1 = llm.generate(prompts, sp_cot)
    reason = [o.outputs[0].text for o in r1]
    lab = []
    for x in reason:
        m = re.findall(r"RATING:\s*([012])", x) or re.findall(r"\b([012])\b", x)
        lab.append(int(m[-1]) if m else np.nan)
    p2 = [a + b + "\n\nFinal RATING (single digit 0, 1, or 2): " for a, b in zip(prompts, reason)]
    r2 = llm.generate(p2, sp_ev)
    ev = []
    for o in r2:
        d = {}
        for tid, obj in o.outputs[0].logprobs[0].items():
            k = obj.decoded_token.strip()
            if k in {"0", "1", "2"}: d[k] = max(d.get(k, -1e9), obj.logprob)
        if not d: ev.append(np.nan); continue
        z = {k: np.exp(v) for k, v in d.items()}; s = sum(z.values())
        ev.append(sum(int(k) * z.get(k, 0.0) / s for k in ("0", "1", "2")))
    return np.array(lab, float), np.array(ev, float)

d = pd.read_parquet("data/audit_gate.parquet").dropna(subset=["text_clean"]).reset_index(drop=True)
lab, ev = score(d.text_clean.tolist())
feats = pd.DataFrame([features(t) for t in d.text_clean])
out = pd.concat([d[["blind_id", "h"]].reset_index(drop=True),
                 pd.DataFrame({"ord": lab, "ev": ev}), feats], axis=1)
out.to_csv("data/v3_scores.csv", index=False)

def sp(x, y):
    m = ~(pd.isna(x) | pd.isna(y)); x, y = np.asarray(x)[m], np.asarray(y)[m]
    return np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]
print(f"n={len(out)}  base LLM scores only:")
print(f"  rho(ord, human) = {sp(out.ord, out.h):.3f}   rho(ev, human) = {sp(out.ev, out.h):.3f}")
print(f"  feature hit-rates: {{'nsn':{feats.nsn.mean():.2f},'partno':{feats.partno.mean():.2f},'brandmodel':{feats.brandmodel.mean():.2f},'or_equal':{feats.or_equal.mean():.2f},'sole':{feats.sole.mean():.2f}}}")
print("wrote data/v3_scores.csv  -> sync back for local nested-CV fusion")
