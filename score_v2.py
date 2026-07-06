import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import pandas as pd, numpy as np, re
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"

RUBRIC = """You rate how TAILORED a US federal solicitation specification is — how much it reads as written so only ONE firm can win. Use this exact scale:
0 = OPEN. Generic requirement; many qualified firms could compete. Brand names, if any, appear with "or equal" and listed salient characteristics; ordinary experience asks.
1 = NARROWED. Somewhat restrictive; real but surmountable barriers (a fairly specific cert, moderately narrow experience, proprietary lean) that thin the field without clearly naming one winner.
2 = WIRED. Reads written-for-one: a brand/make/model or exact part number/NSN is REQUIRED with no genuine "or equal", sole-source / "only one responsible source" / "intent to award to [firm]" language, or incumbent-only experience only one vendor plausibly meets.
Critical rule: a named manufacturer or exact part number/NSN is a STRONG tailoring signal EVEN IF the words "sole source" never appear. But a brand named with genuine "or equal" language AND listed salient characteristics is NOT tailored (that is lawful brand-name-or-equal, score 0-1).
Judge ONLY the specification; ignore FAR boilerplate, dollar amounts, and length."""

NSN = re.compile(r"\b\d{4}-\d{2}-\d{3}-\d{4}\b")
PARTNO = re.compile(r"\b(?:P/N|PN|part\s*(?:number|no)|model|NSN|MFR\s*PN)\s*[:#]?\s*[A-Z0-9][A-Z0-9\-]{3,}\b", re.I)
OR_EQUAL = re.compile(r"\b(brand\s*name\s*or\s*equal|or\s*equal|salient\s*characteristic)\b", re.I)
SOLE = re.compile(r"\b(sole\s*source|only\s*one\s*responsible\s*source|intent\s*to\s*(?:sole.?source|award)|no\s*substitut|single\s*source|brand\s*name\s*only|no\s*equal)\b", re.I)

def features(t):
    t = str(t)
    return {
        "nsn": bool(NSN.search(t)),
        "partno": bool(PARTNO.search(t)),
        "or_equal": bool(OR_EQUAL.search(t)),
        "sole": bool(SOLE.search(t)),
    }

def pretag(f):
    tags = []
    if f["nsn"]: tags.append("exact NSN present")
    if f["partno"]: tags.append("exact part/model number present")
    if f["sole"]: tags.append("sole-source / no-substitute language present")
    if f["or_equal"]: tags.append("'or equal' / salient-characteristics language present")
    return "; ".join(tags) if tags else "none detected"

def prompt1(t, f):
    return (f"{RUBRIC}\n\nAUTOMATED PRE-SCAN (verify against the text): {pretag(f)}\n\n"
            f"SPECIFICATION:\n{str(t)[:1600]}\n\n"
            "Reason in 2-3 sentences: (1) list any named manufacturers, brands, models, part numbers or NSNs; "
            "(2) is there genuine brand-name-or-equal language with salient characteristics, or hard sole-source/single-make lock-in; "
            "(3) conclude.\nAssessment:")

llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)
tok = llm.get_tokenizer()
sp_cot = SamplingParams(temperature=0.0, max_tokens=220)
sp_ev = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

def ev_scores(texts):
    feats = [features(t) for t in texts]
    p1 = [prompt1(t, f) for t, f in zip(texts, feats)]
    r1 = llm.generate(p1, sp_cot)
    reason = [o.outputs[0].text for o in r1]
    p2 = [a + b + "\n\nFinal RATING (reply with a single digit 0, 1, or 2): " for a, b in zip(p1, reason)]
    r2 = llm.generate(p2, sp_ev)
    out = []
    for o in r2:
        lp = o.outputs[0].logprobs[0]
        d = {}
        for tid, obj in lp.items():
            k = obj.decoded_token.strip()
            if k in {"0", "1", "2"}:
                d[k] = max(d.get(k, -1e9), obj.logprob)
        if not d:
            out.append(np.nan); continue
        z = {k: np.exp(v) for k, v in d.items()}
        s = sum(z.values())
        ev = sum(int(k) * (z.get(k, 0.0) / s) for k in ("0", "1", "2"))
        out.append(ev)
    return np.array(out, float), pd.DataFrame(feats)

def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return (np.greater.outer(p, n).sum() + 0.5 * np.equal.outer(p, n).sum()) / (len(p) * len(n))

def spearman(x, y):
    return np.corrcoef(pd.Series(np.asarray(x, float)).rank(), pd.Series(np.asarray(y, float)).rank())[0, 1]

def qwk(a, b, K=3):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((K, K))
    for i, j in zip(a, b): O[i, j] += 1
    w = np.array([[(i - j) ** 2 / (K - 1) ** 2 for j in range(K)] for i in range(K)])
    ha, hb = O.sum(1), O.sum(0)
    E = np.outer(ha, hb) / O.sum()
    return 1 - (w * O).sum() / (w * E).sum()

def cutpoints(ev, h):
    best, bc = -9, (0.5, 1.5)
    for c1 in np.arange(0.2, 1.6, 0.05):
        for c2 in np.arange(c1 + 0.1, 1.9, 0.05):
            lab = np.where(ev < c1, 0, np.where(ev < c2, 1, 2))
            k = qwk(lab, h)
            if k > best: best, bc = k, (c1, c2)
    return bc

d = pd.read_parquet("data/audit_gate.parquet").dropna(subset=["text_clean"]).reset_index(drop=True)
ev, feats = ev_scores(d.text_clean.tolist())
d["ev"] = ev
dd = d.dropna(subset=["ev"]).reset_index(drop=True)
rng = np.random.default_rng(11)
idx = rng.permutation(len(dd)); dev, te = idx[:len(dd) // 2], idx[len(dd) // 2:]
c1, c2 = cutpoints(dd.ev.values[dev], dd.h.values[dev])
lab = np.where(dd.ev < c1, 0, np.where(dd.ev < c2, 1, 2))

print(f"\n=== TIER-0 SCORER (logprob-EV + anchored/CoT + pre-tag) | n={len(dd)} ===")
print(f"baselines: prior ordinal rho=0.55, AUC=0.72")
print(f"AUC (h2 vs h0), continuous EV: {auc(dd[dd.h==2].ev.values, dd[dd.h==0].ev.values):.3f}")
print(f"Spearman(EV, human): full={spearman(dd.ev, dd.h):.3f}  TEST(locked)={spearman(dd.ev.values[te], dd.h.values[te]):.3f}")
print(f"cutpoints (dev-fit c1={c1:.2f} c2={c2:.2f}) -> QWK: dev={qwk(lab[dev], dd.h.values[dev]):.3f}  TEST(locked)={qwk(lab[te], dd.h.values[te]):.3f}")
print("confusion (human rows / model cols), test half:")
print(pd.crosstab(dd.h.values[te], lab[te]).to_string())
print("\nfeature hit-rates:", {k: round(feats[k].mean(), 2) for k in feats.columns})
dd.assign(lab=lab).to_csv("data/v2_scores.csv", index=False)
