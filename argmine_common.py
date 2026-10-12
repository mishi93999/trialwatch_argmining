"""Shared settings, data loading and evaluation helpers for the argument mining notebooks (01 to 07)."""
import re
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 160)
pd.set_option("display.width", 200)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
CORPUS_CSV = BASE / "full_df_clean2.csv"
GOLD_CSV = BASE / "gold_annotations.csv"      # written by extract_annotations.ipynb

OUT_DIR = BASE / "argument_mining"            # results and sheets for the lawyers
DATA_DIR = OUT_DIR / "intermediate"           # data passed between notebooks
PRED_DIR = OUT_DIR / "predictions"            # one file per method
for _d in (OUT_DIR, DATA_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SENTS_PKL = DATA_DIR / "sentences.pkl"            # 01_segmentation
GOLD_SENTS_CSV = DATA_DIR / "gold_sentences.csv"  # 02_gold_standard
TEST_SPLIT_CSV = OUT_DIR / "test_judgments.csv"   # 02_gold_standard, drawn once and then kept fixed
M1_HITS_CSV = DATA_DIR / "m1_hits.csv"            # 03_keyword_methods

# ── Settings shared by several notebooks ──────────────────────────────────────
TEXT_FIELDS = ["court_reasoning", "conclusion"]   # main run; four-section ablation is in archive_four_sections/
ENCODER_NAME = "law-ai/InLegalBERT"
N_BOOT = 1000
SEED = 42

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORIES = {
    "C1": "Ingredients of the offence and exceptions",
    "C2": "Balance of speech and reputation, public officials",
    "C3": "Procedural grounds",
    "C4": "Insufficient evidence",
    "C5": "Abuse of process",
}
CATS = list(CATEGORIES)
K = len(CATS)
SHORT = {"C1": "Ingredients/exceptions", "C2": "Speech vs reputation", "C3": "Procedural",
         "C4": "Insufficient evidence", "C5": "Abuse of process"}

# Paste Saanvi's coded definitions here, verbatim. They serve as extra exemplars in M2.
CATEGORY_DEFINITIONS = {
    "C1": "The court examines whether the ingredients of defamation under Section 499 are made out, such as an imputation, publication to a third person, and intention or knowledge of harm to reputation, or whether an exception applies, such as truth for the public good, fair comment, or good faith.",
    "C2": "The court weighs freedom of speech and expression under Article 19 against the right to reputation under Article 21, including the latitude for criticism of public officials and public figures on matters of public interest.",
    "C3": "The court decides on procedural grounds, such as jurisdiction, sanction, standing of the person aggrieved, limitation, the inquiry required before issuing summons, procedural irregularity, or that a defence can only be examined at trial.",
    "C4": "The court finds that the material or evidence on record is insufficient to prove the allegation or to support the case, or that a party failed to prove or establish its claim.",
    "C5": "The court finds that the complaint is an abuse of the process of law, filed with mala fide intent, an ulterior motive, or to harass or take revenge on the accused.",
}


def normalize_label(raw):
    # Map raw gold labels ("1_ingredients_exceptions", "C3", "abuse of process") to C1..C5.
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    m = re.match(r"^(?:c|cat|category)?\s*_?\s*([1-5])(?![0-9])", s)
    if m:
        return f"C{m.group(1)}"
    for cat, pat in [("C5", r"abuse"), ("C4", r"evidence"), ("C3", r"procedur"),
                     ("C2", r"speech|reputation|balance|public official"),
                     ("C1", r"ingredient|element|exception")]:
        if re.search(pat, s):
            return cat
    return None   # unconfirmed or unknown labels are excluded from evaluation


# ── Loading data written by earlier notebooks ─────────────────────────────────
def load_corpus():
    corpus = pd.read_csv(CORPUS_CSV)
    corpus = corpus.dropna(subset=["doc_id"]).copy()
    corpus["doc_id"] = corpus["doc_id"].astype(np.int64)
    corpus["reason_text"] = (corpus[TEXT_FIELDS].fillna("").astype(str)
                             .agg("\n\n".join, axis=1).str.strip())
    return corpus


def titles(corpus):
    return dict(zip(corpus.doc_id, corpus.get("title", corpus.doc_id.astype(str))))


def _require(path, notebook):
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(BASE)} not found. Run {notebook} first.")


def load_sentences():
    _require(SENTS_PKL, "01_segmentation.ipynb")
    return pd.read_pickle(SENTS_PKL)


def load_gold():
    # Returns GOLD_IDX (row positions into the sentence table), Y (sentences x categories),
    # DOCS (judgment of each gold sentence), the sorted list of gold judgments, and TEST
    # (True for sentences of held out test judgments, which no method may learn or tune on).
    _require(GOLD_SENTS_CSV, "02_gold_standard.ipynb")
    g = pd.read_csv(GOLD_SENTS_CSV)
    docs = g.doc_id.values
    return (g.sid.values, g[CATS].values.astype(int), docs, sorted(np.unique(docs)),
            (g.split == "test").values)


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def save_preds(name, gold, all_sents=None):
    # gold: predictions on the gold sentences, in load_gold() order.
    # all_sents: predictions on every sentence, needed to apply the method to the whole corpus.
    arrays = {"name": np.array(name), "gold": np.asarray(gold, dtype=np.int8)}
    if all_sents is not None:
        arrays["all"] = np.asarray(all_sents, dtype=np.int8)
    np.savez_compressed(PRED_DIR / f"{_slug(name)}.npz", **arrays)


def load_preds():
    # {method name: {"gold": array, "all": array or None}}, ordered M0, M1, ..., M4
    n_gold = len(pd.read_csv(GOLD_SENTS_CSV))
    out = {}
    for path in sorted(PRED_DIR.glob("*.npz")):
        with np.load(path) as f:
            name = str(f["name"])
            if len(f["gold"]) != n_gold:
                print(f"Skipping {name}: saved before the gold set changed. Re-run its notebook.")
                continue
            out[name] = {"gold": f["gold"].astype(int),
                         "all": f["all"].astype(int) if "all" in f.files else None}
    return out


# ── Sentence embeddings (InLegalBERT) ─────────────────────────────────────────
_ENCODER = {}


def encode(texts, batch_size=32, max_length=256):
    import torch
    from transformers import AutoTokenizer, AutoModel
    if not _ENCODER:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() else "cpu")
        _ENCODER.update(tok=AutoTokenizer.from_pretrained(ENCODER_NAME),
                        model=AutoModel.from_pretrained(ENCODER_NAME).to(device).eval(),
                        device=device)
    tok, model, device = _ENCODER["tok"], _ENCODER["model"], _ENCODER["device"]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def load_embeddings(sents):
    # Returns centered, unit length embeddings of every sentence, and `post`, which applies
    # the same centering to newly encoded texts. Cached, keyed on the sentence texts.
    key = hashlib.md5(("||".join(sents.text) + ENCODER_NAME).encode()).hexdigest()[:10]
    cache = DATA_DIR / f"emb_{ENCODER_NAME.replace('/', '_')}_{key}.npy"
    if cache.exists():
        x_raw = np.load(cache)
    else:
        x_raw = encode(sents.text.tolist())
        np.save(cache, x_raw)
    mean = x_raw.mean(0)

    def post(xr):
        xc = xr - mean
        return xc / np.clip(np.linalg.norm(xc, axis=1, keepdims=True), 1e-9, None)

    return post(x_raw), post


# ── Evaluation ────────────────────────────────────────────────────────────────
def macro_f1(Yt, Yp):
    present = Yt.sum(0) > 0
    if not present.any():
        return np.nan
    return f1_score(Yt[:, present], Yp[:, present], average="macro", zero_division=0)


def summarize(Yt, Yp):
    present = Yt.sum(0) > 0
    return {"macro_P": precision_score(Yt[:, present], Yp[:, present], average="macro", zero_division=0),
            "macro_R": recall_score(Yt[:, present], Yp[:, present], average="macro", zero_division=0),
            "macro_F1": macro_f1(Yt, Yp),
            "micro_F1": f1_score(Yt, Yp, average="micro", zero_division=0),
            "exact_match_acc": accuracy_score(Yt, Yp)}   # all five labels right; inflated by sentences with no category


def per_category(Yt, Yp):
    return pd.DataFrame({
        "precision": precision_score(Yt, Yp, average=None, zero_division=0),
        "recall": recall_score(Yt, Yp, average=None, zero_division=0),
        "f1": f1_score(Yt, Yp, average=None, zero_division=0),
        "support": Yt.sum(0)}, index=[SHORT[c] for c in CATS])


def to_case_level(M, docs):
    return pd.DataFrame(M).groupby(docs).max().values


def _doc_rows(docs):
    return {d: np.where(docs == d)[0] for d in np.unique(docs)}


def boot_ci(Yt, Yp, docs, n=N_BOOT, seed=SEED):
    rows = _doc_rows(docs)
    rng, ids, vals = np.random.default_rng(seed), np.array(list(rows)), []
    for _ in range(n):
        idx = np.concatenate([rows[d] for d in rng.choice(ids, len(ids))])
        vals.append(macro_f1(Yt[idx], Yp[idx]))
    return np.nanpercentile(vals, [2.5, 97.5])


def paired_boot(Yt, PA, PB, docs, n=N_BOOT, seed=SEED):
    rows = _doc_rows(docs)
    rng, ids, diff = np.random.default_rng(seed), np.array(list(rows)), []
    for _ in range(n):
        idx = np.concatenate([rows[d] for d in rng.choice(ids, len(ids))])
        diff.append(macro_f1(Yt[idx], PA[idx]) - macro_f1(Yt[idx], PB[idx]))
    diff = np.array(diff)
    return np.nanmean(diff), np.nanpercentile(diff, [2.5, 97.5]), np.nanmean(diff > 0)
