"""
Script to annotate BELB predictions with dataset characteristics.
Phase 1 -> build metrics.json (from arboel)
Phase 2 -> inject metrics.json into each model's predictions
"""

# Imports
import os
import sys
from tqdm import tqdm
import argparse
import json
import re
from joblib import Parallel, delayed  # type: ignore
from typing import List, Dict, Any, Tuple, Optional
from math import comb
import scipy.stats as st
import numpy as np
import pandas as pd
import Levenshtein
from transformers import AutoTokenizer
from pathlib import Path
import sqlite3

# Globals
NORMALIZED_LEVENSHTEIN_THRESHOLD = 0.1
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
args=None


# Utils
def load_json_file(corpus_name: str, model_name: str) -> Any:
    base_dir = ROOT_DIR / "results" / corpus_name / model_name
    if model_name == "biosyn":
        input_filename = "predictions.json"
    else:
        input_filename = "filtered_predictions.json"
    input_path = base_dir / input_filename


    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] Could not find input file: {input_path}")
    
    print(f"[INFO] Loading predictions from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data, base_dir


def save_json_file(data: Any, base_dir: Path) -> None:
    output_path = base_dir / "annotated_filtered_predictions.json"

    print(f"[INFO] Saving annotated predictions to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_corpus_splits(corpus_name: str) -> Tuple[Any, Any]:
    base_dir = ROOT_DIR.parent / "belb" / "processed" / "corpora" / corpus_name
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    print() 
    
    if len(subdirs) == 0:
        raise FileNotFoundError(f"[ERROR] No subdirectory found in {base_dir}.")
    
    print(f"[INFO] Found {len(subdirs)} subdirectories under {base_dir}")

    for split_dir in subdirs:
        train_path = split_dir / "train.bioc.json"
        test_path = split_dir / "test.bioc.json"

        if train_path.exists() and test_path.exists():

            print(f"[INFO] Using split directory: {split_dir}")
            with open(train_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)

            with open(test_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)

            return train_data, test_data
        
    raise FileNotFoundError(f"[ERROR] Could not find a subdirectory in {base_dir} with train/test.")


def build_train_lookup(train_data: Any, metrics_dir: Path) -> Tuple[set, set, Dict[str, int], Dict[str, int]]:
    cache_path = metrics_dir / "train_lookup.json"
    print()

    if cache_path.exists() and not args.force:
        print(f"[INFO] Loading cached train_lookup from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        train_cuis = set(data["train_cuis"])
        train_surface_forms = set(data["train_surface_forms"])
        train_cui_counter = data.get("train_cui_counter", {})
        train_surface_form_counter = data.get("train_surface_form_counter", {})
        return train_cuis, train_surface_forms, train_cui_counter, train_surface_form_counter

    print("[INFO] Building lookup for training set...")
    train_cuis = set()
    train_surface_forms = set()
    train_cui_counter = {}
    train_surface_form_counter = {}

    for doc in train_data["documents"]:
        for passage in doc.get("passages", []):
            for ann in passage.get("annotations", []):
                cui = ann["infons"]["identifier"]
                surface = ann["text"].strip().lower()

                # Add to sets
                train_cuis.add(cui)
                train_surface_forms.add(surface)

                # Count occurrences
                train_cui_counter[cui] = train_cui_counter.get(cui, 0) + 1
                train_surface_form_counter[surface] = train_surface_form_counter.get(surface, 0) + 1

    print(f"  -> {len(train_cuis)} unique CUIs, {len(train_surface_forms)} unique surface forms")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_cuis": list(train_cuis),
            "train_surface_forms": list(train_surface_forms),
            "train_cui_counter": train_cui_counter,
            "train_surface_form_counter": train_surface_form_counter
        }, f, indent=2, ensure_ascii=False)

    return train_cuis, train_surface_forms, train_cui_counter, train_surface_form_counter


def build_test_lookup(test_data: Any, metrics_dir: Path) -> Dict[str, Tuple[str, str]]:
    cache_path = metrics_dir / "test_lookup.json"
    print() 

    if cache_path.exists() and not args.force:
        print(f"[INFO] Loading cached test_lookup from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        test_lookup = {k: tuple(v) for k, v in data.items()}
        return test_lookup

    print("[INFO] Building lookup for test annotations...")
    test_lookup = {}

    for doc in test_data["documents"]:
        for passage in doc.get("passages", []):
            for ann in passage.get("annotations", []):
                mention_id = ann["infons"]["hexdigest"]
                cui = ann["infons"]["identifier"]
                surface = ann["text"].strip().lower()
                test_lookup[mention_id] = (cui, surface)

    print(f"  -> Found {len(test_lookup)} mentions in test set")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(test_lookup, f, indent=2, ensure_ascii=False)

    return test_lookup


def build_synonymy_lookup(kb_path: Path, kb_table_name: str, metrics_dir: Path, test_lookup: Dict[str, Tuple[str, str]]) -> Dict[str, int]:
    cache_path = metrics_dir / "synonymy_lookup.json"
    print() 
    
    if cache_path.exists() and not args.force:
        print(f"[INFO] Loading cached synonymy_lookup from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            synonymy_lookup = json.load(f)
        return synonymy_lookup

    print(f"[INFO] Building synonymy lookup from: {kb_path} (table {kb_table_name}) for test CUIs only")
    conn = sqlite3.connect(kb_path)

    # Check if umls_identifier_mapping exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = set(row[0] for row in cursor.fetchall())
    use_mapping = "umls_identifier_mapping" in tables

    # Get test CUIs
    test_cuis = set(cui for cui, _ in test_lookup.values())
    print(f"[INFO] {len(test_cuis)} unique CUIs in test set")

    if use_mapping:
        print("[INFO] Detected UMLS-based KB → using JOIN with umls_identifier_mapping")
        query = f"""
        SELECT m.original_identifier AS cui, COUNT(k.name) AS num_synonyms
        FROM {kb_table_name} k
        JOIN umls_identifier_mapping m ON k.identifier = m.internal_identifier
        WHERE m.original_identifier IN ({','.join(['?'] * len(test_cuis))})
        GROUP BY m.original_identifier
        """
    else:
        print("[INFO] Detected simple KB → using direct identifier column")
        query = f"""
        SELECT identifier AS cui, COUNT(name) AS num_synonyms
        FROM {kb_table_name}
        WHERE identifier IN ({','.join(['?'] * len(test_cuis))})
        GROUP BY identifier
        """

    synonymy_lookup = {}
    cursor = conn.execute(query, list(test_cuis))
    for row in cursor:
        cui, num_synonyms = row
        synonymy_lookup[cui] = num_synonyms

    conn.close()

    print(f"[INFO] Synonymy lookup built -> {len(synonymy_lookup)} CUIs")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(synonymy_lookup, f, indent=2, ensure_ascii=False)

    return synonymy_lookup


def compute_homonyms_for_cui(cui1, names1, cui_to_names):
    num_homonyms = 0
    for cui2, names2 in cui_to_names.items():
        if cui1 == cui2:
            continue
        found = False
        for name1 in names1:
            for name2 in names2:
                lev_distance = Levenshtein.distance(name1, name2)
                norm_distance = lev_distance / max(len(name1), len(name2))
                if norm_distance < NORMALIZED_LEVENSHTEIN_THRESHOLD:
                    num_homonyms += 1
                    found = True
                    break
            if found:
                break
    return cui1, num_homonyms


def build_homonymy_lookup(kb_path: Path, kb_table_name: str, metrics_dir: Path, test_lookup: Dict[str, Tuple[str, str]]) -> Dict[str, int]:
    cache_path = metrics_dir / "homonymy_lookup.json"
    print()

    if cache_path.exists() and not args.force:
        print(f"[INFO] Loading cached homonymy_lookup from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            homonymy_lookup = json.load(f)
        return homonymy_lookup

    print(f"[INFO] Building homonymy lookup from: {kb_path} (table {kb_table_name}) for test CUIs only")
    conn = sqlite3.connect(kb_path)

    # Check if umls_identifier_mapping exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = set(row[0] for row in cursor.fetchall())
    use_mapping = "umls_identifier_mapping" in tables

    # Get test CUIs
    test_cuis = set(cui for cui, _ in test_lookup.values())
    print(f"[INFO] {len(test_cuis)} unique CUIs in test set")

    # Build query
    if use_mapping:
        print("[INFO] Detected UMLS-based KB → using JOIN with umls_identifier_mapping")
        query = f"""
        SELECT m.original_identifier AS cui, k.name
        FROM {kb_table_name} k
        JOIN umls_identifier_mapping m ON k.identifier = m.internal_identifier
        WHERE m.original_identifier IN ({','.join(['?'] * len(test_cuis))})
        """
    else:
        print("[INFO] Detected simple KB → using direct identifier column")
        query = f"""
        SELECT identifier AS cui, name
        FROM {kb_table_name}
        WHERE identifier IN ({','.join(['?'] * len(test_cuis))})
        """

    # Fetch names
    cui_to_names = {}
    cursor = conn.execute(query, list(test_cuis))
    for row in cursor:
        cui, name = row
        name = name.strip().lower()
        if cui not in cui_to_names:
            cui_to_names[cui] = []
        cui_to_names[cui].append(name)

    conn.close()
    print(f"[INFO] Collected surface forms for {len(cui_to_names)} test CUIs")

    # Parallel comparison
    print("[INFO] Comparing surface forms for homonymy (test CUIs only)... (parallelized with joblib)")
    num_workers = os.cpu_count()
    print(f"[INFO] Using {num_workers} CPU cores") 

    results = Parallel(n_jobs=num_workers)(
        delayed(compute_homonyms_for_cui)(cui1, names1, cui_to_names)
        for cui1, names1 in tqdm(cui_to_names.items(), desc=f"[INFO] Processing CUIs for homonymy", file=sys.stdout)
    )

    # Collect results
    homonymy_lookup = {cui: num_homonyms for cui, num_homonyms in results}

    print(f"[INFO] Homonymy lookup built -> {len(homonymy_lookup)} CUIs")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(homonymy_lookup, f, indent=2, ensure_ascii=False)

    return homonymy_lookup


def load_kb_cui_to_names(kb_path: Path, kb_table_name: str, metrics_dir: Path, test_lookup: Dict[str, Tuple[str, str]]) -> Dict[str, List[str]]:
    cache_path = metrics_dir / "kb_cui_to_names.json"
    print()

    if cache_path.exists() and not args.force:
        print(f"[INFO] Loading cached kb_cui_to_names from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            kb_cui_to_names = json.load(f)
        # Convert back to List[str] properly
        kb_cui_to_names = {k: list(v) for k, v in kb_cui_to_names.items()}
        return kb_cui_to_names

    print(f"[INFO] Loading KB names for test CUIs from table {kb_table_name}")
    conn = sqlite3.connect(kb_path)

    # Check if umls_identifier_mapping exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = set(row[0] for row in cursor.fetchall())
    use_mapping = "umls_identifier_mapping" in tables

    # Get test CUIs
    test_cuis = set(cui for cui, _ in test_lookup.values())
    print(f"[INFO] {len(test_cuis)} unique CUIs in test set")

    # Build query
    if use_mapping:
        print("[INFO] Detected UMLS-based KB → using JOIN with umls_identifier_mapping")
        query = f"""
        SELECT m.original_identifier AS cui, k.name
        FROM {kb_table_name} k
        JOIN umls_identifier_mapping m ON k.identifier = m.internal_identifier
        WHERE m.original_identifier IN ({','.join(['?'] * len(test_cuis))})
        """
    else:
        print("[INFO] Detected simple KB → using direct identifier column")
        query = f"""
        SELECT identifier AS cui, name
        FROM {kb_table_name}
        WHERE identifier IN ({','.join(['?'] * len(test_cuis))})
        """

    # Fetch names
    kb_cui_to_names = {}
    cursor = conn.execute(query, list(test_cuis))
    for row in cursor:
        cui, name = row
        name = name.strip().lower()
        if cui not in kb_cui_to_names:
            kb_cui_to_names[cui] = []
        kb_cui_to_names[cui].append(name)

    conn.close()
    print(f"[INFO] Loaded KB names for {len(kb_cui_to_names)} test CUIs")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(kb_cui_to_names, f, indent=2, ensure_ascii=False)

    return kb_cui_to_names


# Annotation functions -> build metrics dict
def build_metrics_dict(preds: List[Dict], 
                       train_cuis: set, 
                       train_surface_forms: set, 
                       train_cui_counter: Dict[str, int],
                       train_surface_form_counter: Dict[str, int],
                       test_lookup: Dict[str, Tuple[str, str]],
                       synonymy_lookup: Dict[str, int],
                       homonymy_lookup: Dict[str, int],
                       kb_cui_to_names:Dict[str, List[str]],) -> Dict[str, Dict]:
    print()
    print("[INFO] Building metrics dict.")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    global args

    metrics_dict = {}
    n_missing = 0

    for pred in preds:
        mention_id = pred["hexdigest"]
        if mention_id not in test_lookup:
            print(f"[WARN] Mention ID not found in test set: {mention_id}")
            n_missing += 1
            continue

        gold_cui, surface_form = test_lookup[mention_id]

        # zero-shot entity
        zero_shot_entity = 1 if gold_cui not in train_cuis else 0

        # zero-shot surface form
        zero_shot_surface = 1 if surface_form not in train_surface_forms else 0

        # mention length
        tokens = tokenizer.tokenize(surface_form)
        mention_length = len(tokens)
        mention_length_discrete = 1 if mention_length > args.length else 0

        # synonymy
        num_synonyms = synonymy_lookup.get(gold_cui, 0)
        synonymy_difficulty = 1 if num_synonyms <= args.synonymy else 0

        # homonymy
        num_homonyms = homonymy_lookup.get(gold_cui, 0)
        homonymy_difficulty = 1 if num_homonyms >= 1 else 0

        # lexical variation
        entity_names = kb_cui_to_names.get(gold_cui, [])
        lexical_distances = []

        for name in entity_names:
            lev_distance = Levenshtein.distance(surface_form, name)
            norm_distance = lev_distance / max(len(surface_form), len(name))
            lexical_distances.append(norm_distance)

        if lexical_distances:
            lexical_variation = min(lexical_distances)
        else:
            lexical_variation = 1.0  # worst case -> no name found

        lexical_variation_discrete = 1 if lexical_variation > args.variation else 0

        # mention frequency
        mention_frequency = train_surface_form_counter.get(surface_form, 0)
        mention_frequency_difficulty = 1 if mention_frequency <= args.frequency else 0

        # entity frequency
        entity_frequency = train_cui_counter.get(gold_cui, 0)
        entity_frequency_difficulty = 1 if entity_frequency <= args.frequency else 0

        # Save to dict
        metrics_dict[mention_id] = {
            "zero_shot_entity": zero_shot_entity,
            "zero_shot_surface_form": zero_shot_surface,
            "mention_length": mention_length,
            "mention_length_discrete": mention_length_discrete,
            "num_synonyms": num_synonyms,
            "synonymy_difficulty": synonymy_difficulty,
            "num_homonyms": num_homonyms,
            "homonymy_difficulty": homonymy_difficulty,
            "lexical_variation": lexical_variation,
            "lexical_variation_discrete": lexical_variation_discrete,
            "mention_frequency": mention_frequency,
            "mention_frequency_difficulty": mention_frequency_difficulty,
            "entity_frequency": entity_frequency,
            "entity_frequency_difficulty": entity_frequency_difficulty,
        }

    print(f"[INFO] Metrics dict built -> {len(metrics_dict)} mentions")
    if n_missing > 0:
        print(f"[WARN] {n_missing} mentions missing (not matched to test set)")

    return metrics_dict


def build_correct_vector(preds: List[Dict[str, Any]],
                         test_lookup: Dict[str, Tuple[str, str]],
                         pred_key_candidates: List[str] = ["predicted", "pred", "prediction", "entity_id", "cui", "predicted_cui"],
                         internal2orig: Optional[Dict[str, str]] = None
                         ) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      correct: array shape (n,) of 0/1
      ids: mention_ids aligned with correct
    Tries multiple possible prediction keys (adapt to your JSON schema).
    """
    correct = []
    ids = []

    for p in preds:
        mid = p.get("hexdigest")
        if mid is None or mid not in test_lookup:
            continue

        gold_cui = test_lookup[mid][0]

        pred_cui = extract_top1_pred_id(p, internal2orig)
        if pred_cui is None:
            continue

        ids.append(mid)
        correct.append(int(pred_cui == gold_cui))

    return np.array(correct, dtype=np.int32), ids


def unwrap_top1(x: Any) -> Any:
    """
    Unwrap nested containers to get a scalar top-1 value.
    Handles patterns like [['6563']], ['6563'], [6563], etc.
    """
    while isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    return x


def extract_top1_pred_id(
    p: Dict[str, Any],
    internal2orig: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Extract the model's top-1 predicted entity id from a BELB prediction dict.

    - Handles flat keys like 'y_pred', 'predicted_cui', etc.
    - Unwraps nested list/tuple shapes like [['6563']] -> '6563'
    - If internal2orig is provided (UMLS case), maps internal ids -> original CUIs.
    - Handles some common nested patterns: candidates / predictions / candidate_ids / topk
    """

    def maybe_map(v: Any) -> Any:
        v = unwrap_top1(v)
        if v is None:
            return None
        if internal2orig is not None:
            sv = str(v)
            if sv in internal2orig:
                return internal2orig[sv]
        return v

    # 1) Common flat keys (your case: arboel has 'y_pred')
    for k in ["y_pred", "predicted_cui", "predicted", "prediction", "pred", "entity_id", "cui"]:
        if k in p:
            return maybe_map(p[k])

    # 2) Common nested patterns
    # 2.1) {"candidates":[{"cui":"C123",...}, ...]} or {"candidates":[["6563"], ...]}
    if "candidates" in p and isinstance(p["candidates"], list) and len(p["candidates"]) > 0:
        cand0 = unwrap_top1(p["candidates"])
        if isinstance(cand0, dict):
            for k in ["cui", "id", "identifier", "entity_id", "internal_id"]:
                if k in cand0:
                    return maybe_map(cand0[k])
            return maybe_map(cand0)
        return maybe_map(cand0)

    # 2.2) Lists of predictions / ids
    for k in ["predictions", "candidate_ids", "candidates_ids", "topk"]:
        if k in p and isinstance(p[k], list) and len(p[k]) > 0:
            first = unwrap_top1(p[k])
            if isinstance(first, dict):
                for kk in ["cui", "id", "identifier", "entity_id", "internal_id"]:
                    if kk in first:
                        return maybe_map(first[kk])
                return maybe_map(first)
            return maybe_map(first)

    return None


def load_internal_to_original(kb_path: Path) -> Dict[str, str]:
    conn = sqlite3.connect(kb_path)
    cur = conn.execute("SELECT internal_identifier, original_identifier FROM umls_identifier_mapping")
    m = {str(internal): str(original) for internal, original in cur.fetchall()}
    conn.close()
    return m


def correct_map(preds: List[Dict[str, Any]], test_lookup: Dict[str, Tuple[str, str]], internal2orig: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """
    Returns: dict mention_id -> 0/1 correctness
    """
    out = {}
    for p in preds:
        mid = p.get("hexdigest")
        if mid is None or mid not in test_lookup:
            continue
        gold_cui = test_lookup[mid][0]
        pred_cui = extract_top1_pred_id(p, internal2orig)
        if pred_cui is None:
            continue
        out[mid] = int(pred_cui == gold_cui)
    return out


def align_maps(map_a: Dict[str, int], map_b: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    a = np.array([map_a[m] for m in common], dtype=np.int32)
    b = np.array([map_b[m] for m in common], dtype=np.int32)
    return a, b, common


def subset_ids(preds: List[Dict[str, Any]], key: str, value: int) -> set:
    """
    Return mention_ids in preds such that preds[i][key] == value.
    """
    s = set()
    for p in preds:
        mid = p.get("hexdigest")
        if mid is None:
            continue
        if p.get(key) == value:
            s.add(mid)
    return s


def run_significance_analysis(preds_a: List[Dict[str, Any]],
                              preds_b: List[Dict[str, Any]],
                              test_lookup: Dict[str, Tuple[str, str]],
                              label_a: str,
                              label_b: str,
                              characteristics: List[Tuple[str, int]],
                              out_path: Path,
                              internal2orig: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    characteristics: list of (key, value) to define subsets, e.g. ("zero_shot_entity", 1)
    """
    # maps correctness
    map_a = correct_map(preds_a, test_lookup, internal2orig)
    map_b = correct_map(preds_b, test_lookup, internal2orig)

    a_all, b_all, common_all = align_maps(map_a, map_b)
    results = {
        "models": {"A": label_a, "B": label_b},
        "global": mcnemar_test(a_all, b_all),
        "by_subset": {}
    }
    results["global"]["n_common"] = len(common_all)
    results["global"]["acc_A"] = float(a_all.mean()) if len(a_all) else None
    results["global"]["acc_B"] = float(b_all.mean()) if len(b_all) else None

    # subset tests
    for key, val in characteristics:
        ids_sub = subset_ids(preds_a, key, val) & subset_ids(preds_b, key, val)
        ids_sub = sorted(ids_sub & set(common_all))
        if len(ids_sub) == 0:
            continue

        a = np.array([map_a[i] for i in ids_sub if i in map_a and i in map_b], dtype=np.int32)
        b = np.array([map_b[i] for i in ids_sub if i in map_a and i in map_b], dtype=np.int32)
        if len(a) == 0:
            continue

        r = mcnemar_test(a, b)
        r["n_common"] = int(len(a))
        r["acc_A"] = float(a.mean())
        r["acc_B"] = float(b.mean())
        results["by_subset"][f"{key}=={val}"] = r

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, Any]:
    """
    correct_a, correct_b: aligned 0/1 arrays for the same mentions.
    Returns n00,n01,n10,n11 and p-values.

    Uses:
      - exact two-sided binomial test on discordant pairs (recommended, stable)
      - chi-square with continuity correction (approx)
    """
    assert correct_a.shape == correct_b.shape

    a = correct_a
    b = correct_b

    n11 = int(np.sum((a == 1) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    n10 = int(np.sum((a == 1) & (b == 0)))  # A correct, B wrong
    n01 = int(np.sum((a == 0) & (b == 1)))  # A wrong, B correct

    n = n01 + n10  # discordant pairs

    # Exact two-sided McNemar p-value via Binomial test:
    # Under H0, n01 ~ Binomial(n, 0.5). Two-sided uses min(n01,n10).
    p_exact = None
    if n > 0:
        k = min(n01, n10)
        # scipy.stats.binomtest exists in modern scipy; fallback to binom_test for older.
        try:
            p_exact = float(st.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
        except AttributeError:
            p_exact = float(st.binom_test(k, n, 0.5))  # type: ignore

    # Chi-square with continuity correction (Edwards)
    p_chi2 = None
    if n > 0:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n
        p_chi2 = float(st.chi2.sf(chi2, df=1))

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "p_exact": p_exact,
        "p_chi2_cc": p_chi2,
        "n_discordant": int(n),
    }


# Main
def main():
    global args
    parser = argparse.ArgumentParser(description="Annotate BELB predictions with dataset characteristics.")
    parser.add_argument("--corpora", type=str, required=True, help="Corpora to annotate")
    parser.add_argument("--force", action="store_true", help="Force rebuild of the saved results")
    parser.add_argument("--synonymy", type=int, default=10, help="Define the threshold to determine low synonymy")
    parser.add_argument("--length", type=int, default=10, help="Define the threshold to determine long mentions")
    parser.add_argument("--frequency", type=int, default=10, help="Define the threshold to determine low frequency (entities and mentions)")
    parser.add_argument("--variation", type=float, default=0.1, help="Define the threshold to determine lexical variation")
    args = parser.parse_args()

    ### Corpus -> KB and model mappings
    corpus_to_kb = {
        "medmentions": "umls",
        "bc5cdr_chemical": "ctd_chemicals",
        "bc5cdr_disease": "ctd_diseases",
        "bioid_cell_line": "cellosaurus",
        "gnormplus": "ncbi_gene",
        "linnaeus": "ncbi_taxonomy",
        "ncbi_disease": "ctd_diseases",
        "nlm_chem": "ctd_chemicals",
        "nlm_gene": "ncbi_gene",
        "osiris": "dbsnp",
        "s800": "ncbi_taxonomy",
        "snp": "dbsnp",
        "tmvar": "dbsnp"
    }

    corpus_to_model = {
        "medmentions": "scispacy",
        "bc5cdr_chemical": "bc7t2w",
        "bc5cdr_disease": "taggerone",
        "bioid_cell_line": "fuzzysearch",
        "gnormplus": "gnormplus",
        "linnaeus": "sr4gn",
        "ncbi_disease": "taggerone",
        "nlm_chem": "bc7t2w",
        "nlm_gene": "gnormplus",
        "osiris": "tmvar",
        "s800": "sr4gn",
        "snp": "tmvar",
        "tmvar": "tmvar"
    }

    if args.corpora not in corpus_to_kb:
        raise ValueError(f"[ERROR] Unknown corpus: {args.corpora}")
    if args.corpora not in corpus_to_model:
        raise ValueError(f"[ERROR] Unknown corpus: {args.corpora}")

    kb_name = corpus_to_kb[args.corpora]
    kb_table_name = kb_name + "_kb"
    print(f"[INFO] Corpus '{args.corpora}' -> KB '{kb_name}'")

    rb_model = corpus_to_model[args.corpora]
    print(f"[INFO] Corpus '{args.corpora}' -> RB-model '{rb_model}'")

    models = ["arboel", "genbioel", "biosyn", rb_model]

    # For caches:
    metrics_cache_dir = ROOT_DIR / "metrics" / args.corpora
    # For final metrics.json:
    metrics_output_path = ROOT_DIR / "results" / args.corpora / "metrics.json"

    # Load train/test once
    train, test = load_corpus_splits(args.corpora)
    train_cuis, train_surface_forms, train_cui_counter, train_surface_form_counter = build_train_lookup(train, metrics_cache_dir)
    test_lookup = build_test_lookup(test, metrics_cache_dir)

    # Build synonymy lookup
    kb_path = ROOT_DIR.parent / "belb" / "processed" / "kbs" / kb_name / "kb.db"
    internal2orig = None
    # UMLS only (medmentions)
    if kb_name == "umls":
        internal2orig = load_internal_to_original(kb_path)
    if internal2orig is None:
        print("[INFO] No internal2orig mapping (non-UMLS KB).")
    else:
        print(f"[INFO] Loaded internal2orig mapping: {len(internal2orig)} rows")
    synonymy_lookup = build_synonymy_lookup(kb_path, kb_table_name, metrics_cache_dir, test_lookup)

    # Build homonymy lookup
    homonymy_lookup = build_homonymy_lookup(kb_path, kb_table_name, metrics_cache_dir, test_lookup)

    # Load CUI names
    kb_cui_to_names = load_kb_cui_to_names(kb_path, kb_table_name, metrics_cache_dir, test_lookup)

    # PHASE 1 -> Build metrics.json
    print()
    preds_arboel, _ = load_json_file(args.corpora, "arboel")
    metrics_dict = build_metrics_dict(preds_arboel, 
                                  train_cuis, 
                                  train_surface_forms, 
                                  train_cui_counter,
                                  train_surface_form_counter,
                                  test_lookup, 
                                  synonymy_lookup, 
                                  homonymy_lookup,
                                  kb_cui_to_names)
    
    # Save metrics.json
    print(f"[INFO] Saving metrics.json to: {metrics_output_path}")
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f"[DONE] metrics.json saved!")

    # PHASE 2 -> inject into each model
    print()
    print("[INFO] Injecting metrics into each model...")
    for model_name in models:
        model_dir = ROOT_DIR / "results" / args.corpora / model_name
        print()
        if not os.path.exists(model_dir):
            print(f"[WARN] Skipping model '{model_name}' -> directory not found: {model_dir}")
            continue
        print(f"[INFO] Processing model: {model_name}")

        preds, base_dir = load_json_file(args.corpora, model_name)

        for pred in preds:
            mention_id = pred["hexdigest"]
            if mention_id in metrics_dict:
                pred.update(metrics_dict[mention_id])
            else:
                print(f"[WARN] Mention ID {mention_id} not in metrics_dict -> skipping!")

        save_json_file(preds, base_dir)
        print(f"[OK] Annotated predictions saved for model '{model_name}'")

    print()
    print(f"[DONE] All models processed for corpora {args.corpora}")

    # PHASE 3 -> significance testing
    print()
    print("[INFO] Running significance testing (McNemar)...")

    # IMPORTANT: use annotated predictions (with injected metrics) if you want per-bin analysis
    # So load annotated files instead of filtered_predictions.json:
    def load_annotated(corpus_name: str, model_name: str):
        base_dir = ROOT_DIR / "results" / corpus_name / model_name
        path = base_dir / "annotated_filtered_predictions.json"
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] Missing annotated file for {model_name}: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    preds_arboel_ann = load_annotated(args.corpora, "arboel")
    preds_genbioel_ann = load_annotated(args.corpora, "genbioel")
    preds_rbes_ann = load_annotated(args.corpora, rb_model)

    characteristics = [
        ("zero_shot_entity", 1),
        ("zero_shot_surface_form", 1),
        ("mention_length_discrete", 1),
        ("lexical_variation_discrete", 1),
        ("synonymy_difficulty", 1),
        ("homonymy_difficulty", 1),
        ("mention_frequency_difficulty", 1),
        ("entity_frequency_difficulty", 1),
    ]

    print("[DEBUG] Example keys in arboel pred:", list(preds_arboel_ann[0].keys()))
    print("[DEBUG] Example extract_top1_pred_id:", extract_top1_pred_id(preds_arboel_ann[0], internal2orig))
    print("[DEBUG] Example gold:", test_lookup[preds_arboel_ann[0]["hexdigest"]][0])

    sig_out = ROOT_DIR / "results" / args.corpora / "significance_genbioel_vs_rbes.json"
    sig = run_significance_analysis(
        preds_a=preds_genbioel_ann,
        preds_b=preds_rbes_ann,
        test_lookup=test_lookup,
        label_a="genbioel",
        label_b="rbes",
        characteristics=characteristics,
        out_path=sig_out,
        internal2orig=internal2orig
    )

    print(f"[DONE] Significance results saved to: {sig_out}")
    print(f"[INFO] Global: p_exact={sig['global']['p_exact']:.3g}, "
          f"n_discordant={sig['global']['n_discordant']}, "
          f"acc_A={sig['global']['acc_A']:.4f}, acc_B={sig['global']['acc_B']:.4f}")

    # Quick automatic summary: list subsets
    significant = []
    for name, r in sig["by_subset"].items():
        significant.append((name, r["p_exact"], r["acc_A"], r["acc_B"], r["n_discordant"], r["n_common"]))
    significant.sort(key=lambda x: x[1])

    if significant:
        print("[INFO] Significant subsets (p_exact < 0.05):")
        for name, p, accA, accB, nd, nc in significant[:20]:
            print(f"  - {name}: p={p:.3g}, acc_A={accA:.4f}, acc_B={accB:.4f}, discordant={nd}, n={nc}")
    else:
        print("[INFO] No subset reached p_exact < 0.05 (exact McNemar).")


# Entry
if __name__ == "__main__":
    # Example usage from belb-exp:
    # python3 metrics/annotate_preds.py --corpora medmentions --force
    # python3 metrics/annotate_preds.py --corpora linnaeus
    # python3 metrics/annotate_preds.py --corpora s800 --synonymy 5 --variation 0.2

    main()