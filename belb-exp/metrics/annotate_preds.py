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
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import Levenshtein
from transformers import AutoTokenizer
from pathlib import Path
import sqlite3

# Globals
NORMALIZED_LEVENSHTEIN_THRESHOLD = 0.1
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
args = None


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
        for cui1, names1 in tqdm(cui_to_names.items(), desc="[tqdm] Processing CUIs for homonymy")
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
    print("[INFO] Building metrics dict...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        mention_length_discrete = 1 if mention_length > 10 else 0

        # synonymy
        num_synonyms = synonymy_lookup.get(gold_cui, 0)
        synonymy_difficulty = 1 if num_synonyms <= 10 else 0

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

        lexical_variation_discrete = 1 if lexical_variation > 0.1 else 0

        # mention frequency
        mention_frequency = train_surface_form_counter.get(surface_form, 0)
        mention_frequency_difficulty = 1 if mention_frequency <= 10 else 0

        # entity frequency
        entity_frequency = train_cui_counter.get(gold_cui, 0)
        entity_frequency_difficulty = 1 if entity_frequency <= 10 else 0

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


# Main
def main():
    global args
    parser = argparse.ArgumentParser(description="Annotate BELB predictions with dataset characteristics.")
    parser.add_argument("--corpora", type=str, required=True, help="Corpora to annotate")
    parser.add_argument("--force", action="store_true", help="Force rebuild of the saved results")
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


# Entry
if __name__ == "__main__":
    # Example usage from belb-exp:
    # python3 metrics/annotate_preds.py --corpora medmentions --force
    # python3 metrics/annotate_preds.py --corpora linnaeus

    main()