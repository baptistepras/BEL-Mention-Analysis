import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
import Levenshtein

import annotate_preds
from annotate_preds import build_train_lookup, build_test_lookup, load_kb_cui_to_names, load_corpus_splits
from annotate_preds import build_synonymy_lookup, build_homonymy_lookup, compute_homonyms_for_cui

import warnings
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore", message=".*character detection dependency.*")
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
args=None


def analyze_dataset(train_cuis: set, 
                    train_surface_forms: set, 
                    train_cui_counter: Dict[str, int],
                    train_surface_form_counter: Dict[str, int],
                    test_lookup: Dict[str, Tuple[str, str]],
                    synonymy_lookup: Dict[str, int],
                    homonymy_lookup: Dict[str, int],
                    kb_cui_to_names: Dict[str, List[str]],
                    corpus:str) -> Dict[str, Dict]:
    print(f"[INFO] Analyzing dataset: {corpus}")
    global args

    # Flatten all annotations
    mentions = list(test_lookup.values())
    total = len(mentions)
    print(f"[INFO] Found {total} mentions in test set")

    long_mentions = 0
    low_synonyms = 0
    high_lexical_var = 0
    homonymous = 0
    rare_mentions = 0
    rare_entities = 0
    zero_shot_surface = 0
    zero_shot_entity = 0

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for gold_cui, surface_form in tqdm(mentions, desc=f"[INFO] Processing all mentions", file=sys.stdout):
        # zero-shot entity
        zero_shot_entity += 1 if gold_cui not in train_cuis else 0

        # zero-shot surface form
        zero_shot_surface += 1 if surface_form not in train_surface_forms else 0

        # mention length)
        tokens = tokenizer.tokenize(surface_form)
        mention_length = len(tokens)
        long_mentions += 1 if mention_length > args.lenght else 0

        # synonymy
        num_synonyms = synonymy_lookup.get(gold_cui, 0)
        low_synonyms += 1 if num_synonyms <= args.synonymy else 0

        # homonymy
        num_homonyms = homonymy_lookup.get(gold_cui, 0)
        homonymous += 1 if num_homonyms >= 1 else 0

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

        high_lexical_var += 1 if lexical_variation > args.variation else 0

        # mention frequency
        mention_frequency = train_surface_form_counter.get(surface_form, 0)
        rare_mentions += 1 if mention_frequency <= args.frequency else 0

        # entity frequency
        entity_frequency = train_cui_counter.get(gold_cui, 0)
        rare_entities += 1 if entity_frequency <= args.frequency else 0

    print()
    print("===== TESTSET CHARACTERISTICS =====")
    def ratio(x): return f"{x}/{total} ({100 * x / total:.1f}%)"
    print(f"mention_length>10: {ratio(long_mentions)}")
    print(f"num_synonyms<=10: {ratio(low_synonyms)}")
    print(f"homonymy>=1: {ratio(homonymous)}")
    print(f"lexical_variation>0.1: {ratio(high_lexical_var)}")
    print(f"mention_frequency<=10: {ratio(rare_mentions)}")
    print(f"entity_frequency<=10: {ratio(rare_entities)}")
    print(f"zero_shot_surface_form: {ratio(zero_shot_surface)}")
    print(f"zero_shot_entity: {ratio(zero_shot_entity)}")


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", type=str, required=True, help="Corpora to analyze")
    parser.add_argument("--force", action="store_true", help="Force rebuild of the saved results")
    parser.add_argument("--synonymy", type=int, default=10, help="Define the threshold to determine low synonymy")
    parser.add_argument("--lenght", type=int, default=10, help="Define the threshold to determine long mentions")
    parser.add_argument("--frequency", type=int, default=10, help="Define the threshold to determine low frequency (entities and mentions)")
    parser.add_argument("--variation", type=float, default=0.1, help="Define the threshold to determine lexical variation")
    args = parser.parse_args()
    annotate_preds.args = args

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

    if args.corpora not in corpus_to_kb:
        raise ValueError(f"[ERROR] Unknown corpus: {args.corpora}")

    kb_name = corpus_to_kb[args.corpora]
    kb_table_name = kb_name + "_kb"
    print(f"[INFO] Corpus '{args.corpora}' -> KB '{kb_name}'")

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


    analyze_dataset(train_cuis=train_cuis, 
                    train_surface_forms=train_surface_forms, 
                    train_cui_counter=train_cui_counter,
                    train_surface_form_counter=train_surface_form_counter,
                    test_lookup=test_lookup,
                    synonymy_lookup=synonymy_lookup,
                    homonymy_lookup=homonymy_lookup,
                    kb_cui_to_names=kb_cui_to_names,
                    corpus=args.corpora)
    

if __name__ == "__main__":
    # Example usage from belb-exp:
    # python3 metrics/analyze_datasets.py --corpora medmentions --force
    # python3 metrics/analyze_datasets.py --corpora linnaeus
    # python3 metrics/analyze_datasets.py --corpora s800 --synonymy 5 --variation 0.2

    main()
