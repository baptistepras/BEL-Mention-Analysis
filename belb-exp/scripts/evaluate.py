#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get results from entity-specific rule-based systems
"""

import argparse
import os
import random

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

from belb import (ENTITY_TO_CORPORA_NAMES, ENTITY_TO_KB_NAME, AutoBelbCorpus,
                  AutoBelbKb, BelbKb, Entities, Splits)
from belb.resources import Corpora  # type: ignore
from belb.utils import load_stratified, load_zeroshot  # type: ignore

from benchmark.model import CORPORA_MULTI_ENTITY_TYPES, NIL
from benchmark.utils import load_json

EVAL_MODES = ["std", "strict", "lenient"]


CORPORA = [
    # (Corpora.GNORMPLUS.name, Entities.GENE),
    # (Corpora.NLM_GENE.name, Entities.GENE),
    # (Corpora.NCBI_DISEASE.name, Entities.DISEASE),
    # (Corpora.BC5CDR.name, Entities.DISEASE),  # ZIP corrompu
    # (Corpora.BC5CDR.name, Entities.CHEMICAL),  # ZIP corrompu
    # (Corpora.NLM_CHEM.name, Entities.CHEMICAL),
    (Corpora.LINNAEUS.name, Entities.SPECIES),
    (Corpora.S800.name, Entities.SPECIES),
    # (Corpora.BIOID.name, Entities.CELL_LINE), # TAR corrompu
    (Corpora.MEDMENTIONS.name, Entities.UMLS),
    # (Corpora.SNP.name, Entities.VARIANT),  # Demande dbSNP
    # (Corpora.OSIRIS.name, Entities.VARIANT),  # Demande dbSNP
    # (Corpora.TMVAR.name, Entities.VARIANT),  # Demande dbSNP
]

ENTITY_TYPE_STRING_IDENTIFIERS = [
    Entities.DISEASE,
    Entities.CHEMICAL,
    Entities.CELL_LINE,
    Entities.UMLS,
    Entities.GENE,
]

CORPUS_TO_RBES = {
    "gnormplus": "gnormplus",
    "nlm_gene": "gnormplus",
    "linnaeus": "sr4gn",
    "s800": "sr4gn",
    "ncbi_disease": "taggerone",
    "bc5cdr_disease": "taggerone",
    "bc5cdr_chemical": "bc7t2w",
    "nlm_chem": "bc7t2w",
    "medmentions": "scispacy",
    "snp": "tmvar",
    "osiris": "tmvar",
    "tmvar": "tmvar",
    "bioid_cell_line": "fuzzysearch",
}

RBES_JOINT = ["gnormplus", "taggerone", "tmvar", "sr4gn"]

BINNING_CONFIG = {
    "mention_length": None,
    "num_synonyms": None,
    "num_homonyms": 20,
    "lexical_variation": 20,
    "mention_frequency": None,
    "entity_frequency": None
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate results")
    parser.add_argument(
        "--belb_dir",
        type=str,
        required=True,
        help="Directory where all BELB data is stored",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Ranks to consider in prediction",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=tuple(EVAL_MODES),
        default="std",
        help="If multiple predictions are return consider it wrong",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Do not include joint ner-nen models in comparison (full test set)",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="If set, will give advanced metrics"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, plot continuous characteristics"
    )
    return parser.parse_args()


def multi_label_recall(gold: dict, pred: dict, k: int = 1, mode: str = "std") -> float:
    hits = 0

    for h, y_true in gold.items():
        # int
        y_true = set(int(y) for y in y_true)

        # get topk predictions
        y_pred = [list(set(yp)) for yp in pred[h][:k]]

        if mode in ["std", "strict"]:
            # get single prediction
            if mode == "strict":
                # in strict mode default wrong if multiple predictions
                y_pred = [NIL if len(y) > 1 else y[0] for y in y_pred]
            elif mode == "std":
                # sample if multiple predictions
                y_pred = [random.sample(y, 1)[0] for y in y_pred]

            # go over k predicitons
            for y in y_pred:
                # int
                y = -1 if y == NIL else int(y)
                if y in y_true:
                    hits += 1
                    # if you get a hit stop
                    break
        else:
            for ys in y_pred:
                ys = [-1 if y == NIL else int(y) for y in ys]
                if any(y in y_true for y in ys):
                    hits += 1
                    # if you get a hit stop
                    break

    return round(hits / len(gold), 2)


def get_integer_identifiers(kb: BelbKb, gold: dict) -> dict:
    identifiers = set(i for ids in gold.values() for i in ids)

    with kb as handle:
        map = handle.get_identifier_mapping(identifiers)

    gold = {h: set([map[i] for i in ids]) for h, ids in gold.items()}

    return gold


def load_gold(
    corpus_name: str,
    entity_type: str,
    belb_dir: str,
    db_config,
) -> dict:
    corpus = AutoBelbCorpus.from_name(
        name=corpus_name,
        directory=belb_dir,
        entity_type=entity_type,
        sentences=True,
        mention_markers=False,
        add_foreign_annotations=False,
    )

    gold = {
        a.infons["hexdigest"]: a.identifiers
        for e in corpus[Splits.TEST]
        for p in e.passages
        for a in p.annotations
    }

    if entity_type in ENTITY_TYPE_STRING_IDENTIFIERS:
        kb = AutoBelbKb.from_name(
            directory=belb_dir,
            name=ENTITY_TO_KB_NAME[entity_type],
            db_config=db_config,
            debug=False,
        )

        gold = get_integer_identifiers(kb=kb, gold=gold)

    return gold


def filter_gold(gold: dict, directory: str, corpus_name: str) -> dict:
    ner_tp = None
    if corpus_name in CORPUS_TO_RBES:
        rbes_pred_path = os.path.join(
            directory,
            CORPUS_TO_RBES[corpus_name],
            "filtered_predictions.json",
        )
        if os.path.exists(rbes_pred_path):
            rbes_pred = load_json(rbes_pred_path)
            ner_tp = set(p["hexdigest"] for p in rbes_pred)

    if ner_tp is not None:
        gold = {h: y for h, y in gold.items() if h in ner_tp}

    return gold


def get_results_by_corpus(
    gold: dict, preds: dict, mode: str = "std", k: int = 1
) -> pd.DataFrame:
    # TODO: add document level
    data: dict = {}

    for corpus, corpus_gold in gold.items():
        for model, corpora_pred in preds.items():
            corpus_pred = corpora_pred.get(corpus, [])

            if len(corpus_pred) == 0:
                continue

            if model not in data:
                data[model] = {}

            try:
                data[model][corpus] = multi_label_recall(
                    gold=corpus_gold, pred=corpus_pred, mode=mode, k=k
                )
            except KeyError as e:
                print(f"[WARN] Missing key {e} in predictions for corpus '{corpus}' and model '{model}'")
                continue


    return pd.DataFrame(data)


def get_results_by_entity(
    gold: dict, preds: dict, mode: str = "std", k: int = 1
) -> pd.DataFrame:
    data: dict = {}
    for entity, corpora in ENTITY_TO_CORPORA_NAMES.items():
        corpora = [
            f"{c}_{entity}" if c in CORPORA_MULTI_ENTITY_TYPES else c for c in corpora
        ]

        entity_gold = {
            h: y
            for corpus, corpus_gold in gold.items()
            for h, y in corpus_gold.items()
            if corpus in corpora
        }

        for model, corpora_pred in preds.items():
            entity_pred = {
                h: y
                for corpus, corpus_pred in corpora_pred.items()
                for h, y in corpus_pred.items()
                if corpus in corpora
            }

            if len(entity_pred) == 0:
                continue

            if model not in data:
                data[model] = {}

            try:
                data[model][entity] = multi_label_recall(
                    gold=entity_gold, pred=entity_pred, mode=mode, k=k
                )
            except KeyError as e:
                print(f"[WARN] Missing key {e} in model {model} for entity {entity}")
                continue

    return pd.DataFrame(data)


def get_results_by_subset(
    gold: dict, preds: dict, mode: str = "std", k: int = 1
) -> dict:
    # zeroshot = load_zeroshot()
    # zeroshot_entity_df = zeroshot[zeroshot["is_zeroshot_entity"]]
    # zeroshot_name_df = zeroshot[zeroshot["is_zeroshot_name"]]
    subsets = {
        # "zeroshot_entity": zeroshot_entity_df,
        #"zeroshot_name": zeroshot_name_df,
        "zeroshot": load_zeroshot(),
        "stratified": load_stratified(),
        # "homonyms": load_homonyms(),
    }

    out = {}

    for subset_name, subset_df in subsets.items():
        data: dict = {}
        for entity_type in Entities:
            subset = set(
                subset_df[subset_df["entity_type"] == entity_type]["hexdigest"]
            )
            if len(subset) == 0:
                continue

            subset_gold = {
                h: y
                for name, corpus_gold in gold.items()
                for h, y in corpus_gold.items()
                if h in subset
            }

            for model, corpora_pred in preds.items():
                subset_pred = {
                    h: y
                    for _, corpus_pred in corpora_pred.items()
                    for h, y in corpus_pred.items()
                    if h in subset_gold
                }

                if len(subset_gold) != len(subset_pred):
                    continue

                if entity_type not in data:
                    data[entity_type] = {}
                    
                if len(subset_gold) == 0:
                    continue

                try:
                    data[entity_type][model] = multi_label_recall(
                        gold=subset_gold, pred=subset_pred, mode=mode, k=k
                    )
                except KeyError as e:
                    print(f"[WARN] Missing key {e} in subset {subset_name} for model {model} and entity {entity_type}")
                    continue    

        out[subset_name] = pd.DataFrame(data)

    return out


def get_results_by_characteristic(
    gold: dict,
    preds: dict,
    characteristic_name: str,
    mode: str = "std",
    k: int = 1
) -> pd.DataFrame:
    data: dict = {}

    for model, corpora_pred in preds.items():
        # We want to accumulate all mentions across all corpora (global evaluation)
        char_to_mentions = {}

        for corpus, corpus_gold in gold.items():
            corpus_pred = corpora_pred.get(corpus, {})

            if len(corpus_pred) == 0:
                continue

            # Recover the correct directory name for this model on this corpus
            if model == "rbes":
                model_dir = CORPUS_TO_RBES.get(corpus.split("_")[0], None)
                if model_dir is None:
                    print(f"[ERROR] No RBES mapping found for corpus {corpus}")
                    continue
            else:
                model_dir = model

            # Path to the corresponding annotated_filtered_predictions.json
            annotated_path = os.path.join(
                os.getcwd(),
                "results",
                corpus,
                model_dir,
                "annotated_filtered_predictions.json"
            )

            if not os.path.exists(annotated_path):
                print(f"[WARN] No annotated predictions found for {model} on {corpus}")
                continue

            annotated_preds = load_json(annotated_path)

            # Group mention IDs by the value of the characteristic (global across corpora)
            for p in annotated_preds:
                h = p["hexdigest"]
                if h not in corpus_gold:
                    continue  # Ignore mentions that are not in gold

                if characteristic_name not in p:
                    continue

                char_value = p[characteristic_name]

                if char_value not in char_to_mentions:
                    char_to_mentions[char_value] = set()

                char_to_mentions[char_value].add((corpus, h))  # Store both corpus + mention ID

        # Now compute global recall for each characteristic value
        for char_value, mentions in char_to_mentions.items():
            subset_gold = {}
            subset_pred = {}

            for corpus, h in mentions:
                corpus_gold = gold[corpus]
                corpus_pred = corpora_pred.get(corpus, {})

                if h in corpus_gold:
                    subset_gold[h] = corpus_gold[h]
                if h in corpus_pred:
                    subset_pred[h] = corpus_pred[h]

            if len(subset_gold) == 0:
                continue

            if char_value not in data:
                data[char_value] = {}

            try:
                data[char_value][model] = multi_label_recall(
                    gold=subset_gold,
                    pred=subset_pred,
                    mode=mode,
                    k=k
                )
            except Exception as e:
                print(f"[ERROR] Error on characteristic {char_value}, model {model}: {e}")
                continue

    # Return a DataFrame with characteristic values as rows
    return pd.DataFrame(data).T, char_to_mentions


def save_characteristic_table(char_df, char_name, args):
    # Create output dir
    output_dir = os.path.join(os.getcwd(), "results", "tables", "characteristics")
    os.makedirs(output_dir, exist_ok=True)

    # Save as TSV
    tsv_path = os.path.join(
        output_dir,
        f"{char_name}_k{args.k}_mode{args.mode}_full{int(args.full)}.tsv"
    )
    char_df.to_csv(tsv_path, sep="\t")

    # Clean print
    print(char_df.to_markdown(tablefmt="github", floatfmt=".2f"))


def plot_performance_by_continuous_characteristic_simple(
    gold: dict,
    preds: dict,
    characteristic_name: str,
    mode: str = "std",
    k: int = 1,
    save_csv: bool = True
):
    print(f"\n==== PLOTTING CHARACTERISTIC (SIMPLE): {characteristic_name} ====")

    # Init dict : valeur -> (score_total, nb_fois)
    dict_of_characteristic = {}

    # Loop over models
    for model, corpora_pred in preds.items():
        # Loop over corpora
        for corpus, corpus_gold in gold.items():
            corpus_pred = corpora_pred.get(corpus, {})

            if len(corpus_pred) == 0:
                continue

            # Recover model dir
            if model == "rbes":
                model_dir = CORPUS_TO_RBES.get(corpus.split("_")[0], None)
                if model_dir is None:
                    print(f"[ERROR] No RBES mapping found for corpus {corpus}")
                    continue
            else:
                model_dir = model

            # Load annotated predictions
            annotated_path = os.path.join(
                os.getcwd(),
                "results",
                corpus,
                model_dir,
                "annotated_filtered_predictions.json"
            )

            if not os.path.exists(annotated_path):
                print(f"[WARN] No annotated predictions found for {model} on {corpus}")
                continue

            annotated_preds = load_json(annotated_path)

            # Loop over mentions
            for p in annotated_preds:
                h = p["hexdigest"]
                if h not in corpus_gold:
                    continue

                if characteristic_name not in p:
                    continue

                try:
                    value = float(p[characteristic_name])
                    if np.isnan(value):
                        continue  # skip nan values
                except (ValueError, TypeError):
                    continue  # skip bad values

                y_true = set(int(y) for y in corpus_gold[h])
                y_pred_topk = [list(set(pred_h)) for pred_h in corpus_pred.get(h, [])][:k]

                correct = 0
                if mode in ["std", "strict"]:
                    y_pred = []
                    for pred_set in y_pred_topk:
                        if mode == "strict":
                            y_pred.append(NIL if len(pred_set) > 1 else pred_set[0])
                        else:
                            y_pred.append(random.sample(pred_set, 1)[0])
                    for y in y_pred:
                        y = -1 if y == NIL else int(y)
                        if y in y_true:
                            correct = 1
                            break
                else:
                    for pred_set in y_pred_topk:
                        pred_set = [-1 if y == NIL else int(y) for y in pred_set]
                        if any(y in y_true for y in pred_set):
                            correct = 1
                            break

                # Update dict
                if value not in dict_of_characteristic:
                    dict_of_characteristic[value] = (0, 0)  # (score_total, nb_fois)

                prev_score, prev_count = dict_of_characteristic[value]
                dict_of_characteristic[value] = (prev_score + correct, prev_count + 1)

    # If empty
    if len(dict_of_characteristic) == 0:
        print(f"[WARN] No data collected for characteristic: {characteristic_name}")
        return

    # Build x/y lists
    x = sorted(dict_of_characteristic.keys())
    y = []
    counts = []

    for val in x:
        score_total, nb_fois = dict_of_characteristic[val]
        mean_perf = float(score_total) / float(nb_fois)
        y.append(mean_perf)
        counts.append(nb_fois)

    x = np.array([float(val) for val in x], dtype=np.float64)
    y = np.array([float(val) for val in y], dtype=np.float64)

    # bins when too much points
    n_bins = BINNING_CONFIG.get(characteristic_name, None)
    if n_bins is not None:
        print(f"[INFO] Applying binning: {n_bins} bins for {characteristic_name}")

        bins = np.linspace(x.min(), x.max(), n_bins + 1)
        bin_indices = np.digitize(x, bins)

        x_binned = []
        y_binned = []
        for i in range(1, n_bins + 1):
            mask = (bin_indices == i)
            if mask.sum() == 0:
                continue
            x_binned.append(x[mask].mean())
            y_binned.append(y[mask].mean())

        # Remplacer x et y par les versions binned
        x = np.array(x_binned)
        y = np.array(y_binned)

    else:
        print(f"[INFO] No binning applied for {characteristic_name}")

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name=characteristic_name
    ))

    fig.update_layout(
        title=f"Performance vs {characteristic_name}",
        xaxis_title=characteristic_name,
        yaxis_title="Mean Recall (k=1)",
        template='plotly_white'
    )

    # Trend line 
    degree = 2
    coeffs = np.polyfit(x, y, degree)
    poly_eq = np.poly1d(coeffs)
    y_poly_pred = poly_eq(x)

    fig.add_trace(go.Scatter(
        x=x, y=y_poly_pred,
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2)
    ))

    # Save
    save_dir = os.path.join(os.getcwd(), "metrics", "plots")
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{characteristic_name}_k{k}_mode{mode}_full0.png")

    pio.write_image(fig, plot_path, format='png', scale=2)
    print(f"[DONE] Plot Saved: {plot_path}")

    # Optionally save CSV
    if save_csv:
        csv_dir = os.path.join(os.getcwd(), "metrics", "plots/tables")
        os.makedirs(csv_dir, exist_ok=True) 

        csv_path = os.path.join(csv_dir, f"{characteristic_name}_k{k}_mode{mode}_full0.csv")    

        with open(csv_path, "w") as f:
            f.write("value,mean_perf\n")
            for val_i in range(len(x)):
                val = x[val_i]
                mean_perf = y[val_i]
                f.write(f"{val},{mean_perf:.4f}\n") 

        print(f"[DONE] CSV Saved: {csv_path}")


def main():
    args = parse_args()

    results_dir = os.path.join(os.getcwd(), "results")

    DB_CONFIG = os.path.join(os.getcwd(), "config", "db.yaml")

    gold = {}
    preds = {}
    for corpus_name, entity_type in CORPORA:
        full_corpus_name = (
            f"{corpus_name}_{entity_type}"
            if corpus_name in CORPORA_MULTI_ENTITY_TYPES
            else corpus_name
        )

        try:
            corpus_gold = load_gold(
                corpus_name=corpus_name,
                entity_type=entity_type,
                belb_dir=args.belb_dir,
                db_config=DB_CONFIG,
            )
        except Exception as e:
            print(f"[WARN] Skipping {corpus_name} due to error: {e}")
            continue

        corpus_dir = os.path.join(results_dir, full_corpus_name)

        if not args.full:
            corpus_gold = filter_gold(
                gold=corpus_gold,
                directory=corpus_dir,
                corpus_name=full_corpus_name,
            )

        for model in os.listdir(corpus_dir):
            model_path = os.path.join(corpus_dir, model)
            if not os.path.isdir(model_path):
                continue
            model_name = "rbes" if model in set(CORPUS_TO_RBES.values()) else model

            if args.full and model_name == "rbes":
                continue

            pred = {
                p["hexdigest"]: p["y_pred"]
                for p in load_json(os.path.join(corpus_dir, model, "predictions.json"))
            }
            pred = {h: y for h, y in pred.items() if h in corpus_gold}

            if model_name not in preds:
                preds[model_name] = {}

            preds[model_name][full_corpus_name] = pred
            gold[full_corpus_name] = corpus_gold
        
    print("CORPORA:")
    corpora_df = get_results_by_corpus(gold=gold, preds=preds, mode=args.mode, k=args.k)
    corpora_df = corpora_df.drop(columns=[c for c in corpora_df.columns if c.endswith("_ar")], errors="ignore")
    print(corpora_df)
    print("\n")
    
    print("ENTITIES:")
    entity_df = get_results_by_entity(
          gold=gold, preds=preds, mode=args.mode, k=args.k
    )
    entity_df = entity_df.drop(columns=[c for c in entity_df.columns if c.endswith("_ar")], errors="ignore")
    print(entity_df)
    print("\n")

    if args.advanced:
        print("CHARACTERISTICS:")
        characteristics_to_compute = [
            "zero_shot_entity",
            "zero_shot_surface_form",
            "mention_length_discrete",
            "synonymy_difficulty",
            "homonymy_difficulty",
            "lexical_variation_discrete",
            "mention_frequency_difficulty",
            "entity_frequency_difficulty"
        ]

        # Legend to explain each characteristic value
        CHARACTERISTIC_LEGENDS = {
            "zero_shot_entity": {
                1: "<-- zero-shot entity",
                0: "<-- seen in train"
            },
            "zero_shot_surface_form": {
                1: "<-- zero-shot surface form",
                0: "<-- seen in train"
            },
            "mention_length_discrete": {
                1: "<-- long mention (T > 10)",
                0: "<-- short mention (T <= 10)"
            },
            "synonymy_difficulty": {
                1: "<-- few synonyms (<= 10)",
                0: "<-- many synonyms (> 10)"
            },
            "homonymy_difficulty": {
                1: "<-- homonyms exist",
                0: "<-- no homonyms"
            },
            "lexical_variation_discrete": {
                1: "<-- high variation (> 0.1)",
                0: "<-- low variation (<= 0.1)"
            },
            "mention_frequency_difficulty": {
                1: "<-- rare mention (<= 10)",
                0: "<-- frequent mention (> 10)"
            },
            "entity_frequency_difficulty": {
                1: "<-- rare entity (<= 10)",
                0: "<-- frequent entity (> 10)"
            }
        }

        for char_name in characteristics_to_compute:
            print(f"\n==== {char_name.upper()} ====")
            char_df, char_to_mentions = get_results_by_characteristic(
                gold=gold,
                preds=preds,
                characteristic_name=char_name,
                mode=args.mode,
                k=args.k
            )

            # Reorder so that 1 is on top, 0 is below
            if 1 in char_df.index and 0 in char_df.index:
                char_df = char_df.loc[[1, 0]]

            # Add legend column
            legend = CHARACTERISTIC_LEGENDS.get(char_name, {})
            char_df["Legend"] = [legend.get(i, "") for i in char_df.index]

            # Add "Weighted Avg" column
            weighted_avgs = []

            for idx, row in char_df.iterrows():
                total_mentions = 0
                weighted_sum = 0

                for model in row.index:
                    if model == "Legend":
                        continue
                    value = row[model]
                    if pd.isna(value):
                        continue
                    
                    # number of mentions is len(char_to_mentions[idx])
                    num_mentions = len(char_to_mentions[idx])
                    total_mentions += num_mentions
                    weighted_sum += value * num_mentions

                if total_mentions == 0:
                    avg = float('nan')
                else:
                    avg = weighted_sum / total_mentions

                weighted_avgs.append(avg)

            # Insert column between last model and Legend
            model_columns = [col for col in char_df.columns if col != "Legend"]
            insertion_index = len(model_columns)

            char_df.insert(insertion_index, "Weighted Avg", weighted_avgs)

            # Print nicely + save
            save_characteristic_table(char_df, char_name, args)

    if args.plot:
        print("\n==== PLOTTING CONTINUOUS CHARACTERISTICS ====")

        continuous_characteristics = [
            "mention_length",
            "num_synonyms",
            "num_homonyms",
            "lexical_variation",
            "mention_frequency",
            "entity_frequency"
        ]

        for char_name in continuous_characteristics:
            plot_performance_by_continuous_characteristic_simple(
                gold=gold,
                preds=preds,
                characteristic_name=char_name,
                mode=args.mode,
                k=args.k
        )
            
    """
    print("SUBSETS")
    subsets_results = get_results_by_subset(
         gold=gold, preds=preds, mode=args.mode, k=args.k
    )
    for name in list(subsets_results.keys()):
        df = subsets_results[name]
        if "cell_line" in df.columns:
            df = df.drop(columns="cell_line")
        if "arboel_ar" in df.index:
            df = df.drop("arboel_ar", axis=0)
        if "genbioel_ar" in df.index:
            df = df.drop("genbioel_ar", axis=0)
        if "biosyn_ar" in df.index:
            df = df.drop("biosyn_ar", axis=0)
        subsets_results[name] = df
    for name, subset_df in subsets_results.items():
         print(f"\t{name.upper()}")
         print(subset_df)

    # print(subsets_results)
    print("\n")
    
    # for result, df in [
    #     ("corpora", corpora_df),
    #     ("entity", entity_df),
    #     ("subsets", subsets_df),
    # ]:
    #     df.to_csv(
    #         os.path.join(
    #             os.getcwd(),
    #             "results",
    #             "tables",
    #             f"{result}_k{args.k}_mode{int(args.mode)}_full{int(args.full)}.tsv",
    #         ),
    #     )
    """
    

if __name__ == "__main__":
    main()
