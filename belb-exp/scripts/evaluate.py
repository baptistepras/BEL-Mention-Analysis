#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get results from entity-specific rule-based systems
"""

import argparse
import os
import random
import json
from pathlib import Path

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
    "mention_length": 15,
    "num_synonyms": 40,
    "num_homonyms": "auto",
    "lexical_variation": 20,
    "mention_frequency": 100,
    "entity_frequency": 100
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
    parser.add_argument(
        "--focus",
        nargs="*",
        default=None,
        help="Characteristics on which we should focus (continuous)",
    )
    parser.add_argument(
        "--others",
        nargs="*",
        default=None,
        help="Characteristics to use for comparison (discrete)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to chose for focus vs others"
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
        if model == "biosyn" and (Corpora.MEDMENTIONS.name, Entities.UMLS) in CORPORA:
                continue  # skip BioSyn when medMentions is activa
        
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
    save_csv: bool = True,
    show: bool = True,
    chosen_model: str = None
):
    # Init dict : valeur -> (score_total, nb_fois)
    dict_of_characteristic = {}

    # Loop over models
    for model, corpora_pred in preds.items():
        if chosen_model is not None and chosen_model != model:
            continue
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
    save_x = x.copy()
    save_y = y.copy()

    # bins when too much points    
    n_bins = BINNING_CONFIG.get(characteristic_name, None)
    if n_bins == "auto":
        n_bins = len(x)
    if n_bins is not None:
        print(f"[INFO] Applying binning: {n_bins} bins for {characteristic_name}")

        bins = np.linspace(x.min(), x.max(), n_bins + 1)
        bin_indices = np.digitize(x, bins)

        x_binned = []
        y_binned = []
        counts_binned = [] 
        for i in range(1, n_bins + 1):
            mask = (bin_indices == i)
            if mask.sum() == 0:
                continue
            x_binned.append(x[mask].mean())
            y_binned.append(y[mask].mean())
            counts_binned.append(sum(counts[j] for j in range(len(x)) if bin_indices[j] == i))

        x = np.array(x_binned)
        y = np.array(y_binned)
        counts = np.array(counts_binned)

    else:
        print(f"[INFO] No binning applied for {characteristic_name}")

    if show:
        # Plot
        fig = go.Figure()

        # Performance vs characteristic
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name='Mean Recall (k=1)',
            line=dict(color='blue', width=2),
            yaxis='y1'
        ))

        # Trend line 
        degree = 2
        coeffs = np.polyfit(x, y, degree)
        poly_eq = np.poly1d(coeffs)
        y_poly_pred = poly_eq(x)

        fig.add_trace(go.Scatter(
            x=x,
            y=y_poly_pred,
            mode='lines',
            name='Trend (poly deg 2)',
            line=dict(color='red', width=2, dash='dash'),
            yaxis='y1'
        ))

        # Number of examples per characteristic
        fig.add_trace(go.Scatter(
            x=x,
            y=counts,
            mode='lines+markers',
            name='Number of Mentions',
            line=dict(color='grey', width=1),
            opacity=0.5,
            yaxis='y2'
        ))

        # Layout with secondary y-axis
        fig.update_layout(
            title=f"Performance vs {characteristic_name}",
            xaxis_title=characteristic_name,
            yaxis=dict(
                title="Mean Recall (k=1)",
                side='left',
                showgrid=False
            ),
            yaxis2=dict(
                title='Number of Mentions',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(
                orientation="v",
                x=1.02,
                y=1,
                bordercolor="Black",
                borderwidth=0.5
            ),
            template='plotly_white'
        )

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

    return save_x, save_y


def aggregate_annotations():
    print(f"[INFO] Aggregating all predictions from results/.")
    results_dir = os.path.join(os.getcwd(), "results")
    output_path = os.path.join("metrics", "tmp", "aggregated_annotations.json")
    existing_path = Path("metrics/tmp/aggregated_annotations.json")

    if existing_path.exists():
        print(f"[INFO] Aggregated file already exists at {existing_path}. Skipping aggregation.")
        return

    all_annotations = []

    for corpus in os.listdir(results_dir):
        corpus_path = os.path.join(results_dir, corpus)
        if not os.path.isdir(corpus_path):
            continue

        for model in os.listdir(corpus_path):
            model_path = os.path.join(corpus_path, model)
            if not os.path.isdir(model_path):
                continue

            json_path = os.path.join(model_path, "annotated_filtered_predictions.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[ERROR] JSON decode error in {json_path}")
                    continue

            for p in data:
                p["corpus"] = corpus
                p["model"] = model
                all_annotations.append(p)

    os.makedirs(os.path.join("metrics", "tmp"), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_annotations, f, indent=2)

    print(f"[DONE] Aggregated {len(all_annotations)} mentions from results/ -> {output_path}")


def plot_focus_vs_others(x, y, focus_char, continuous_char_to_discrete_char, preds, save_path):
    """
    Plot recall for a given continuous characteristic, and show average value of discrete characteristics at each bin.
    """
    # Deduce the discrete version of the focus char to skip
    discrete_focus_char = continuous_char_to_discrete_char[focus_char]

    # Discrete characteristics to display (except the one mapped from focus_char)
    discrete_chars = {c: {} for c in list(continuous_char_to_discrete_char.values()) if c != discrete_focus_char}

    # Binning
    n_bins = BINNING_CONFIG[focus_char]
    if n_bins == "auto":
        n_bins = len(x)
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_indices = np.digitize(x, bins)

    x_binned = []
    y_binned = []
    bin_to_mentions = {}
    bin_idx_to_center_x = {}

    for i in range(1, n_bins + 1):
        mask = (bin_indices == i)
        if mask.sum() == 0:
            continue

        x_center = x[mask].mean()
        y_center = y[mask].mean()

        x_binned.append(x_center)
        y_binned.append(y_center)

        bin_idx_to_center_x[i] = x_center

        mentions = []
        for p in preds:
            if focus_char in p:
                try:
                    val = float(p[focus_char])
                    if bins[i - 1] <= val < bins[i] or (i == n_bins and val == bins[-1]):
                        mentions.append(p)
                except:
                    continue
        bin_to_mentions[i] = mentions

    x = np.array(x_binned)
    y = np.array(y_binned)

    # Mean value of each discrete char for each bin
    for char in discrete_chars:
        for i, mentions in bin_to_mentions.items():
            vals = [float(p[char]) for p in mentions if char in p and isinstance(p[char], (int, float))]
            mean_val = np.mean(vals) if vals else np.nan
            x_val = bin_idx_to_center_x[i]
            discrete_chars[char][x_val] = mean_val


    # Plot
    fig = go.Figure()

    # Focus characteristic
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Mean Recall (k=1)',
        line=dict(color='blue', width=2),
        yaxis='y1'
    ))

    # Discrete characteristics
    colors = ['blue', 'darkgreen', 'darkred', 'purple', 'brown', 'black']
    for i, (char, values_dict) in enumerate(discrete_chars.items()):
        x_vals = list(values_dict.keys())
        y_vals = list(values_dict.values())

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name=char,
            line=dict(color=colors[i % len(colors)], width=1),
            opacity=0.4,
            yaxis='y2'
        ))

    fig.update_layout(
        title=f"Performance vs {focus_char} (with discrete characteristics)",
        xaxis=dict(title=focus_char),
        yaxis=dict(title='Recall (k=1)', side='left'),
        yaxis2=dict(
            title='Mean Discrete Char Value',
            side='right',
            overlaying='y',
            range=[0, 1]
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            x=0,
            y=-0.3,
            xanchor="left",
            yanchor="top",
            bordercolor="Black",
            borderwidth=0.5
        )
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_image(save_path)
    print(f"[DONE] Focus plot saved -> {save_path}")


def plot_focus_vs_chosen_others(x, y, focus_char, continuous_char_to_discrete_char, preds, save_path):
    """
    Plot recall for a given continuous characteristic, and show average value of discrete characteristics at each bin.
    """
    # Deduce the discrete version of the focus char to skip
    discrete_focus_char = continuous_char_to_discrete_char[focus_char]

    # Discrete characteristics to display (except the one mapped from focus_char)
    discrete_chars = {c: {} for c in list(continuous_char_to_discrete_char.values()) if c != discrete_focus_char}

    # Binning
    n_bins = BINNING_CONFIG[focus_char]
    if n_bins == "auto":
        n_bins = len(x)
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_indices = np.digitize(x, bins)

    x_binned = []
    y_binned = []
    bin_to_mentions = {}
    bin_idx_to_center_x = {}

    for i in range(1, n_bins + 1):
        mask = (bin_indices == i)
        if mask.sum() == 0:
            continue

        x_center = x[mask].mean()
        y_center = y[mask].mean()

        x_binned.append(x_center)
        y_binned.append(y_center)

        bin_idx_to_center_x[i] = x_center

        mentions = []
        for p in preds:
            if focus_char in p:
                try:
                    val = float(p[focus_char])
                    if bins[i - 1] <= val < bins[i] or (i == n_bins and val == bins[-1]):
                        mentions.append(p)
                except:
                    continue
        bin_to_mentions[i] = mentions

    x = np.array(x_binned)
    y = np.array(y_binned)

    # Mean value of each discrete char for each bin
    for char in discrete_chars:
        for i, mentions in bin_to_mentions.items():
            vals = [float(p[char]) for p in mentions if char in p and isinstance(p[char], (int, float))]
            mean_val = np.mean(vals) if vals else np.nan
            x_val = bin_idx_to_center_x[i]
            discrete_chars[char][x_val] = mean_val


    # Plot
    fig = go.Figure()

    # Focus characteristic
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Mean Recall (k=1)',
        line=dict(color='blue', width=2),
        yaxis='y1'
    ))

    # Discrete characteristics
    colors = ['blue', 'darkgreen', 'darkred', 'purple', 'brown', 'black']
    for i, (char, values_dict) in enumerate(discrete_chars.items()):
        x_vals = list(values_dict.keys())
        y_vals = list(values_dict.values())

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name=char,
            line=dict(color=colors[i % len(colors)], width=1),
            opacity=0.4,
            yaxis='y2'
        ))

    fig.update_layout(
        title=f"Performance vs {focus_char} (with discrete characteristics)",
        xaxis=dict(title=focus_char),
        yaxis=dict(title='Recall (k=1)', side='left'),
        yaxis2=dict(
            title='Mean Discrete Char Value',
            side='right',
            overlaying='y',
            range=[0, 1]
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            x=0,
            y=-0.3,
            xanchor="left",
            yanchor="top",
            bordercolor="Black",
            borderwidth=0.5
        )
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_image(save_path)
    print(f"[DONE] Focus plot saved -> {save_path}")


def plot_characteristics_correlation_matrix():
    """
    Compute and plot Spearman correlation matrix between continuous and discrete characteristics
    from the aggregated annotated predictions.
    """
    # Load enriched annotations
    with open("metrics/tmp/aggregated_annotations.json", "r") as f:
        data = json.load(f)

    # List of continuous and discrete characteristics to analyze
    selected_columns = [
        "mention_length", "num_synonyms", "num_homonyms",
        "lexical_variation", "mention_frequency", "entity_frequency",
        "mention_length_discrete", "synonymy_difficulty", "homonymy_difficulty",
        "lexical_variation_discrete", "mention_frequency_difficulty", "entity_frequency_difficulty",
        "zero_shot_entity", "zero_shot_surface_name"
    ]

    # Build DataFrame with only selected columns
    df = pd.DataFrame([{k: p.get(k, None) for k in selected_columns} for p in data])

    # Convert all values to numeric (ignore errors silently)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop incomplete rows
    df = df.dropna()

    # If after cleaning there's nothing left, alert
    if df.empty:
        print("[ERROR] No data available after cleaning. Correlation matrix not computed.")
        return

    # Compute Spearman correlation
    corr_matrix = df.corr(method="spearman")

    # Save to CSV
    output_path = "metrics/plots/characteristics_correlation_matrix.csv"
    corr_matrix.to_csv(output_path)
    print(f"[DONE] Correlation matrix saved to {output_path}")


def main():
    args = parse_args()
    if (args.others is None) ^ (args.focus is None):
        raise ValueError("You must specify both --focus and --others (or neither).")
    if (args.model is not None) and (args.model not in ["rbes", "arboel", "genbioel"]):
        raise ValueError("You must specify a valid model (rbes, arboel or genbioel).")

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

    if args.advanced:
        print("CHARACTERISTICS:")

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
                    
                    # Count how many mentions this model actually produced for this bin
                    model_mentions = [
                        (corpus, h)
                        for (corpus, h) in char_to_mentions[idx]
                        if h in preds[model].get(corpus, {})
                    ]

                    num_mentions = len(model_mentions)

                    if num_mentions == 0:
                        continue  # Do not contribute to weighted sum / total_mentions
                    
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
        aggregate_annotations()
        print()

        continuous_characteristics = [
            "mention_length",
            "num_synonyms",
            "num_homonyms",
            "lexical_variation",
            "mention_frequency",
            "entity_frequency"
        ]

        for char_name in continuous_characteristics:
            print(f"[INFO] Plotting continuous characteritics for {char_name}.")
            x, y = plot_performance_by_continuous_characteristic_simple(
                gold=gold,
                preds=preds,
                characteristic_name=char_name,
                mode=args.mode,
                k=args.k
            )

            print(f"[INFO] Plotting focus vs others for {char_name}.")
            plot_focus_vs_others(
                x=x, y=y,
                focus_char=char_name,
                continuous_char_to_discrete_char={
                    "mention_length": "mention_length_discrete",
                    "num_synonyms": "synonymy_difficulty",
                    "num_homonyms": "homonymy_difficulty",
                    "lexical_variation": "lexical_variation_discrete",
                    "mention_frequency": "mention_frequency_difficulty",
                    "entity_frequency": "entity_frequency_difficulty"
                },
                preds=load_json("metrics/tmp/aggregated_annotations.json"),
                save_path=f"metrics/plots/focus_{char_name}_vs_others.png"
                )
            print()

        # print(f"[INFO] Plotting correlation matrix.")
        # plot_characteristics_correlation_matrix()
        # print()

    if args.focus is not None and args.others is not None:
        continuous_char_to_discrete_char={
                "mention_length": "mention_length_discrete",
                "num_synonyms": "synonymy_difficulty",
                "num_homonyms": "homonymy_difficulty",
                "lexical_variation": "lexical_variation_discrete",
                "mention_frequency": "mention_frequency_difficulty",
                "entity_frequency": "entity_frequency_difficulty",
                "zero_shot_entity": "zero_shot_entity",
                "zero_shot_surface_form": "zero_shot_surface_form"
            }
        invalid_keys = []
        for val in args.others:
            if val not in continuous_char_to_discrete_char:
                invalid_keys.append(f"focus: {args.focus}")
        for val in args.others:
            if val not in continuous_char_to_discrete_char:
                invalid_keys.append(f"others: {val}")
        
        if invalid_keys:
            raise ValueError(f"The following keys are not valid continuous characteristics: {', '.join(invalid_keys)}")
        
        print(f"[INFO] Plotting focus for {args.focus} vs {args.others}.")
        for char_name in args.focus:
            chosen_char = {c: continuous_char_to_discrete_char[c] for c in args.others + [char_name]}
            all_preds = load_json("metrics/tmp/aggregated_annotations.json")

            if args.model is not None:
                all_preds = [p for p in all_preds if p.get("model") == args.model]

            x, y = plot_performance_by_continuous_characteristic_simple(
                gold=gold,
                preds=preds,
                characteristic_name=char_name,
                mode=args.mode,
                k=args.k,
                show=False,
                chosen_model=args.model
            )

            if args.model is None:
                model_name = all
            else:
                model_name = args.model
            plot_focus_vs_chosen_others(
                x=x, y=y,
                focus_char=char_name,
                continuous_char_to_discrete_char=chosen_char,
                preds=all_preds,
                save_path=f"metrics/plots/focus_{char_name}_vs_others_{'_'.join(args.others)}_{model_name}.png"
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
