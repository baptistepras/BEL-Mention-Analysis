# BEL-Mention-Analysis

A mention-level analysis of biomedical entity linking models, built on the Biomedical Entity Linking Benchmark (BELB). Global accuracy hides where models fail. This repository measures performance across interpretable characteristics of each mention (length, lexical variation, synonymy, homonymy, training frequency, zero-shot entities and surface forms) and compares neural and rule-based systems on several corpora.

It is the code of our MIE 2026 paper, also included in this repository (`Fine-Grained_Mention-Level_Analysis_of_Biomedical_Entity_Linking_Models.pdf`).

## Citation

If you use this code, please cite:

> B. Pras and N. Naderi. Fine-Grained Mention-Level Analysis of Biomedical Entity Linking Models. *Medical Informatics Europe (MIE)*, 2026. https://ebooks.iospress.nl/volumearticle/78623

The benchmark itself comes from:

> S. Garda, L. Weber-Genzel, R. Martin, and U. Leser. BELB: a Biomedical Entity Linking Benchmark. *Bioinformatics*, 2023. https://academic.oup.com/bioinformatics/article/39/11/btad698/7425450

## Repository content

| Path | Content |
| --- | --- |
| `belb/` | The BELB library, modified to run on this setup |
| `belb-exp/` | The BELB experiments, extended with our mention-level analysis (`belb-exp/metrics/`) |
| `environment.yml` | Conda environment for Linux and macOS |

`belb/` and `belb-exp/` are modified copies of [sg-wbi/belb](https://github.com/sg-wbi/belb) and [sg-wbi/belb-exp](https://github.com/sg-wbi/belb-exp) by Garda et al. The upstream repositories do not state a license, so their authors keep all rights on that code. This repository adds no license either.

## Setup

### Knowledge bases

Download the processed knowledge bases (675 MB archive, 2 GB once unzipped) and unzip them in `belb/processed/`, which creates `belb/processed/kbs/`:

https://drive.google.com/file/d/1qDdQIhkGduWKGi-VrVn5aFOmQd3xDipd/view?usp=sharing

### Environment

One conda environment works on Linux and macOS. Create it from the root of the repository:

```bash
conda env create -f environment.yml    # or: mamba env create -f environment.yml
conda activate belb-env
```

It installs Python 3.9, the pinned versions used for the paper, the scispacy model, and the local `belb/` library in editable mode.

On Apple Silicon, `nmslib` comes from the Anaconda main channel. If conda asks you to accept its terms of service, run `conda tos accept --channel https://repo.anaconda.com/pkgs/main`, or use mamba.

## Usage

All commands run from `belb-exp/`.

### Benchmark

Choose the corpora in the `CORPORA` list of `scripts/evaluate.py`, then run:

```bash
PYTHONPATH=../belb:. python scripts/evaluate.py --belb_dir ../belb --k 1 --mode std
```

`--k` takes any integer, `--mode` takes `std`, `strict` or `lenient`, and `--full` is optional.

### Mention-level metrics

Annotate the predictions with the mention characteristics:

```bash
python3 metrics/annotate_preds.py --corpora <corpus_name>
```

Then run the benchmark again with `--advanced`. Useful options:

| Option | Effect |
| --- | --- |
| `--force` | Recompute the metrics instead of reading the saved ones |
| `--synonymy <int>` | Maximum number of synonyms of a poorly annotated entity (default 10) |
| `--length <int>` | Minimum number of tokens of a long mention (default 10) |
| `--variation <float>` | Minimum lexical variation considered high (default 0.1) |
| `--frequency <int>` | Maximum frequency of a rare entity or mention (default 10) |
| `--plot` | Save the plots in `metrics/plots/` |
| `--focus <name> --others <names>` | Plot one continuous characteristic against one or more discrete ones |
| `--model <name>` | Restrict the plots to `arboel`, `genbioel` or `rbes` |

Valid characteristic names: `mention_length`, `num_synonyms`, `num_homonyms`, `lexical_variation`, `mention_frequency`, `entity_frequency`, `zero_shot_entity`, `zero_shot_surface_form`.

### Dataset characteristics

```bash
python3 metrics/analyze_datasets.py --corpora <corpus_name>
```

It accepts the same `--force`, `--synonymy`, `--length`, `--variation` and `--frequency` options.

### Converting raw data

All corpora and knowledge bases used in the paper are already converted. To convert new ones, store the raw data in `belb/raw/corpora/<corpus_name>` or `belb/raw/kbs/<kb_name>`, then run:

```bash
PYTHONPATH=../belb:. python -m belb.corpora.<corpus_name> --dir ../belb --db ../belb/db.yaml --pubtator ../belb/pubtator/pubtator.db --sentences
PYTHONPATH=../belb:. python -m belb.kbs.<kb_name> --dir ../belb --data_dir ../belb/raw/kbs/<kb_name> --db ../belb/db.yaml
```

`--pubtator` is only needed by some corpora. The results go to `belb/processed/corpora/` and `belb/processed/kbs/`.

## Corpora

These corpora run correctly: s800, MedMentions, Linnaeus, NCBI Disease and NLM-Chem. NCBI Disease and NLM-Chem give low scores, which may point to a corrupted corpus or knowledge base.

These could not be processed: BC5CDR Disease and BC5CDR Chemical (corrupted zip files), BioID (corrupted tar file), GNormPlus and NLM-Gene (corrupted NCBI Gene knowledge base), and SNP, Osiris and tmVar (they need dbSNP).

PubTator (a 32 GB archive turned into a 100 GB SQLite database) and dbSNP (over 100 GB) are only needed to convert some corpora, not to run the benchmark.

## Acknowledgments

This work was supported by the CHIST-ERA grant CHIST-ERA22-ORD-02 and by the Agence Nationale de la Recherche under projects ANR23-CHRO-0008-01 and ANR-22-CPJ1-0087-01.
