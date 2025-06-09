# BELB-Working-Setup

This is a working setup for the Biomedical Entity-Linking Benchmark (BELB), 
including corpora such as medMentions, linnaeus, s800, ncbi_disease and nlm_chem. 
This repository contains all necessary steps to install and run BELB correctly across multiple corpora.

BELB was introduced in this paper: 

https://academic.oup.com/bioinformatics/article/39/11/btad698/7425450?login=true

The original repositories (belb and belb-exp) can be found there 
without my configuration, and with instructions to do your own.

## Usage Instructions

### Downloading the KBs

You will first need to download the processed KBs here:

https://drive.google.com/file/d/1qDdQIhkGduWKGi-VrVn5aFOmQd3xDipd/view?usp=sharing

It's a 675MB archive that you will then need to unzip (2GB).

Unzip it in the following directory: `belb/processed/`

It should create a folder named `kbs`.

### Environment

#### Conda / Mini-conda

From the root, use the corresponding `.yml` file:

`conda env create -f belb-env-linux.yml`

or

`conda env create -f belb-env-macOS.yml` 

and then

`pip install -e belb/.`

`conda activate belb-env`

<br>

Alternatively, you can install it manually (recommended) by running:

`conda create -n belb-env python=3.9 -y`

`conda activate belb-env`

`pip install -r belb-exp/requirements.txt`

`pip install -e belb/.`

`pip install --force-reinstall pandas==1.4.1`

`pip install --force-reinstall numpy==1.26.4`

#### Venv

From the root, install it manually by running:

`python3 -m venv belb-venv`

`source belb-venv/bin/activate`

`pip install -r belb-exp/requirements.txt`

`pip install -e belb/.`

`pip install --force-reinstall pandas==1.4.1`

`pip install --force-reinstall numpy==1.26.4`

<br>

## Converting raw data to processed

All corpora and knowledge bases are already converted in this setup, 
and raw data is not included in the repository, but if you ever need 
to download some, they must be stored in:

`belb/raw/corpora/<corpora_name>` for corpora

or

`belb/raw/kbs/<kb_name>` for knowledge bases

<br>

To convert corpora, run from `belb-exp`:

`PYTHONPATH=../belb:. python -m belb.corpora.<corpora_name> --dir ../belb --db ../belb/db.yaml --pubtator ../belb/pubtator/pubtator.db --sentences`

Note that the argument `pubtator` is optional, not all corpora need it but
some do. Make sure to have the `pubtator.db` file in `belb/pubtator`.

<br>

To convert a knowledge base, run from `belb-exp`:

`PYTHONPATH=../belb:. python -m belb.kbs.<kb_name> --dir ../belb --data_dir ../belb/raw/kbs/<kb_name> --db ../belb/db.yaml`
 
<br>

The processed corpora and knowledges bases will be stored in:

`belb/processed/corpora/<corpora_name>` for corpora

or

`belb/processed/kbs/<kb_name>` for knowledge bases


### Running the benchmark

Edit the CORPORA list in `belb-exp/scripts/evaluate.p` to specify 
which corpora to include in evaluation.

Edit the 

<br>

Then from `belb-exp` run:

`PYTHONPATH=../belb:. python scripts/evaluate.py --belb_dir ../belb --k 1 --mode std`

<br>

The argument `k` can take any integer value.

The argument `mode` can take the values `std`, `strict` or `lenient`.

You can also add the argument `--full`.

<br>

Compute advanced metrics on predictions by first running from `belb-exp`:

`python3 metrics/annotate_preds.py --corpora <corpora_name>`

Add the argument `--force` to force the recomputation of the metrics,
otherwise the script will try to pick them up from saved files.

Then run the evaluation script with the argument `--advanced`.

Add the argument `--plot` to plot the metrics in `metrics/plots/`


### Working corpora

✅ These corpora have been tested and verified to run correctly:

- s800

- medMentions

- linnaeus

- NCBI_disease (Warning: low performances, might suggest a corrupted corpora/KB)

- NLM_chem (Warning: low performances, might suggest a corrupted corpora/KB)

<br>

⚠️ The following corpora could not be processed due to data corruption or missing dependencies:

- bc5cdr_disease (corrupted zip file)

- bc5cdr_chemical (corrupted zip file)

- bioID (corrupted tar file)

- gnormplus (corrupted NCBI_gene KB)

- NLM_gene (corrupted NCBI_gene KB)

- SNP (requires dbSNP knowledge base)

- osiris (requires dbSNP knowledge base)

- tmVar (requires dbSNP knowledge base)


### Pubtator

Pubtator is a tool required by several corpora to be converted to the
BELB format. It consists of a 32GB archive which is then processed
into a 100GB SQLite database.

It is advised not to keep it locally (on your laptop for instance) as it 
is very heavy and unnecessary unless you want to reconvert some corpora.

⚠️ It is not required to run the benchmark evaluation.


### dbSNP

dbSNP is a big knowledge base needed by a few corpora (with VARIANTS). 
It consists in an archive that is over 100GB, and then needs to be 
converted into a processed knowledge base that is probably over 700/800GB.

⚠️ It is not required to run the benchmark evaluation. and advised not 
to use it unless you really need it and know what you are doing.
