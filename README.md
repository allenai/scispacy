
<p align="center"><img width="50%" src="docs/scispacy-logo.png" /></p>


This repository contains custom pipes and models related to using spaCy for scientific documents.

In particular, there is a custom tokenizer that adds tokenization rules on top of spaCy's
rule-based tokenizer, a POS tagger and syntactic parser trained on biomedical data and
an entity span detection model. Separately, there are also NER models for more specific tasks.


## Installation
Installing scispacy requires two steps: installing the library and intalling the models. To install the library, run:
```bash
pip install scispacy
```

to install a model (see our full selection of available models below), run a command like the following:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_core_sci_sm-0.1.0.tar.gz
```

Note: We strongly recommend that you use an isolated Python environment (such as virtualenv or conda) to install scispacy.
Take a look below in the "Setting up a virtual environment" section if you need some help with this.
Additionally, scispacy uses modern features of Python and as such is only available for **Python 3.5 or greater**.



#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for scispaCy.  If you already have a Python 3.6 or 3.7
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment called "scispacy" with Python 3.6:

    ```bash
    conda create -n scispacy python=3.6
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use scispaCy.

    ```bash
    source activate scispacy
    ```

Now you can install `scispacy` and one of the models using the steps above.


Once you have completed the above steps and downloaded one of the models below, you can load a scispaCy model as you would any other spaCy model. For example:
```python
import spacy
nlp = spacy.load("en_core_sci_sm")
doc = nlp("Alterations in the hypocretin receptor 2 and preprohypocretin genes produce narcolepsy in some animals.")
```

## Available Models

To install a model, click on the link below to download the model, and then run 

```python
pip install </path/to/download>
```

Alternatively, you can install directly from the URL by right-clicking on the link, selecting "Copy Link Address" and running 
```python
pip install CMD-V(to paste the copied URL)
```

| Model          | Description       | Install URL
|:---------------|:------------------|:----------|
| en_core_sci_sm | A full spaCy pipeline for biomedical data. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_core_sci_sm-0.1.0.tar.gz)|
| en_core_sci_md |  A full spaCy pipeline for biomedical data with a larger vocabulary and word vectors. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_core_sci_md-0.1.0.tar.gz)|
| en_ner_craft_md|  A spaCy NER model trained on the CRAFT corpus.|[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_ner_craft_md-0.1.0.tar.gz)|
| en_ner_jnlpba_md | A spaCy NER model trained on the JNLPBA corpus.| [Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_ner_jnlpba_md-0.1.0.tar.gz)|
| en_ner_bc5cdr_md |  A spaCy NER model trained on the BC5CDR corpus. | [Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_ner_bc5cdr_md-0.1.0.tar.gz)|
| en_ner_bionlp13cg_md |  A spaCy NER model trained on the BIONLP13CG corpus. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_ner_bionlp13cg_md-0.1.0.tar.gz)|

## Paper on arXiv

Further details about this module can be found in [ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing](https://arxiv.org/abs/1902.07669)

ScispaCy is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

