
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

to install a model, run:

```bash
pip install <model url>
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


<table>
<tr>
    <td><b> en_core_sci_sm </b></td>
    <td> A full spaCy pipeline for biomedical data. </td>
</tr>
<tr>
    <td><b> en_core_sci_md </b></td>
    <td>  A full spaCy pipeline for biomedical data with a larger vocabulary and word vectors. </td>
</tr>
<tr>
    <td><b> en_ner_craft_md </b></td>
    <td> A spaCy NER model trained on the CRAFT corpus. </td>
</tr>
<tr>
    <td><b> en_ner_jnlpba_md </b></td>
    <td> A spaCy NER model trained on the JNLPBA corpus. </td>
</tr>
<tr>
    <td><b> en_ner_bc5cdr_md </b></td>
    <td> A spaCy NER model trained on the BC5CDR corpus. </td>
</tr>
<tr>
    <td><b> en_ner_bionlp13cg_md </b></td>
    <td> A spaCy NER model trained on the BIONLP13CG</td>
</tr>
</table>

