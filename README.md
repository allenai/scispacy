
<p align="center"><img width="40%" src="docs/scispacy-logo.png" /></p>


# SciSpaCy
This repository contains custom pipes and models related to using spaCy for scientific documents. In particular, there is a custom tokenizer that adds tokenization rules on top of spaCy's rule-based tokenizer, and a custom sentence segmenter that adds sentence segmentation rules on top of spaCy's statistical sentence segmenter.


## Installation
Installing scispacy requires two steps: installing the library and intalling the models. To install the library, run:
```
pip install scispacy
```

to install a model, run:

```
pip install <model url>
```

Note: We strongly recommend that you use an isolated Python environment (such as virtualenv or conda) to install scispacy.
Additionally, scispacy uses modern features of Python and is such only available for Python 3.5 or greater.


Once you have completed the above steps and downloaded one of the models below, you can load SciSpaCy as you would any other spaCy model. For example:
```
import spacy
nlp = spacy.load("en_scispacy_core_web_sm")
```

## Available Models


<table>
<tr>
    <td><b> en_core_sci_sm </b></td>
    <td> A full SpaCy pipeline for biomedical data. </td>
</tr>
<tr>
    <td><b> en_core_sci_md </b></td>
    <td>  A full SpaCy pipeline for biomedical data with a larger vocabulary and word vectors. </td>
</tr>
<tr>
    <td><b> en_ner_craft_md </b></td>
    <td> A SpaCy NER model trained on the CRAFT corpus. </td>
</tr>
<tr>
    <td><b> en_ner_jnlpba_md </b></td>
    <td> A SpaCy NER model trained on the JNLPBA corpus. </td>
</tr>
<tr>
    <td><b> en_ner_bc5cdr_md </b></td>
    <td> A SpaCy NER model trained on the BC5CDR corpus. </td>
</tr>
<tr>
    <td><b> en_ner_bionlp13cg_md </b></td>
    <td> A SpaCy NER model trained on the BIONLP13CG</td>
</tr>
</table>

