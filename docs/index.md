---
layout: default
---

**scispaCy is a Python package containing [spaCy](https://spacy.io/) models for processing _biomedical_, _scientific_ or _clinical_ text.**

## Installing
```python
pip install scispacy
pip install <Model URL>
```
## Models

| Model          | Description       | Install URL
|:---------------|:------------------|:----------|
| en_core_sci_sm | A full spaCy pipeline for biomedical data. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_core_sci_sm-0.1.0.tar.gz)|
| en_core_sci_md |  A full spaCy pipeline for biomedical data with a larger vocabulary and word vectors. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_core_sci_md-0.1.0.tar.gz)|
| en_ner_craft_md|  A spaCy NER model trained on the CRAFT corpus.|[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_ner_craft_md-0.1.0.tar.gz)|
| en_ner_jnlpba_md | A spaCy NER model trained on the JNLPBA corpus.| [Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_ner_jnlpba_md-0.1.0.tar.gz)|
| en_ner_bc5cdr_md |  A spaCy NER model trained on the BC5CDR corpus. | [Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_ner_bc5cdr_md-0.1.0.tar.gz)|
| en_ner_bionlp13cg_md |  A spaCy NER model trained on the BIONLP13CG corpus. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.1.0/en_ner_bionlp13cg_md-0.1.0.tar.gz)|




### Performance

Our models achieve performance within 2% of published state of the art dependency parsers and within 0.2% accuracy of state of the art biomedical POS taggers.

| model          | UAS | LAS   | POS   | Mentions (F1) | Web UAS | 
|:---------------|:----|:------|:------|:---|:---|
| en_core_sci_sm | 89.69| 87.67  |  98.38  |  68.19  |  85.85  |
| en_core_sci_md | 90.60| 88.79  |  98.51  |  68.97  |  84.75  |


### Example Usage

```python
import scispacy
import spacy

nlp = spacy.load("en_core_sci_sm")
text = """
Myeloid derived suppressor cells (MDSC) are immature 
myeloid cells with immunosuppressive activity. 
They accumulate in tumor-bearing mice and humans 
with different types of cancer, including hepatocellular 
carcinoma (HCC).
"""
doc = nlp(text)

print(list(doc.sents))
>>> ["Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity.", 
     "They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC)."]

# Examine the entities extracted by the mention detector.
# Note that they don't have types like in SpaCy, and they
# are more general (e.g including verbs) - these are any
# spans which might be an entity in UMLS, a large
# biomedical database.
print(doc.ents)
>>> (Myeloid derived suppressor cells,
     MDSC,
     immature,
     myeloid cells,
     immunosuppressive activity,
     accumulate,
     tumor-bearing mice,
     humans,
     cancer,
     hepatocellular carcinoma,
     HCC)

# We can also visualise dependency parses
# (This renders automatically inside a jupyter notebook!):
from spacy import displacy
displacy.render(next(doc.sents), style='dep', jupyter=True)

# See below for the generated SVG.
# Zoom your browser in a bit!

```

![Branching](./example.svg)

### Data Sources

scispaCy models are trained on data from a variety of sources. In particular,
we use:

*   **[The GENIA 1.0 Treebank](https://nlp.stanford.edu/~mcclosky/biomedical.html)**, converted to basic Universal Dependencies using the [Stanford Dependency Converter](https://nlp.stanford.edu/software/stanford-dependencies.shtml).
We have made this dataset available along with the original raw data here. TODO(Mark) add link.
*   **[word2vec word vectors](http://bio.nlplab.org/#word-vectors)** trained on the Pubmed Central Open Access Subset.
*   **[The MedMentions Entity Linking dataset](https://github.com/chanzuckerberg/MedMentions)**, used for training a mention detector.
*  **[Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)** to make the parser and tagger more robust to non-biomedical text. Unfortunately this is not publically available.
