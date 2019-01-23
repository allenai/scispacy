---
layout: default
---

**SciSpaCy is a Python package containing [SpaCy](https://spacy.io/) models for processing _biomedical_, _scientific_ or _clinical_ text.**

## Installing
```python
pip install <Mark add package here>
pip install <add paths to models here.>
```
## Models

### Performance

Our models achieve performance within 1% of published state of the art dependency parsers and within 0.2% accuracy of state of the art biomedical POS taggers.

| model          | UAS | LAS   | POS   | NER | Web UAS
|:---------------|:----|:------|:------|:---|:---|
| en_core_sci_sm | good| nice  |       |    |    |
| en_core_sci_md | good| nice  |       |    |    |


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

SciSpaCy models are trained on data from a variety of sources. In particular,
we use:

*   **[The GENIA 1.0 Treebank](https://nlp.stanford.edu/~mcclosky/biomedical.html)**, converted to basic Universal Dependencies using the [Stanford Dependency Converter](https://nlp.stanford.edu/software/stanford-dependencies.shtml).
We have made this dataset available along with the original raw data here. TODO(Mark) add link.
*   **[word2vec word vectors](http://bio.nlplab.org/#word-vectors)** trained on the Pubmed Central Open Access Subset.
*   **[The MedMentions Entity Linking dataset](https://github.com/chanzuckerberg/MedMentions)**, used for training a mention detector.
*  **[Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)** to make the parser and tagger more robust to non-biomedical text. Unfortunately this is not publically available.
