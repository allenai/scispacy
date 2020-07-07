---
layout: default
---

**scispaCy is a Python package containing [spaCy](https://spacy.io/) models for processing _biomedical_, _scientific_ or _clinical_ text.**


## Interactive Demo
Just looking to test out the models on your data? Check out our [demo](https://scispacy.apps.allenai.org).

## Installing
```python
pip install scispacy
pip install <Model URL>
```
## Models

| Model          | Description       | Install URL
|:---------------|:------------------|:----------|
| en_core_sci_sm | A full spaCy pipeline for biomedical data. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz)|
| en_core_sci_md |  A full spaCy pipeline for biomedical data with a larger vocabulary and 50k word vectors. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_md-0.2.5.tar.gz)|
| en_core_sci_lg |  A full spaCy pipeline for biomedical data with a larger vocabulary and 600k word vectors. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_lg-0.2.5.tar.gz)|
| en_ner_craft_md|  A spaCy NER model trained on the CRAFT corpus.|[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_craft_md-0.2.5.tar.gz)|
| en_ner_jnlpba_md | A spaCy NER model trained on the JNLPBA corpus.| [Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_jnlpba_md-0.2.5.tar.gz)|
| en_ner_bc5cdr_md |  A spaCy NER model trained on the BC5CDR corpus. | [Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bc5cdr_md-0.2.5.tar.gz)|
| en_ner_bionlp13cg_md |  A spaCy NER model trained on the BIONLP13CG corpus. | [Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bionlp13cg_md-0.2.5.tar.gz)|




### Performance

Our models achieve performance within 3% of published state of the art dependency parsers and within 0.4% accuracy of state of the art biomedical POS taggers.

| model          | UAS | LAS   | POS   | Mentions (F1) | Web UAS | 
|:---------------|:----|:------|:------|:---|:---|
| en_core_sci_sm | 89.26| 87.38  |  98.38  |  67.14  |  87.18  |
| en_core_sci_md | 89.92| 88.01  |  98.54  |  69.46  |  88.20  |
| en_core_sci_lg | 89.81| 88.02  |  98.57  |  69.29  |  88.11  |


| model          | F1 |   Entity Types|
|:---------------|:-----|:--------|
| en_ner_craft_md | 75.02|GGP, SO, TAXON, CHEBI, GO, CL|
| en_ner_jnlpba_md | 73.56| DNA, CELL_TYPE, CELL_LINE, RNA, PROTEIN |
| en_ner_bc5cdr_md | 84.94| DISEASE, CHEMICAL|
| en_ner_bionlp13cg_md | 78.09|CANCER, ORGAN, TISSUE, ORGANISM, CELL, AMINO_ACID, GENE_OR_GENE_PRODUCT, SIMPLE_CHEMICAL, ANATOMICAL_SYSTEM, IMMATERIAL_ANATOMICAL_ENTITY, MULTI-TISSUE_STRUCTURE, DEVELOPING_ANATOMICAL_STRUCTURE, ORGANISM_SUBDIVISION, CELLULAR_COMPONENT|


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
We have made this [dataset available along with the original raw data](https://github.com/allenai/genia-dependency-trees).
*   **[word2vec word vectors](http://bio.nlplab.org/#word-vectors)** trained on the Pubmed Central Open Access Subset.
*   **[The MedMentions Entity Linking dataset](https://github.com/chanzuckerberg/MedMentions)**, used for training a mention detector.
*  **[Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)** to make the parser and tagger more robust to non-biomedical text. Unfortunately this is not publically available.
