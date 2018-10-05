# SciSpaCy
This repository contains custom pipes and models related to using spaCy for scientific documents. In particular, there is a custom tokenizer that adds tokenization rules on top of spaCy's rule-based tokenizer, and a custom sentence segmenter that adds sentence segmentation rules on top of spaCy's statistical sentence segmenter.

# Usage
## Using SciSpaCy as is
To use SciSpaCy as is, follow these steps:
1. Clone this repository
1. From within this repository, run
`./scripts/create_model_package.sh ./SciSpaCy/models/combined_rule_tokenizer_and_segmenter`
1. Run `python setup.py sdist`
1. Run `pip install --user dist/scispacy-1.0.0.tar.gz`
1. Run `pip install --user dist/en_scispacy_core_web_sm-1.0.0.tar.gz`

Once you have completed the above steps, you can load SciSpaCy as you would any other spaCy model. For example: 
```
import spacy
nlp = spacy.load("en_scispacy_core_web_sm")
```

To make full use of this package, you will also need to preprocess the text that you will be running through spaCy. This means passing the raw text through `custom_tokenizer.remove_new_lines()` before passing it through spaCy.

## Modifying SciSpaCy
### Changing the tokenizer or segmenter
To change the tokenizer or segmenter, all you need to do is change the tokenization or segmentation function, rebuild the model folder, and then follow the above steps for using SciSpaCy as is. In detail:

1. Change the tokenizer (`combined_rule_tokenizer()` in `SciSpaCy/custom_tokenizer.py`) and/or segmenter(`combined_rule_sentence_segmenter()` in `SciSpaCy/custom_sentence_segmenter.py`)
1. Rebuild the model folder by running `save_model(create_combined_rule_model, /path/to/model/folder)` in `SciSpaCy/util.py`
1. Edit the newly create `meta.json` as you see fit
1. Go through the steps above for using SciSpaCy as is

### Adding a new pipe or further customization
Adding a new pipe requires that you 
1. Create your new pipe
1. Save your model following the pattern described above for changing the tokenizer or segmenter (steps 2 and 3)
1. Add your new pipe to `Language.factories` in `proto_model/__init__.py`
1. Follow the steps for using SciSpaCy as is
