from typing import Optional, Callable
from pathlib import Path

import random
import itertools
import spacy
from spacy.training import Corpus

from scispacy.custom_tokenizer import combined_rule_tokenizer


def iter_sample(iterable, sample_percent):
    for item in iterable:
        if len(item.reference) == 0:
            continue
        coin_flip = random.uniform(0, 1)
        if coin_flip < sample_percent:
            yield item


@spacy.registry.callbacks("replace_tokenizer")
def replace_tokenizer_callback():
    def replace_tokenizer(nlp):
        nlp.tokenizer = combined_rule_tokenizer(nlp)
        return nlp

    return replace_tokenizer


@spacy.registry.readers("parser_tagger_data")
def parser_tagger_data(
    path: Path,
    mixin_data_path: Optional[Path],
    mixin_data_percent: float,
    gold_preproc: bool,
    max_length: int = 0,
    limit: int = 0,
    augmenter: Optional[Callable] = None,
    seed: int = 0,
):
    random.seed(seed)
    main_corpus = Corpus(
        path,
        gold_preproc=gold_preproc,
        max_length=max_length,
        limit=limit,
        augmenter=augmenter,
    )
    if mixin_data_path is not None:
        mixin_corpus = Corpus(
            mixin_data_path,
            gold_preproc=gold_preproc,
            max_length=max_length,
            limit=limit,
            augmenter=augmenter,
        )

    def mixed_corpus(nlp):
        if mixin_data_path is not None:
            main_examples = main_corpus(nlp)
            mixin_examples = iter_sample(mixin_corpus(nlp), mixin_data_percent)
            return itertools.chain(main_examples, mixin_examples)
        else:
            return main_corpus(nlp)

    return mixed_corpus


@spacy.registry.callbacks("ontonotes_dev")
def ontonotes_dev_callback():
    def ontonotes_dev():
        pass

    return ontonotes_dev


@spacy.registry.callbacks("ontonotes_test")
def ontonotes_test_callback():
    def ontonotes_test():
        pass

    return ontonotes_test