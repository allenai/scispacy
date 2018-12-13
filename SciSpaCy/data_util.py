
from typing import NamedTuple, List, Iterator


class MedMentionEntity(NamedTuple):
    start: int
    end: int
    mention_text: str
    mention_type: str
    umls_id: str

class MedMentionExample(NamedTuple):
    title: str
    abstract: str
    text: str
    pubmed_id: str
    entities: List[MedMentionEntity]


def process_example(lines: List[str]) -> MedMentionExample:
    """
    Processes the text lines of a file corresponding to a single MedMention abstract,
    extracts the title, abstract, pubmed id and entities. The lines of the file should
    have the following format:
    PMID | t | Title text
    PMID | a | Abstract text
    PMID TAB StartIndex TAB EndIndex TAB MentionTextSegment TAB SemanticTypeID TAB EntityID
    ...
    """
    pubmed_id, _, title = [x.strip() for x in lines[0].split("|", maxsplit=2)]
    _, _, abstract = [x.strip() for x in lines[1].split("|", maxsplit=2)]

    entities = []
    for entity_line in lines[2:]:
        _, start, end, mention, mention_type, umls_id = entity_line.split("\t")
        entities.append(MedMentionEntity(int(start), int(end),
                                         mention, mention_type, umls_id))
    return MedMentionExample(title, abstract, title + " " + abstract, pubmed_id, entities)

def med_mentions_example_iterator(filename: str) -> Iterator[MedMentionExample]:
    """
    Iterates over a Med Mentions file, yielding examples.
    """
    with open(filename, "r") as med_mentions_file:
        lines = []
        for line in med_mentions_file:
            line = line.strip()
            if line:
                lines.append(line)
            else:
                yield process_example(lines)
                lines = []
        # Pick up stragglers
        if lines:
            yield process_example(lines)

def read_med_mentions(filename: str):
    """
    Reads in the MedMentions dataset into Spacy's
    NER format.
    """
    examples = []
    for example in med_mentions_example_iterator(filename):
        spacy_format_entities = [(x.start, x.end, x.mention_type) for x in example.entities]
        examples.append((example.text, {"entities": spacy_format_entities}))

    return examples
