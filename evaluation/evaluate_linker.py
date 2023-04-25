import spacy
from scispacy.linking import EntityLinker
from scispacy.data_util import read_full_med_mentions
import os
from tqdm import tqdm

EVALUATION_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))


def main():
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe(
        "scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"}
    )
    linker = nlp.get_pipe("scispacy_linker")

    med_mentions = read_full_med_mentions(
        os.path.join(EVALUATION_FOLDER_PATH, os.pardir, "data", "med_mentions"),
        use_umls_ids=True,
    )

    test_data = med_mentions[2]

    total_entities = 0
    correct_at_1 = 0
    correct_at_2 = 0
    correct_at_10 = 0
    correct_at_40 = 0
    correct_at_60 = 0
    correct_at_80 = 0
    correct_at_100 = 0
    for text_doc, entities in tqdm(test_data):
        for start, end, label in entities["entities"]:
            text_span = text_doc[start:end]
            candidates = linker.candidate_generator([text_span], 40)[0]
            sorted_candidates = sorted(
                candidates, reverse=True, key=lambda x: max(x.similarities)
            )
            candidate_ids = [c.concept_id for c in sorted_candidates]
            if label in candidate_ids[:1]:
                correct_at_1 += 1
            if label in candidate_ids[:2]:
                correct_at_2 += 1
            if label in candidate_ids[:10]:
                correct_at_10 += 1
            if label in candidate_ids[:40]:
                correct_at_40 += 1
            # if label in candidate_ids[:60]:
            #     correct_at_60 += 1
            # if label in candidate_ids[:80]:
            #     correct_at_80 += 1
            # if label in candidate_ids[:100]:
            #     correct_at_100 += 1

            total_entities += 1

    print("Total entities: ", total_entities)
    print(
        "Correct at 1: ", correct_at_1, "Recall at 1: ", correct_at_1 / total_entities
    )
    print(
        "Correct at 2: ", correct_at_2, "Recall at 2: ", correct_at_2 / total_entities
    )
    print(
        "Correct at 10: ",
        correct_at_10,
        "Recall at 10: ",
        correct_at_10 / total_entities,
    )
    print(
        "Correct at 40: ",
        correct_at_40,
        "Recall at 40: ",
        correct_at_40 / total_entities,
    )
    # print(
    #     "Correct at 60: ",
    #     correct_at_60,
    #     "Recall at 60: ",
    #     correct_at_60 / total_entities,
    # )
    # print(
    #     "Correct at 80: ",
    #     correct_at_80,
    #     "Recall at 80: ",
    #     correct_at_80 / total_entities,
    # )
    # print(
    #     "Correct at 100: ",
    #     correct_at_100,
    #     "Recall at 100: ",
    #     correct_at_100 / total_entities,
    # )


if __name__ == "__main__":
    main()
