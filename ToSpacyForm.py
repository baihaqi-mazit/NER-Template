import pathlib
import re
import warnings

import jsonlines
import numpy
import spacy
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
from tqdm import tqdm


def getData():
    json_path = list(pathlib.Path('JsonL').glob('*.jsonl'))
    ALL_DATA = []
    data = []

    for i in json_path:
        with jsonlines.open(i) as f:
            for line in f.iter():
                data.append(line)

    for entry in data:
        entities = []
        for e in entry['label']:
            entities.append((e[0], e[1], e[2]))
        spacy_entity = (entry["data"], {"entities": entities})
        ALL_DATA.append(spacy_entity)

    split_data = numpy.array(ALL_DATA)
    train_data, test_data = train_test_split(split_data, test_size=0.2, random_state=11)
    train_data = train_data.tolist()
    test_data = test_data.tolist()

    return train_data, test_data, data


TRAIN_DATA = getData()[0]
TEST_DATA = getData()[1]
nlp = spacy.blank("en")  # load a new spacy model
dbTrain = DocBin()  # create a DocBin object
dbTest = DocBin()  # create a DocBin object


def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
    data (list): The data to be cleaned in spaCy JSON format.

    Returns:
    list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    # count = 0
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            # count += 1
            # print(f"text : {text} , count : {count}")
            # print(f"valid_end : {valid_end}")
            # print(f"text length : {len(text)}")
            # if there's preceding spaces, move the start position to nearest character
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    # print(f"count : {count}")
    return cleaned_data


TRAIN_DATA = trim_entity_spans(TRAIN_DATA)
TEST_DATA = trim_entity_spans(TEST_DATA)

for text, annot in tqdm(TRAIN_DATA):  # data in previous format
    doc = nlp.make_doc(text)  # create doc object from text
    ents = []
    for start, end, label in annot["entities"]:  # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is None:
            msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
            warnings.warn(msg)
            print("Skipping entity")
        else:
            ents.append(span)
    try:
        doc.ents = ents  # label the text with the ents
    except ValueError:
        print(ents)
    dbTrain.add(doc)

for text, annot in tqdm(TEST_DATA):  # data in previous format
    doc = nlp.make_doc(text)  # create doc object from text
    ents = []
    for start, end, label in annot["entities"]:  # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is None:
            msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
            warnings.warn(msg)
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents  # label the text with the ents
    dbTest.add(doc)

dbTrain.to_disk("./train.spacy")  # save the docbin object
dbTest.to_disk("./test.spacy")  # save the docbin object
