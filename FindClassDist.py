import collections
import pathlib

import jsonlines
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split


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

DATA = getData()[1]
entities = []
for _,entry in DATA:
    for e in entry["entities"]:
        if e[2] == 'DEGREE' or e[2] == 'MASTER' or e[2] == 'ADVDIP' or e[2] == 'DIPLOMA' or e[2] == 'PHD' or e[2] == 'RELEVANT':
            e[2] = 'EDU'
        entities.append(e[2])
# distributions tag according to category
count = collections.Counter(entities)
print(count)
tag = list(count.keys())
val = list(count.values())
# plot distribution
plt.barh(tag, val)
plt.title('Distribution of Class in the Samples')
plt.xlabel('Count')
plt.ylabel('Class')
plt.show()
