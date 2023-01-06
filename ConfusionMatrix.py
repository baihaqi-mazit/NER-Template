import pathlib
import pprint

import jsonlines
import matplotlib.pyplot as plt
import numpy
import spacy
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from spacy.training import offsets_to_biluo_tags
from spacy.training.iob_utils import biluo_tags_from_offsets


class ConfMatrix(object):

    def __init__(self):
        self.model = spacy.load('output/model-best')
        self.n_epoch = 200
        self.batch_size = 5
        self.loss = []
        self.train_data, self.test_data, _ = self.getData()
        pprint.pprint(self.train_data)
        print(len(self.train_data))
        print(len(self.test_data))

    def getData(self):
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

    def get_cleaned_label(self, label: str):
        if "-" in label:
            return label.split("-")[1]
        else:
            return label

    def get_label_target(self):
        data = self.test_data
        target_vector = []

        for doc, entity in data:
            new = self.model.make_doc(doc)
            entities = []
            # # Change degree/master/advip/diploma/phd/relevant to one Edu label---
            for e in entity["entities"]:
                if e[2] == 'DEGREE' or e[2] == 'MASTER' or e[2] == 'ADVDIP' or e[2] == 'DIPLOMA' or e[2] == 'PHD' or e[
                    2] == 'RELEVANT':
                    e[2] = 'EDU'
                entities.append((e[0], e[1], e[2]))
            bilou_entities = offsets_to_biluo_tags(new, entities)
            # --------------------------------------------------------------------
            new_bilou_entities = []
            for e in bilou_entities:
                if e == 'I-POSITION' or e == 'L-POSITION' or e == 'I-COMPANY' or e == 'L-COMPANY' or e == 'I-YEAR' or e == 'L-YEAR' or e == 'I-EDU' or e == 'L-EDU' or e == '-':
                    e = 'O'
                new_bilou_entities.append(e)
            final = []
            for item in new_bilou_entities:
                final.append(self.get_cleaned_label(item))
            target_vector.extend(final)
            lenght = len(target_vector)
        return target_vector

    def get_all_ner_prediction(self):
        data = self.test_data
        label = []
        z = []
        for entry, _ in data:
            y = self.model(entry)
            entities = [(e.start_char, e.end_char, e.label_) for e in y.ents]
            bilou_entities = biluo_tags_from_offsets(y, entities)
            new_bilou_entities = []
            for e in bilou_entities:
                if e == 'I-POSITION' or e == 'L-POSITION' or e == 'I-COMPANY' or e == 'L-COMPANY' or e == 'I-YEAR' or e == 'L-YEAR' or e == 'I-EDU' or e == 'L-EDU':
                    e = 'O'
                new_bilou_entities.append(e)
            final = []
            for item in new_bilou_entities:
                final.append(self.get_cleaned_label(item))
            label.extend(final)
        length = len(label)
        return label

    def generate_confusion_matrix(self):
        classes = sorted(set(self.get_label_target()))
        y_true = self.get_label_target()
        h = len(y_true)
        y_pred = self.get_all_ner_prediction()
        k = len(y_pred)
        return confusion_matrix(y_true, y_pred, classes)

    def get_dataset_labels(self):
        return sorted(set(self.get_label_target()))

    def plot_confusion_matrix(self, classes, normalize=False, cmap=pyplot.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        title = 'Confusion Matrix, for SpaCy NER'

        # Compute confusion matrix
        cm = self.generate_confusion_matrix()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

        fig, ax = pyplot.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=numpy.arange(cm.shape[1]),
               yticks=numpy.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return cm, ax, pyplot


if __name__ == '__main__':
    m=ConfMatrix()
    classe = m.get_dataset_labels()
    print(classe)
    m.plot_confusion_matrix(classes=classe, normalize=False)  # Plot the confusion matrix
    plt.show()
