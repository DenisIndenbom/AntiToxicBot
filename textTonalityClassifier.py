import re

import numpy as np
from fuzzywuzzy import process, fuzz

from catboost import CatBoostClassifier

import torch
from slovnet.model.emb import NavecEmbedding


class RulesClassifier:
    """
        Classifier of the tonality of the text according to the rules.
    """
    valid_symbols_re = re.compile('[^a-zа-я]', flags=re.IGNORECASE)

    def __init__(self, bad_words: list, bad_word_threshold=0.75) -> None:
        """
        :param bad_words: list of bad words
        :param bad_word_threshold: float in the range 0 to 1
        """
        self.list_of_bad_words = bad_words
        self.bad_word_threshold = bad_word_threshold

    def clear_text(self, text: str) -> str:
        """
        :param text: str
        :return: clean text
        """
        return self.valid_symbols_re.sub('', text)

    def predict(self, x: list) -> np.array:
        """
        :param x: input 2d list with the str. Example [['Hello','my',friends'],['My','name','is','Jack']]

        :return: numpy array with predictions. Example np.array([0,0])
        """
        y = []
        for row in x:
            in_list = False
            for word in row:
                clear_word = self.clear_text(word)
                if clear_word == '':
                    continue
                if process.extractOne(clear_word, self.list_of_bad_words, scorer=fuzz.ratio)[1] > self.bad_word_threshold * 100:
                    in_list = True
                    break
            if in_list:
                y.append(1)
            else:
                y.append(0)

        return np.array(y)


class CBClassifier:
    def __init__(self, model_path: str) -> None:
        """
        :param model_path: path to catboost model
        """
        self.cb_clf = CatBoostClassifier()
        self.cb_clf.load_model(model_path, format='cbm')

    def predict(self, x) -> np.array:
        """
        :param x: input 1d array-like or 2d array-like.
        :return: numpy array with predictions.
        """
        x = self.cb_clf.predict(x)
        return x


class TextClassifierNN(torch.nn.Module):
    """
        Neural network model for the classification of text tonality
    """
    def __init__(self, embedding_dim: int, gru_hidden_size: int, fc_hidden_size: int, output_size: int, navec) -> None:
        """
        :param embedding_dim: embedding dim
        :param gru_hidden_size: gru hidden size
        :param fc_hidden_size: full connected hidden size
        :param output_size: output size
        :param navec: navec model
        """
        super(TextClassifierNN, self).__init__()

        self.relu = torch.nn.ReLU()

        self.softmax = torch.nn.Softmax(dim=1)

        self.embedding = NavecEmbedding(navec)  # torch.nn.Embedding(input_size, embedding_dim)

        #weights = torch.FloatTensor(model.vectors)
        #self.embedding = torch.nn.Embedding.from_pretrained(weights)

        self.conv1 = torch.nn.Conv1d(embedding_dim, 512, kernel_size=(5,), padding=2)
        self.conv2 = torch.nn.Conv1d(512, 1024, kernel_size=(3,), padding=1)
        self.conv3 = torch.nn.Conv1d(1024, 2048, kernel_size=(5,), padding=2)

        self.gru = torch.nn.GRU(2048, gru_hidden_size, batch_first=True)

        self.fc1 = torch.nn.Linear(gru_hidden_size, fc_hidden_size)

        self.fc2 = torch.nn.Linear(fc_hidden_size, output_size)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embedding(x)

        x = x.permute((0, 2, 1))

        x = self.conv1(x)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.conv3(x)

        x = self.relu(x)

        x = x.permute((0, 2, 1))

        x = self.gru(x)[0].mean(dim=1)  # (batch_size, L, hidden_size)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x

    def predict(self, x: torch.tensor) -> torch.tensor:
        return self.softmax(self.forward(x))
