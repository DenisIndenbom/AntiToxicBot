import numpy as np
from fuzzywuzzy import process, fuzz

from catboost import CatBoostClassifier

import torch
from slovnet.model.emb import NavecEmbedding

class RulesClassifier:
    """
            The rules tonality model.
    """
    def __init__(self, bad_words: list):
        """
        :param bad_words: list of bad words
        """
        self.list_of_bad_words = bad_words

    def predict(self, x: list):
        """
        :param x: input 2d list with the str. Example [['Hello','my',friends'],['My','name','is','Jack']]

        :return: numpy array with predictions. Example np.array([0,0])
        """
        y = []
        for row in x:
            in_list = False
            for word in row:
                #print(word)
                #process.extractOne()
                #if word in self.list_of_bad_words:
                #    in_list = True
                #    break
                if process.extractOne(word,self.list_of_bad_words)[1] > 75:
                    in_list = True
                    break
            if in_list:
                y.append(1)
            else:
                y.append(0)
        return np.array(y)

class CBClassifier:
    def __init__(self, model_path: str):
        """
        :param model_path: path to catboost model
        """
        self.cb_clf = CatBoostClassifier()
        self.cb_clf.load_model(model_path, format='cbm')

    def predict(self,x):
        """
        :param x: input 1d array-like or 2d array-like.
        :return: numpy array with predictions.
        """
        x = self.cb_clf.predict(x)
        return x


class TextClassifierNN(torch.nn.Module):

    def __init__(self, input_size=519, embedding_dim=300, gru_hidden_size=256, fc_hidden_size=512, output_size=2, navec=None):
        super(TextClassifierNN, self).__init__()

        self.relu = torch.nn.ReLU()

        self.softmax = torch.nn.Softmax()

        self.flatten = torch.nn.Flatten()

        self.embedding = NavecEmbedding(navec)  # torch.nn.Embedding(input_size, embedding_dim)

        self.conv1 = torch.nn.Conv1d(embedding_dim, 512, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(512, 1024, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv1d(1024, 2048, kernel_size=3, padding=1)

        self.gru = torch.nn.GRU(2048, gru_hidden_size, batch_first=True)

        self.fc1 = torch.nn.Linear(gru_hidden_size * input_size, fc_hidden_size)

        self.fc2 = torch.nn.Linear(fc_hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)

        x = x.permute((0, 2, 1))

        x = self.conv1(x)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.conv3(x)

        x = self.relu(x)

        x = x.permute((0, 2, 1))

        x = self.gru(x)[0]  # (seql, batch_size, hidden_size)

        x = self.flatten(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x

    def predict(self, x):
        return self.softmax(self.forward(x))
