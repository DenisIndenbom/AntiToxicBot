from abc import ABC, abstractmethod

from nltk.tokenize import WordPunctTokenizer
from navec import Navec

from text_tonality_classifier import TextTonalityClassifierNN, TextTonalityClassifierByRules

from torch import load as load_nn
from torch import device as torch_device
from torch import cuda, long, no_grad, as_tensor

import numpy as np

import re

__all__ = ['ToxicityClassifier', 'NNClassifier', 'RulesClassifier']

class ToxicityClassifier(ABC):
    @abstractmethod
    def check_is_toxic(self, text: str) -> bool:
        return False


class RulesClassifier(ToxicityClassifier):
    def __init__(self, path_to_badwords: str, bad_word_threshold: int = 0.75):
        with open(path_to_badwords, 'r') as file:
            badwords = file.read().split('\n')

        self.model = TextTonalityClassifierByRules(badwords, bad_word_threshold)

        self.tokenizer = WordPunctTokenizer()

    def check_is_toxic(self, text: str) -> bool:
        tokenized_data = self.tokenizer.tokenize(text)

        return self.model.predict(tokenized_data)


class NNClassifier(ToxicityClassifier):
    def __init__(self, *, gpu: bool, message_toxicity_threshold: float, model_path: str, navec_path: str):
        self.tokenizer = WordPunctTokenizer()
        self.device = torch_device('cuda:0' if gpu and cuda.is_available() else 'cpu')

        # init navec model
        self.navec_model = Navec.load(navec_path)

        # init model
        self.model = TextTonalityClassifierNN(300, 512, 256, 2, self.navec_model)
        self.model.load_state_dict(load_nn(model_path, map_location=torch_device('cpu')))
        self.model.eval()
        self.model.to(self.device)

        self.message_toxicity_threshold = message_toxicity_threshold

    def __get_text_indexes(self, words: list) -> np.array:
        indexes = []

        for word in words:
            try:
                indexes.append(self.navec_model.vocab[word])
            except KeyError:
                indexes.append(self.navec_model.vocab.unk_id)

        return np.array(indexes, dtype=np.int64)

    # check the text for toxicity
    def check_is_toxic(self, text: str) -> bool:
        text = re.sub('[^a-zа-я]', ' ', text.lower())

        tokenized_data = self.tokenizer.tokenize(text)

        if len(tokenized_data) == 0:
            return False

        x = self.__get_text_indexes(tokenized_data)

        with no_grad():
            x = as_tensor(x, dtype=long).to(self.device)
            x = x.unsqueeze(0)
            probability_of_toxicity = self.model.predict(x)[0][1]  # we take the predicted probability of toxicity

        y = float(probability_of_toxicity) > self.message_toxicity_threshold

        return y
