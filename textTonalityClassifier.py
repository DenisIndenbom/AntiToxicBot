import numpy as np
from catboost import CatBoostClassifier
from fuzzywuzzy import process, fuzz

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