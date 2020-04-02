import pandas as pd

from sa_logic.support import Support
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LexiconExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        sp = Support()
        self.lexicon = sp.load_file_polarity()

    def fit(self, x, y=None):
        self.transform(x)
        return self

    def transform(self, df):
        features = []
        list_pos = self.lexicon['POSITIVE']
        list_neg = self.lexicon['NEGATIVE']
        data = df.values.tolist()
        for row in data:
            list_token = word_tokenize(row)
            count_pos = 0.0
            count_neg = 0.0
            tag = 0
            for word in list_token:
                if word in list_pos:
                    count_pos += 1
                elif word in list_neg:
                    count_neg += 1
            if count_pos > count_neg:
                tag = 1.0
            elif count_pos < count_neg:
                tag = -1
            elif count_pos == count_neg:
                tag = 0
            features.append([count_pos, count_neg, tag])
        result = pd.DataFrame(features)
        return result
