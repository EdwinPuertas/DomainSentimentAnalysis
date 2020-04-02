from sa_logic.support import Support
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TerminologyExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        sp = Support()
        self.terminology = sp.import_terminology(filename='bank_terminology')

    def fit(self, x, y=None):
        self.transform(x)
        return self

    def transform(self, df):
        data_list = []
        data = df.values.tolist()
        for row in data:
            msg = str(row)
            count = 0
            for term in self.terminology:
                if msg.find(term) > -1:
                    count += 1
            if count > 0:
                data_list.append([count, True])
            else:
                data_list.append([count, False])
        return np.array(data_list)

