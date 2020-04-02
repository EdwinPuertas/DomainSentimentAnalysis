import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion

# creation of array
s1 = np.array(['foo', 'bar', 'baz'])
s2 = np.array(['a', 'b', 'c'])
X = np.column_stack([s1, s2])
print('base array: \n', X, '\n')

# A fake example that appends a column (Could be a score, ...) calculated on specific columns from X
class DummyTransformer(TransformerMixin):
    def __init__(self, value=None):
        TransformerMixin.__init__(self)
        self.value = value

    def fit(self, *_):
        return self

    def transform(self, X):
        # appends a column (in this case, a constant) to X
        s = np.full(X.shape[0], self.value)
        X = np.column_stack([X, s])
        return X

# as such, the transformer gives what I need first
transfo = DummyTransformer(value=1)
tt = transfo.fit_transform(X)
print('single transformer: \n', tt , '\n')

# but when I try to chain them and create a pipeline I run into the replication of existing columns
stages = []
for i in range(2):
    transfo = DummyTransformer(value=i+1)
    stages.append(('step'+str(i+1),transfo))
pipeunion = FeatureUnion(stages)
pp = pipeunion.fit_transform(X)
print('Given result of the Feature union pipeline: \n', pp , '\n')
# columns 1&2 from X are replicated

# I would expect:
expected = np.column_stack([X, np.full(X.shape[0], 1), np.full(X.shape[0], 2) ])
print('Expected result of the Feature Union pipeline: \n', expected, '\n')