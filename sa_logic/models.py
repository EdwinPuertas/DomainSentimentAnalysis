from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

class ClassifierModels:

    def __init__(self, classifier_name):
        try:
            self.classifiers = {}
            for m in classifier_name:
                if m == 'SVM':
                    svm = SVC(kernel='linear', C=0.5, random_state=0)
                    # classifier_svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr',
                    #                      degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False,
                    #                      random_state=None,shrinking=True,tol=0.001, verbose=False)
                    self.classifiers['SVM'] = svm
                elif m == 'GaussianNB':
                    gnb = GaussianNB()
                    self.classifiers['GaussianNB'] = gnb
                elif m == 'MultinomialNB':
                    nb = MultinomialNB()
                    self.classifiers['MultinomialNB'] = nb
                elif m == 'LogisticRegression':
                    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=500, random_state=0, n_jobs=-1)
                    self.classifiers['LogisticRegression'] = lr
                elif m in 'RandomForest':
                    rf = RandomForestClassifier(bootstrap=True, n_estimators=400, max_depth=70, max_features='auto',
                                                       oob_score=True, min_samples_leaf=4, min_samples_split=10, n_jobs=-1)
                    self.classifiers['RandomForest'] = rf

                elif m in 'MLPClassifier':
                    mlp = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto', beta_1=0.9,
                                                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                                  hidden_layer_sizes=(50, 50, 50), learning_rate='constant',
                                                  learning_rate_init=0.001,momentum=0.9,
                                                  nesterovs_momentum=True, power_t=0.5, random_state=None,
                                                  shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                                                  verbose=False, warm_start=False)

                    # classifier_mlp = MLPClassifier(solver='adam', activation='relu',alpha=1e-4, hidden_layer_sizes=(50, 50, 50),
                    #                                 random_state=1, max_iter=3, verbose=10,learning_rate_init=.1)
                    self.classifiers['MLPClassifier'] = mlp

                elif m in 'ExtraTreesClassifier':
                    et = ExtraTreesClassifier(n_estimators=700, max_features=500, criterion='entropy',
                                               min_samples_split=5, max_depth=50, min_samples_leaf=5)
                    self.classifiers['ExtraTreesClassifier'] = et
        except Exception as e:
            print("\t ERROR load_models: ", e)
