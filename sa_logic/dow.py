import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sa_logic.models import ClassifierModels
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sa_logic.support import Support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sa_logic.lexicon_extractor import LexiconExtractor
from sa_logic.terminology_extractor import TerminologyExtractor
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class DomainOfWord:

    def __init__(self, clean_data=True):
        self.sp = Support()
        self.data = self.sp.process_data(filename='bank_message',size_msg=3,clean_data=clean_data,
                                               replace_text=True, stemmed=None,lemmatize=None, spelling=None)



    def baseline(self, model_name, list_classifier, ngram= None, min_df=3, max_features=5000,
                 fold=5, stop_words=None, iteration=3):

        try:
            start_time = time.time()
            result = []
            ngram = (1, 2) if ngram is None else ngram
            stopw = set(stopwords.words("spanish")) if stop_words is not None else None
            models = ClassifierModels(list_classifier).classifiers
            for classifier_name, classifier in models.items():
                dict_data = {}
                x_train, x_test, y_train, y_test = train_test_split(self.data['msg'], self.data['label'], test_size=0.25,
                                                                    random_state=1000)
                # training and sa_test are balanced
                df_train = pd.DataFrame({'msg': x_train, 'label': y_train})
                df_test = pd.DataFrame({'msg': x_test, 'label': y_test})
                train = self.sp.balanced_data(df_train)
                x_train = train['msg']
                y_train = train['label']
                test = self.sp.balanced_data(df_test)
                x_test = test['msg']
                y_test = test['label']
                vec_bow = CountVectorizer(analyzer='word', lowercase=True, encoding='utf-8', ngram_range=ngram,
                                          min_df=min_df, max_features=max_features, stop_words=stopw)

                vec_tfidf = TfidfVectorizer(analyzer='word', lowercase=True, encoding='utf-8', ngram_range=ngram,
                                            min_df=min_df, max_features=max_features, stop_words=stopw)


                lexicon = LexiconExtractor()
                terminology = TerminologyExtractor()

                vec_bow.fit(x_train)
                vec_tfidf.fit(x_train)
                lexicon.fit(x_train)
                terminology.fit(x_train)

                pipeline = Pipeline([
                    ('feats', FeatureUnion([
                        ('bow', vec_bow),
                        ('tfidf', vec_tfidf),
                        ('lexicon', lexicon),
                        ('terminology', terminology)
                    ],

                        # weight components in FeatureUnion
                        transformer_weights={
                            'bow': 0.2,
                            'tfidf': 0.2,
                            'lexicon': 0.3,
                            'terminology': 0.3
                        }
                    )),

                    (classifier_name, classifier)  # classifier
                ])

                pipeline.fit(x_train, y_train)
                predict = pipeline.predict(x_test)
                sum_recall = 0.0
                sum_precision = 0.0
                sum_f1 = 0.0
                sum_accuracy = 0.0
                for i in range(0, iteration):
                    print('Iteration {0} for Cross-Validation'.format(i+1))
                    #Recall
                    recall_scores = cross_val_score(pipeline, x_train, y_train, cv=fold, scoring='recall_macro', n_jobs=-1)
                    sum_recall += recall_scores
                    #Precision
                    precision_score = cross_val_score(pipeline, x_train, y_train, cv=fold, scoring='precision_weighted', n_jobs=-1)
                    sum_precision += precision_score
                    #F1
                    f1_score = cross_val_score(pipeline, x_train, y_train, cv=fold, scoring='f1_weighted', n_jobs=-1)
                    sum_f1 += f1_score
                    #Accuracy
                    accuracy_score = cross_val_score(pipeline, x_train, y_train, cv=fold, scoring='balanced_accuracy', n_jobs=-1)
                    sum_accuracy += accuracy_score

                # Calculated Scores
                dict_data['classifier_name'] = classifier_name

                recall = sum_recall / iteration
                dict_data['recall'] = round(np.mean(recall) * 100, 2)

                precision = sum_precision / iteration
                dict_data['precision'] = round(np.mean(precision) * 100, 2)

                f1 = sum_f1 / iteration
                dict_data['f1'] = round(np.mean(f1) * 100, 2)

                accuracy = sum_accuracy / iteration
                dict_data['accuracy'] = round(np.mean(accuracy) * 100, 2)

                # Calculated Time processing
                t_sec = round(time.time() - start_time)
                (t_min, t_sec) = divmod(t_sec, 60)
                (t_hour, t_min) = divmod(t_min, 60)
                time_processing = '{} hour:{} min:{} sec'.format(t_hour, t_min, t_sec)
                dict_data['time_processing'] = time_processing

                Support.print_score(dict_data=dict_data)
                dict_data['model_name'] = model_name
                result.append(dict_data)
            return result
        except Exception as e:
            print("\t ERROR baseline DomainOfWord: ", e)

# if __name__ == '__main__':
#     clean_data = True
#     base = DomainOfWord(clean_data=clean_data)
#     for i in range(1,3):
#         for ngram in range(2,5):
#             min_df = 3 * i
#             model_name = 'DoW_N@' + str(ngram) + '_df@' + str(min_df) + '_clean_data' if clean_data else '_raw_data'
#             list_classifier = ['SVM', 'MultinomialNB', 'LogisticRegression', 'RandomForest', 'MLPClassifier', 'ExtraTreesClassifier']
#             base.baseline(model_name=model_name, list_classifier=list_classifier, ngram=(1, ngram), min_df=min_df, max_features=5000, stop_words=True, fold=10, iteration=3)
