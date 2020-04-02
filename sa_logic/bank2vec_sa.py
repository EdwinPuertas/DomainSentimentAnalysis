import multiprocessing
import time
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sa_logic.models import ClassifierModels
from sa_logic.support import Support
from sa_root import ROOT_DIR_DATA_EMBEDDING
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

class Bank2Vec:

    def __init__(self, clean_data=True):
        self.sp = Support()
        file_model = ROOT_DIR_DATA_EMBEDDING + "bank2vec.model"
        self.model = Word2Vec.load(file_model)
        self.data = Support.process_data(filename='bank_message', size_msg=3, clean_data=clean_data,
                                         replace_text=True, stemmed=None, lemmatize=None, spelling=None)


    def baseline(self, model_name, list_classifier, num_features=300, fold=5, iteration=3):
        try:
            start_time = time.time()
            result = []
            models = ClassifierModels(list_classifier).classifiers
            for classifier_name, classifier in models.items():
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

                trainDataVecs = self.sp.getAvgFeatureVecs(x_train, self.model, num_features)
                testDataVecs = self.sp.getAvgFeatureVecs(x_test, self.model, num_features)

                classifier.fit(trainDataVecs, y_train)
                predict = classifier.predict(testDataVecs)

                sum_recall = 0.0
                sum_precision = 0.0
                sum_f1 = 0.0
                sum_accuracy = 0.0
                for i in range(0, iteration):
                    # Recall Scores
                    recall_scores = cross_val_score(classifier, trainDataVecs, y_train, cv=fold, scoring='recall_macro')
                    sum_recall += recall_scores
                    # Precision Score
                    precision_score = cross_val_score(classifier, trainDataVecs, y_train, cv=fold, scoring='precision_weighted')
                    sum_precision += precision_score
                    # F1 Score
                    f1_score = cross_val_score(classifier, trainDataVecs, y_train, cv=fold, scoring='f1_weighted')
                    sum_f1 += f1_score
                    # Accuracy Score
                    accuracy_score = cross_val_score(classifier, trainDataVecs, y_train, cv=fold, scoring='balanced_accuracy')
                    sum_accuracy += accuracy_score

                # Calculated Scores
                dict_data = {}
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
                print(dict_data)
                result.append(dict_data)
            return result
        except Exception as e:
            print("\t ERROR baseline Bank2Vec: ", e)

if __name__ == '__main__':
    result_models = []
    list_tupla = [(5, True), (10, True), (5, False), (10, False)]
    list_classifier = ['SVM', 'LogisticRegression', 'RandomForest', 'MLPClassifier', 'ExtraTreesClassifier']
    for fold, clean_data in list_tupla:
        result_tmp = []
        wv = Bank2Vec(clean_data=clean_data)
        model_name_b2v = 'Bank2Vec' + '_fold@' + str(fold)
        model_name_b2v += '_clean_data' if clean_data else '_raw_data'
        print("#" * 10 + '| Model: ' + model_name_b2v + ' |' + "#" * 10)
        result_tmp = wv.baseline(model_name=model_name_b2v, list_classifier=list_classifier, fold=fold, iteration=3)
        result_models.extend(result_tmp)
    Support.save_to_csv(result_models)
