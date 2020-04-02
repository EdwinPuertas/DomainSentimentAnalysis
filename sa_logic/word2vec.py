from gensim.models import Word2Vec
from sa_logic.models import wor2vec_model
from sa_logic.support import Support
from sa_root import ROOT_DIR_DATA_EMBEDDING
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

class Wordk2VecSA:

    def __init__(self):
        print("Start Word2Vec - Sentiment Analysis")
        self.sp = Support()
        file_model = ROOT_DIR_DATA_EMBEDDING + "SBW-vectors-300-min5.txt"
        model_tmp = self.sp.load_vectors_from_csv(file_model)
        self.model = Word2Vec.load(model_tmp)
        self.data, self.label = self.sp.process_data(filename='bank_message', size_msg=3, clean=True, replace_text=True,
                                                     stemmed=None, lemmatize=None, spelling=None)


    def baseline(self,  num_features=300, fold=5, iteration=3):
        for model_type, classifier in wor2vec_model.items():
            sum_recall = 0.0
            sum_precision = 0.0
            sum_f1 = 0.0
            sum_accuracy = 0.0
            for i in range(0, iteration):
                x_train, x_test, y_train, y_test = train_test_split(self.data, self.label, test_size=0.25, random_state=1000)
                x_train, y_train = self.sp.balanced_data(x_train, y_train)
                x_test, y_test = self.sp.balanced_data(x_test, y_test)


                trainDataVecs = self.sp.getAvgFeatureVecs(x_train, self.model, num_features)
                testDataVecs = self.sp.getAvgFeatureVecs(x_test, self.model, num_features)

                classifier.fit(trainDataVecs, y_train)
                predict = classifier.predict(testDataVecs)

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

            recall = sum_recall / iteration
            precision = sum_precision / iteration
            f1 = sum_f1 / iteration
            accuracy = sum_accuracy / iteration
            self.sp.print_score(model_type, predicted_classes=predict, recall=recall, precision=precision, f1=f1,
                                   accuracy=accuracy, test=y_test)


if __name__=='__main__':
    wv = Wordk2VecSA()
    wv.baseline(fold=5, iteration=3)
