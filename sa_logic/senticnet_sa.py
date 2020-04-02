from sklearn.model_selection import train_test_split
from sa_logic.support import Support
from senticnet.senticnet import SenticNet

class SenticNetSA:

    def __init__(self):
        print("Start SenticNet - Sentiment Analysis")
        self.sp = Support()
        self.sn = SenticNet()
        self.corpus = self.sp.import_corpus_bank()
        self.terminology = self.sp.import_bank_terminology(filename='bank_terminology')
        self.data, self.label = self.sp.process_data(filename='bank_message',
                                                size_msg=3,
                                                clean=True,
                                                replace_text=True,
                                                stemmed=None,
                                                lemmatize=None,
                                                spelling=None)

    def baseline(self):
        TP = 0
        FP = 0
        FN = 0
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.label, test_size=0.20, random_state=1000)
        for i in range (0, len(x_train)):
            msg = str(x_train[i])
            value = float(y_train[i])
            result = self.sn.message_concept(msg)
            polarity_value = float(result['polarity_value'])
            polarity_value = 0.0 if polarity_value < 0.10 or polarity_value > -0.1 else polarity_value
            if value == polarity_value:
                TP += 1
            else:
                FP += 1
                if value == 1 and (polarity_value == 0.0 or polarity_value == -1.0):
                    FN += 1
                elif value == 0.0 and (polarity_value == 1 or polarity_value == -1.0):
                    FN += 1
                elif value == -1.0 and (polarity_value == 0.0 or polarity_value == 1.0):
                    FN += 1

        precision = TP/(TP + FP)
        recall = TP / (TP + FN)
        f1 = 2*((precision*recall) / (precision + recall))
        print("f1-score : {}%".format(round(f1 * 100, 2)))

if __name__=='__main__':
    sa = SenticNetSA()
    sa.baseline()



