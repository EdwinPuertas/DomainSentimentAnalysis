from sa_logic.bank2vec_sa import Bank2Vec
from sa_logic.bow_sa import BoWSA
from sa_logic.dow import DomainOfWord
from sa_logic.tfidf_sa import TFIDFSA
from sa_logic.support import Support

list_tmp = [{'fold': 5, 'clean_data': True, 'stop_words': False},
              {'fold': 5, 'clean_data': False, 'stop_words': True},
              {'fold': 5, 'clean_data': False, 'stop_words': False},
              {'fold': 10, 'clean_data': True, 'stop_words': False},
              {'fold': 10, 'clean_data': False, 'stop_words': True},
              {'fold': 10, 'clean_data': False, 'stop_words': False}]
list_classifier = ['SVM', 'LogisticRegression', 'RandomForest', 'MLPClassifier', 'ExtraTreesClassifier']
result_models = []
for item in list_tmp:
    bow_sa = BoWSA(clean_data=item['clean_data'])
    dow = DomainOfWord(clean_data=item['clean_data'])
    tfidf_sa = TFIDFSA(clean_data=item['clean_data'])
    wv = Bank2Vec(clean_data=item['clean_data'])
    for i in range(1, 4):
        for ngram in range(2, 5):
            result_bow = []
            result_tfidf = []
            result_dow = []
            result_wv = []
            min_df = 3 * i
            model_name_bow = 'BoW_N@' + str(ngram) + '_df@' + str(min_df) + '_fold@' + str(item['fold'])
            model_name_bow += '_clean_data' if item['clean_data'] else '_raw_data'
            model_name_bow += '_stop_words' if item['stop_words'] else '_without_stop_words'
            print("#" * 10 + '| Model: ' + model_name_bow + ' |' + "#" * 10)
            result_bow = bow_sa.baseline(model_name=model_name_bow, list_classifier=list_classifier, ngram=(1, ngram), min_df=3,
                            max_features=5000, stop_words=item['stop_words'], fold=item['fold'], iteration=3)
            result_models.extend(result_bow)

            model_name_tfidf = 'TFIDF_N@' + str(ngram) + '_df@' + str(min_df) + '_fold@' + str(item['fold'])
            model_name_tfidf += '_clean_data' if item['clean_data'] else '_raw_data'
            model_name_tfidf += '_stop_words' if item['stop_words'] else '_without_stop_words'
            print("#" * 10 + '| Model: ' + model_name_tfidf + ' |' + "#" * 10)
            result_tfidf = tfidf_sa.baseline(model_name=model_name_tfidf, list_classifier=list_classifier, ngram=(1, ngram), min_df=3,
                              max_features=5000, stop_words=item['stop_words'], fold=item['fold'], iteration=3)
            result_models.extend(result_tfidf)

            model_name_dow = 'DoW_N@' + str(ngram) + '_df@' + str(min_df) + '_fold@' + str(item['fold'])
            model_name_dow += '_clean_data' if item['clean_data'] else '_raw_data'
            model_name_dow += '_stop_words' if item['stop_words'] else '_without_stop_words'
            print("#" * 10 + '| Model:' + model_name_dow + ' |' + "#" * 10)
            result_dow = dow.baseline(model_name=model_name_dow, list_classifier=list_classifier, ngram=(1, ngram), min_df=min_df,
                          max_features=5000, stop_words=item['stop_words'], fold=item['fold'], iteration=3)
            result_models.extend(result_dow)

    model_name_b2v = 'Bank2Vec' + '_fold@' + str(item['fold'])
    model_name_b2v += '_clean_data' if item['clean_data'] else '_raw_data'
    print("#" * 10 + '| Model: ' + model_name_b2v + ' |' + "#" * 10)
    result_wv = wv.baseline(model_name=model_name_b2v, list_classifier=list_classifier, fold=item['fold'], iteration=3)
    result_models.extend(result_wv)


Support.save_to_csv(result_models)