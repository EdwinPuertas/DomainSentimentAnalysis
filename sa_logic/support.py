import csv
import json
import datetime
import os
import re
import codecs
import timeit

import numpy as np
import pandas as pd
import spacy
from numpy import zeros, float32 as REAL
from gensim.models import keyedvectors
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from random import sample
from textblob import TextBlob
from sa_root import ROOT_DIR_DATA_INPUT
from sa_root import ROOT_DIR_DATA_OUTPUT
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from xml.etree import ElementTree as ET
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

class Support:
    def __init__(self):
        self.nlp = spacy.load('es')
        self.stop = set(stopwords.words("spanish"))
        self.dict_polarity = self.load_file_polarity()

    @staticmethod
    def clean_text(text):
        try:
            text = re.sub(r'((http|https|ftp|smtp).\/\/\d*|w{3}\..)|w{3}.', '', text) # Cleaning all the urls in the text
            text = re.sub(r'\w\d+|\d+\w|\d+|\d+\\|\/|\-|\s\d+|\w{22}', '', text)# Elimina número y númeron con letra
            text = re.sub(r'\©|\×|\⇔|\_|\»|\«|\~|\!|\@|\#|\$|\€|\&|\(|\)', '', text)# Elimina caracteres especilaes
            text = re.sub(r'\-|\/|\;|\:|\’|\‘|\Â|\�|\”|\“|\"|\'|\`|\}|\{|\[|\]', '', text)# Elimina caracteres especilaes
            text = re.sub(r'[\+\*\=\^\%\<\>\?\,\.]\?\¿', '', text)  # Elimina operadores
            #text = re.sub(r'\s[ \t]|\n\W', '', text)  # Eliminamos los espacios en blanco y tabuladores al comienzo de cada línea
            text = re.sub(r'"]+', '', text) # Elimina comillas dobles
            text = re.sub(r'(.)\1{2,}', '', text)  # Elimina caracteres que se repiten mas de dos veces
            text = re.sub(r'\Bltr|\bltr', '', text)  # Elimina ltr de caualquier texto
            return text
        except Exception as e:
            print("\t ERROR clean_text: ", e)
            return None

    @staticmethod
    def remplace_text(text):
        try:
            text = str(text)
            text = re.sub(r'ACT_((\w*\d+|\d*\w+)+)', '[NOMBRE]', text)
            text = re.sub(r'NUM_((\w*\d+|\d*\w+)+)', '[NUMERO]', text)
            text = re.sub(r'PHO_((\w*\d+|\d*\w+)+)', '[TELEFONO]', text)
            text = re.sub(r'EML_((\w*\d+|\d*\w+)+)', '[EMAIL]', text)

            return text
        except Exception as e:
            print("\t ERROR replace_text: ", e)
            return None

    @staticmethod
    def stemming_text(text):
        try:
            tokens = word_tokenize(text)
            porter = PorterStemmer()
            stemmed = [porter.stem(word) for word in tokens]
            text = ' '.join(stemmed)
            return text
        except Exception as e:
            print("\t ERROR stemming_text: ", e)
            return None

    @staticmethod
    def lemmatize_text(text):
        try:
            # lemmatize
            tokens = word_tokenize(text)
            lm = WordNetLemmatizer()
            tokens = [lm.lemmatize(word) for word in tokens]
            text = ' '.join(tokens)
            return text
        except Exception as e:
            print("\t ERROR lemmatize_text: ", e)
            return None

    @staticmethod
    def spelling_correction(text):
        text = str(TextBlob(text).correct())
        return text

    @staticmethod
    def import_dataset_bank():
        try:
            file = ROOT_DIR_DATA_INPUT + 'chats_bank.json'
            result = []
            with open(file, newline='', encoding='UTF-8') as json_data:
                chat_json = json.load(json_data)
                for row in chat_json:
                    for item in row['messages_all']:
                        msg = str(item['text'])
                        msg = msg.lower()
                        msg = Support.remplace_text(msg)
                        msg = Support.clean_text(msg)
                        if msg != '':
                            result.append(msg)
            return result
        except Exception as e:
            print("\t ERROR import_dataset_bank: ", e)
            return None

    @staticmethod
    def process_data(filename, size_msg=3, clean_data=None, replace_text=None, stemmed=None, lemmatize=None, spelling=None):
        try:
            list_msg = []
            list_label =[]
            file = ROOT_DIR_DATA_INPUT + str(filename) + '.csv'
            with open(file, newline='', encoding='utf-8-sig',mode='r') as f:
                lines = csv.reader(f, delimiter=';')
                for line in lines:
                    label = str(line[1])
                    msg = str(line[0])
                    msg = msg.lower()
                    msg = Support.spelling_correction(msg) if spelling is not None else msg
                    msg = Support.remplace_text(msg) if replace_text is not None else msg
                    msg = Support.clean_text(msg) if clean_data is not None else msg
                    msg = Support.stemming_text(msg) if stemmed is not None else msg
                    msg = Support.lemmatize_text(msg) if lemmatize is not None else msg
                    if msg is not None:
                        size = len(msg.split(' '))
                        if size > size_msg:
                            list_msg.append(msg)
                            list_label.append(label)

            temp = pd.DataFrame({'msg': list_msg, 'label': list_label})
            result = Support.balanced_data(temp)
            #print('Dataset size {0}'.format(len(result)))
            return result
        except Exception as e:
            print("\t ERROR dataset_bank: ", e)
            return None

    @staticmethod
    def balanced_data(data, mim_size_msg=3):
        try:
            x_result = []
            y_result = []
            negatives = []
            positives = []
            neutrales = []
            for index, row in data.iterrows():
                msg = str(row['msg'])
                label = str(row['label'])
                size_msg = len(msg.split(' '))
                if size_msg >= mim_size_msg:
                    if label == '1':
                        positives.append([msg, label])
                    elif label == '-1':
                        negatives.append([msg, label])
                    else:
                        neutrales.append([msg, label])

            size = len(positives)
            if len(negatives) < size:
                size = len(negatives)
            elif len(neutrales) < size:
                size = len(neutrales)

            positives = sample(positives, k=size)
            neutrales = sample(neutrales, k=size)
            negatives = sample(negatives, k=size)
            data_balance = negatives + positives + neutrales
            for item in data_balance:
                x_result.append(item[0])
                y_result.append(item[1])
            result = pd.DataFrame({'msg': x_result, 'label': y_result})
            return result
        except Exception as e:
            print("\t ERROR balanced_data: ", e)
            return None

    def import_terminology(self, filename):
        try:
            result = {}
            file = ROOT_DIR_DATA_INPUT + str(filename) + '.csv'
            with open(file, newline='', encoding='utf-8-sig',mode='r') as f:
                lines = csv.reader(f, delimiter='\n')
                for item in lines:
                    text = str(item[0]).replace('\t','')
                    text = text.lower().lstrip()
                    #word_list = word_tokenize(text)
                    #filtered_words = [word for word in word_list if word not in set(stopwords.words("spanish"))]
                    #filtered_words = [word for word in word_list if word not in ['a', 'la', 'el']]
                    #text = ' '.join(filtered_words)
                    if text != '' and text not in result :
                        result[text] = 1

            d = {}
            result = [d.setdefault(x, x) for x in result if x not in d]
            return result

        except Exception as e:
            print("\t ERROR bank_terminology: ", e)
            return None

    def import_terminology_embedding(self, filename):
        try:
            result = []
            file = ROOT_DIR_DATA_INPUT + str(filename) + '.csv'
            with open(file, newline='', encoding='utf-8-sig',mode='r') as f:
                lines = csv.reader(f, delimiter='\n')
                for item in lines:
                    text = str(item[0]).replace('\t','')
                    text = text.lower().lstrip()
                    if text != '' and text not in result:
                        embedding = text.split(' ')
                        result.append(embedding)

            return result

        except Exception as e:
            print("\t ERROR bank_terminology: ", e)
            return None
    @staticmethod
    def load_file_polarity():
        try:
            dict_polarity = {}
            list_positive = []
            list_negative = []
            # TODO: Config File Lexicon in config file.
            with open(ROOT_DIR_DATA_INPUT + 'polarity_isol.csv', newline='') as csv_file:
                data_reader = csv.reader(csv_file, delimiter=';', quotechar='|')
                for row in data_reader:
                    # print(row)
                    if row[1] == '1':
                        list_positive.append(row[0])
                    elif row[1] == '-1':
                        list_negative.append(row[0])
                dict_polarity['POSITIVE'] = list_positive
                dict_polarity['NEGATIVE'] = list_negative
                return dict_polarity
            print('* load_file_polarity POSITIVE: {0} NEGATIVE: {1}  Words entity.'.format(len(list_positive), len(list_negative)))
        except Exception as e:
            print("\t ERROR load_file_polarity: ", e)
            return None

    @staticmethod
    def print_score(dict_data):
        try:
            print("#" * 10 + ' Classifier: ' + dict_data['classifier_name'] + " - " + "#" * 10)
            print('Accuracy:        {0}%'.format(dict_data['accuracy']))
            print('F1 score:        {0}%'.format(dict_data['f1']))
            print('Recall:          {0}%'.format(dict_data['recall']))
            print('Precision:       {0}%'.format(dict_data['precision']))
            print('Time Processing: {0}'.format(dict_data['time_processing']))
        except Exception as e:
            print("\t ERROR print_score: ", e)

    def generate_ngram(self, text, max_ngram):
        result = []
        word = ''
        for i in range(1, max_ngram):
            blob = TextBlob(text)
            ngram_var = blob.ngrams(n=i)
            word = ' '.join(ngram_var[0])
            result.append(word)
        return result

    def sentence_vec(self, data):
        try:
            result = []
            for text in data:
                doc = self.nlp(text)
                for stm in doc.sents:
                    stm = str(stm).rstrip()
                    vec_words = re.findall(r"[\w']+", stm)
                    if len(vec_words) > 0 and stm != '':
                        result.append(vec_words)
                        print('Sentence: {0} \nVector: {1}'.format(stm, vec_words))
            return result
        except Exception as e:
            print("\t ERROR load_file_polarity: ", e)

    # Function to average all word vectors in a paragraph
    def featureVecMethod(self, msg, model, num_features):
        try:
            # Pre-initialising empty numpy array for speed
            featureVec = np.zeros(num_features, dtype="float32")
            nwords = 1
            # Converting Index2Word which is a list to a set for better speed in the execution.
            index2word_set = set(model.wv.index2word)
            words = word_tokenize(msg, language='spanish')
            for word in words:
                if word in index2word_set:
                    nwords = nwords + 1
                    featureVec = np.add(featureVec, model[word])
            # Dividing the result by number of words to get average
            featureVec = np.divide(featureVec, nwords)
            return featureVec
        except Exception as e:
            print("\t ERROR featureVecMethod: ", e)
            return None

    # Function for calculating the average feature vector
    def getAvgFeatureVecs(self, messages, model, num_features):
        try:
            counter = 0
            msgFeatureVecs = np.zeros((len(messages), num_features), dtype="float32")
            for msg in messages:
                msgFeatureVecs[counter] = self.featureVecMethod(msg, model, num_features)
                counter = counter + 1
            return msgFeatureVecs
        except Exception as e:
            print("\t ERROR getAvgFeatureVecs: ", e)
            return None

    @staticmethod
    def import_corpus_ccd():
        try:
            output = []
            file = ROOT_DIR_DATA_INPUT + 'CCD_v7_esp.csv'
            with open(file, newline='', encoding='utf-8') as f:
                lines = csv.reader(f, delimiter=';', dialect='excel', )
                lines = list(lines)
                for item in lines:
                    text = str(item[2]).strip()
                    text = text.lower()
                    text = Support.clean_text(text)
                    if text != '':
                        output.append(text)
                        print(text)
            return output
        except Exception as e:
            print("\t ERROR import_corpus: ", e)

    @staticmethod
    def indent(elem, level=0):
        try:
            i = "\n" + level * "  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for elem in elem:
                    Support.indent(elem, level + 1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i
        except Exception as e:
            print("\t ERROR indent: ", e)
            return None

    @staticmethod
    def corpus_bank_xml(tag='row'):
        try:
            result = []
            path_input = ROOT_DIR_DATA_INPUT +'corpus_wikipedia_web.xml'
            file_input = open(path_input, encoding='utf8')
            tree = ET.parse(file_input)
            root = tree.getroot()
            for rows in root.iter(tag):
                for row in rows:
                    if row.tag == 'content':
                        text = row.text
                        text = Support.clean_text(text)
                        if text != ' ':
                            result.append(text)
                            print(text)
            return result
        except Exception as e:
            print("\t ERROR import_xml: ", e)
            return None

    @staticmethod
    def export_list(output_file, list):
        try:
            path_output = ROOT_DIR_DATA_OUTPUT + output_file + '.json'
            with open(path_output, 'w', encoding='UTF-8') as output:
                json.dump(list, output)
            print('JSON file successfully exported!')
        except Exception as e:
            print("\t ERROR export_cvs2: ", e)

    @staticmethod
    def import_corpus():
        try:
            file = ROOT_DIR_DATA_INPUT + 'corpus_bank.json'
            result = []
            with open(file, newline='', encoding='UTF-8') as json_data:
                messages = json.load(json_data)
                for row in messages:
                    msg = str(row)
                    msg = msg.lower()
                    msg = Support.remplace_text(msg)
                    msg = Support.clean_text(msg)
                    if msg != '':
                        result.append(msg)
            return result
        except Exception as e:
            print("\t ERROR import_dataset_bank: ", e)
            return None

    def load_vectors_from_csv(self, fname, vocab_size=973265, vector_size=100):
        print("Loading vectors from file: {0}".format(fname))
        result = keyedvectors.KeyedVectors(fname)
        result.syn0 = zeros((vocab_size, vector_size), dtype=REAL)
        result.vecor_size=vector_size
        counts=None
        def add_word(word, weights):
            word_id = len(result.vocab)
            if word in result.vocab:
                print("duplicate word '%s' in %s, ignoring all but first", word, fname)
                return
            if counts is None:
                # most common scenario: no vocab file given. just make up some bogus counts, in descending order
                result.vocab[word] = keyedvectors.Vocab(index=word_id, count=vocab_size - word_id)
            elif word in counts:
                # use count from the vocab file
                result.vocab[word] = keyedvectors.Vocab(index=word_id, count=counts[word])
            else:
                # vocab file given, but word is missing -- set count to None (TODO: or raise?)
                print("vocabulary file is incomplete: '%s' is missing", word)
                result.vocab[word] = keyedvectors.Vocab(index=word_id, count=None)
            result.syn0[word_id] = weights
            result.index2word.append(word)
        file=codecs.open(fname,"r","utf-8")
        i=0
        for line in file:
            i+=1
            if i==1: #ommit header
                continue
            parts=line.strip().split(",")
            word,weights=parts[1], [REAL(x) for x in parts[2:]]
            add_word(word,weights)
            if i%100000==0:
                print(i,"word vectors loaded so far ...")
        file.close()
        print(i-1,"word vectors loaded!")
        return result

    @staticmethod
    def save_to_csv(list_model):
        date_file = datetime.datetime.now().strftime("%Y-%m-%d")
        file_path_csv = ROOT_DIR_DATA_OUTPUT + "result_{1}.csv".format(date_file)
        type_file = 'a' if os.path.isfile(file_path_csv) else 'w'
        with open(file_path_csv, type_file) as out_csv:
            writer = csv.writer(out_csv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
            if type_file == 'w':
                writer.writerow(['model_name','accuracy_score', 'recall_score','precision_score'])
            for item in list_model:
             writer.writerow(item)

