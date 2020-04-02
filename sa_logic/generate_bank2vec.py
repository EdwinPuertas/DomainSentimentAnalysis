import multiprocessing
from sa_logic.support import Support
from gensim.models import Word2Vec
from sa_root import ROOT_DIR_DATA_OUTPUT
from sa_root import ROOT_DIR_DATA_EMBEDDING
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

class GenerateBank2Vec:

    def __init__(self):
        self.sp = Support()
        self.lexicon = self.sp.load_file_polarity()
        self.cores = multiprocessing.cpu_count()
        self.data = Support.process_data(filename='bank_message', size_msg=3, clean_data=True, replace_text=True,
                                         stemmed=None, lemmatize=None, spelling=None)

    def generate_embedding(self, size=300, min_count=3, window=5, downsampling=1e-3, negative=3):
        try:
            full_corpus = []
            corpus_ccd = Support.import_corpus_ccd()
            corpus_wikipedia_web = Support.corpus_bank_xml()
            corpus_bank = Support.import_dataset_bank()
            dataset = self.data['msg'].values.tolist()
            full_corpus.extend(corpus_ccd)
            full_corpus.extend(corpus_wikipedia_web)
            full_corpus.extend(corpus_bank)
            full_corpus.extend(dataset)
            print('Transform sentences to vectors ...')
            corpus = self.sp.sentence_vec(full_corpus)

            terminology = self.sp.import_terminology_embedding(filename='bank_terminology')
            print('Generate Embedding...')
            model = Word2Vec(corpus, cbow_mean=1, workers=self.cores, size=size,min_count=min_count,
                             window=window, sample=downsampling, negative=negative, iter=30)
            model.train(terminology, total_examples=model.corpus_count, epochs=model.epochs)

            model.init_sims(replace=True)
            # Saving the model for later use. Can be loaded using Word2Vec.load()
            model_name = ROOT_DIR_DATA_EMBEDDING + "bank2vec.model"
            model.save(model_name)
            print('Model generated sucesfull!')
        except Exception as e:
            print("\t ERROR generate_embedding: ", e)
            return None

if __name__=='__main__':
    embedding = GenerateBank2Vec()
    embedding.generate_embedding()