import argparse, logging
from gensim.models import KeyedVectors
from LoadData import LoadData

class Initial():
    def __init__(self):
        
        self.max_dim_len = 300

    def InitialMain(self):
        dataset = self.GetDataset(self.ArgParse())
        word2vec = self.LoadWord2vec()
        return dataset, word2vec, self.max_dim_len
    def ArgParse(self):
        print("Start setting argparse.")
        # args setting
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--optional-dataType", help="optional dataType", dest="dataType", default="with")
        parser.add_argument("-o", "--optional-dataSet", help="optional dataSet", dest="dataSet", default="train")
        args = parser.parse_args()
        return args

    def GetDataset(self, args):
        print("Start getting dataset.")
        dataset = LoadData(args.dataType, args.dataSet).getDataSet()
        return dataset

    def LoadWord2vec(self):
        print("Start loading word2vec.")
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        # get vector with weight of words from w2v model
        model_weight_vector = model.wv
        return model_weight_vector