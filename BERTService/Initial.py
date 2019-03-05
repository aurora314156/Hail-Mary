import argparse, logging
from gensim.models import KeyedVectors
from LoadData import LoadData

class Initial():
    def __init__(self):
        self.d_model = 512
        
    def InitialMain(self):
        dataset = self.GetDataset(self.ArgParse())
        #word2vec = self.LoadWord2vec()
        return dataset, self.d_model
    
    def ArgParse(self):
        print("Start setting argparse.")
        # args setting
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--optional-dataType", help="optional dataType", dest="dataType", default="without")
        parser.add_argument("-o", "--optional-dataSet", help="optional dataSet", dest="dataSet", default="train")
        args = parser.parse_args()
        return args

    def GetDataset(self, args):
        print("Start getting dataset.")
        dataSetList = []
        if args.dataSet == "all":
            dataSetList.append("test")
            dataSetList.append("train")
            dataSetList.append("dev")
        else:
            dataSetList.append(args.dataSet)
        
        dataset = LoadData(args.dataType, dataSetList).getDataSet()
        return dataset

    def LoadWord2vec(self):
        print("Start loading word2vec.")
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        # get vector with weight of words from w2v model
        model_weight_vector = model.wv
        return model_weight_vector