import argparse, logging, os
from gensim.models import KeyedVectors
from LoadData import LoadData

class Initial():
    def __init__(self):
        self.d_model = 512
        # choose run model
        self.model = ['NineteenthModel']
        #self.model = ['FirstModel','SecondModel', 'ThirdModel', 'ForthModel', 'FifthModel', 'SixthModel', 'SeventhModel', 'EighthModel', 'NinthModel', \
        # 'TenthModel', 'EleventhModel', 'TwelfthModel', 'ThirteenthModel', 'FourteenthModel', 'FifteenthModel', 'SixteenthModel']
    def InitialMain(self):
        self.createLogFile()
        dataset, dataType = self.GetDataset(self.ArgParse())
        #word2vec = self.LoadWord2vec()
        return dataset, dataType, self.model, self.d_model

    def ArgParse(self):
        print("***********************************")
        print("Start setting argparse.")
        # args setting
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--optional-dataType", help="optional dataType", dest="dataType", default="without")
        parser.add_argument("-o", "--optional-dataSet", help="optional dataSet", dest="dataSet", default="train")
        args = parser.parse_args()
        return args

    def GetDataset(self, args):
        print("Start getting dataset.")
        dataSetList, dataType = [], []
        if args.dataSet == "all":
            dataSetList.append("test")
            dataSetList.append("train")
            dataSetList.append("dev")
        else:
            dataSetList.append(args.dataSet)

        if args.dataType == "all":
            dataType.append("with")
            dataType.append("without")
        elif args.dataType != "all":
            dataType.append(args.dataType)

        dataset = LoadData(dataType, dataSetList).getDataSetMain()
        return dataset, dataType

    def LoadWord2vec(self):
        print("Start loading word2vec.")
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        # get vector with weight of words from w2v model
        model_weight_vector = model.wv
        return model_weight_vector

    def createLogFile(self):
        # create log file
        print("create log file")
        with open('log.txt', 'w') as log:
            log.close()