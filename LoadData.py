import json
import os

class LoadData():
    
    def __init__(self, dataType, dataSet):
        self.dataType = dataType
        self.dataSet = dataSet
        self.dataSetPath = os.path.join(os.listdir("~/Hail-Mary"),"")
    
    def getDataSet(self):
        try:
            self.checkArgs()
        except:
            return
        print("Start processing datatype: %s\nStart processing dataset: %s" % (self.dataType, self.dataSet))
        print(self.dataSetPath)
        dataSetPath = self.joinDataSetPath(self.dataSetPath, self.dataType, self.dataSet)
        with open(dataSetPath, 'r') as data:
            data = json.load(data)
            
        return data

    def joinDataSetPath(self, dataSetPath, dataType, dataSet):
        if dataType == "with":
            dataSetPath = os.path.join(dataSetPath,'data_with_punctuation', dataSet + ".json")
        elif dataType == "without":
            dataSetPath = os.path.join(dataSetPath, "data_without_punctuation", dataSet + ".json")
            print(dataSetPath)
        return dataSetPath

    def checkArgs(self):
        dataType = {"with", "without"}
        dataSet = {"train", "test", "dev"}
        if self.dataType not in dataType or self.dataSet not in dataSet:
            print("================================")
            print("Please check your input args.")
            print("Try [-d] [with/without] [-o] [train/test/dev].")
            print("================================")
        