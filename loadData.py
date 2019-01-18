import json
import os

class loadData():
    
    def __init__(self, dataType, dataSet):
        self.dataType = dataType
        self.dataSet = dataSet
        self.dataSetPath = os.path.join(os.getcwd(),"")
    
    def getDataSet(self):
        self.checkArgs();
        dataSetPath = self.joinDataSetPath(self.dataSetPath, self.dataType, self.dataSet)
        with open(dataSetPath, 'r') as data:
            data = json.load(data)
            
        return data

    def joinDataSetPath(self, dataSetPath, dataType, dataSet):
        if dataType == "with":
            dataSetPath = os.path.join(dataSetPath,'data_with_punctuation', dataSet + ".json")
        elif dataType == "without":
            dataSetPath = os.path.join(dataSetPath, "data_without_punctuation", dataSet + ".json")
        
        return dataSetPath

    def checkArgs(self):
        dataType = {"with", "without"}
        dataSet = {"train", "test", "dev"}
        if self.dataType not in dataType or self.dataSet not in dataSet:
            print("================================")
            print("Please check your input args.")
            print("Try [-d] [with/without] [-o] [train/test/dev].")
            print("================================")
        