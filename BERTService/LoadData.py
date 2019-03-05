import json
import os
from pathlib import Path

class LoadData():
    
    def __init__(self, dataType, dataSetList):
        self.dataType = dataType
        self.dataSetList = dataSetList
        self.dataSetPath = os.path.join(os.path.dirname("/home/wirl/Desktop/Hail-Mary/Setting.txt"),"")

    def getDataSet(self):
        try:
            self.checkArgs()
        except:
            return
        print("***********************************\nStart processing datatype: %s\n" % (self.dataType))
        # iterator for get all selection dataset
        dataSet = []
        for dataName in self.dataSetList:
            dataSetPath = self.joinDataSetPath(self.dataSetPath, self.dataType, dataName)
            with open(dataSetPath, 'r') as d:
                data = json.load(d)

            dataSet.append(dataName)
            dataSet.append(data)

        return dataSet

    def joinDataSetPath(self, dataSetPath, dataType, dataName):
        if dataType == "with":
            dataSetPath = os.path.join(dataSetPath,'data_with_punctuation', dataName + ".json")
        elif dataType == "without":
            dataSetPath = os.path.join(dataSetPath, "data_without_punctuation", dataName + ".json")
        return dataSetPath

    def checkArgs(self):
        dataType = {"with", "without"}
        allDataSetName = {"train","test","dev"}
        if self.dataType not in dataType or self.dataSetList[0] not in allDataSetName:
            print("================================")
            print("Please check your input args.")
            print("Try [-d] [with/without] [-o] [train/test/dev/all].")
            print("================================")
