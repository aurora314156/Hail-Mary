import json
import os
from pathlib import Path

class LoadData():
    
    def __init__(self, dataType, dataSetList):
        self.dataType = dataType
        self.dataSetList = dataSetList
        #self.dataSetPath = os.path.join(os.path.dirname("/home/wirl/Desktop/Hail-Mary/Setting.txt"),"")
        self.dataSetPath = os.path.join(os.path.dirname("/project/Divh/Hail-Mary/data_wtih_punctuation"),"")

    def getDataSetMain(self):
        print(self.dataType)
        print(self.dataSetList)
        try:
            bug = self.checkArgs()
        except bug == 1:
            return
        
        # iterator for get all selection dataset
        dataSet = []
        for dataType in self.dataType:
            print("***********************************\nStart processing datatype: " + dataType + "\n")
            for dataName in self.dataSetList:
                dataSetPath = self.joinDataSetPath(self.dataSetPath, dataType, dataName)
                print(dataSetPath)
                with open(dataSetPath, 'r') as d:
                    data = json.load(d)

                dataSet.append(dataName)
                dataSet.append(data)
        
        print(dataSet[0])
        print(dataSet[2])
        print(dataSet[4])

        return dataSet, self.dataType

    def joinDataSetPath(self, dataSetPath, dataType, dataName):
        if dataType == "with":
            dataSetPath = os.path.join(dataSetPath,'data_with_punctuation', dataName + ".json")
        elif dataType == "without":
            dataSetPath = os.path.join(dataSetPath, "data_without_punctuation", dataName + ".json")
        return dataSetPath

    def checkArgs(self):
        dataType = {"with", "without"}
        allDataSetName = {"train","test","dev"}
        bug = 0
        for d in self.dataSetList:
            if d not in allDataSetName:
                print("================================")
                print("Please check your input args.")
                print("Try [-d] [with/without/all] [-o] [train/test/dev/all].")
                print("================================")
                bug = 1

        for t in self.dataType:
            if t not in allDataSetName:
                print("================================")
                print("Please check your input args.")
                print("Try [-d] [with/without/all] [-o] [train/test/dev/all].")
                print("================================")
                bug = 1

        return bug