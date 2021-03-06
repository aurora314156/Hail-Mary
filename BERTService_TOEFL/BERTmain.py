import os, time, json
from numpy import array
from TFIDF import TFIDF
from Initial import Initial
from Mymodel import Mymodel
from SaveLog import SaveLog
from ContentParser import ContentParser
from bert_serving.client import BertClient


def main():
    # initial dataset
    dataset, dataType, model = Initial().InitialMain()
    TF_words, TF_scores = TFIDF(dataset).getTFIDFWeigths()
    # initial BERT model
    bc = BertClient()
    AccuracyList = []
    constant = 0
    for m in model:
        for i in range(1):
            print("***********************************\nStart getting datatype: ")
            print(dataType)
            print("***********************************\n")
            model = "Start run model: " + m + "\n"
            print(model)
            typeChange=0
            for single_dataset in dataset:
                correct, tTime = 0, time.time()
                if isinstance(single_dataset, str):
                    typeChange+=1
                    Process_dataset = "Start processing dataset: " + single_dataset + "\n"
                    print(Process_dataset)
                    continue
                for single_data in single_dataset:
                    s_string, q_string, options, answer = ContentParser(single_data).getContent()
                    guessAnswer = Mymodel(bc, s_string, q_string, options, m, TF_words, TF_scores, constant).MymodelMain()
                    if guessAnswer == answer:
                        print(single_data['storyName'])
                        correct += 1
                accuracy = round(correct/len(single_dataset),3)
                Accuracy = "Accuracy: " + str(accuracy) + "\n"
                CostTime = "Total cost time: "+ str(time.time()-tTime) + "\n\n"
                AccuracyList.append(accuracy)
                print(Accuracy)
                print(CostTime)
                if typeChange <4:
                    dataTypeLog = "Data type: " + dataType[0] + "\n"
                else:
                    dataTypeLog = "Data type: " + dataType[1] + "\n"
                SaveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime).saveLogTxt()
            constant += 0
    SaveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime, AccuracyList).saveLogExcel()
            

if __name__ == "__main__":
    main()
