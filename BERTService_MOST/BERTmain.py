import os, time, json
from numpy import array
from TFIDF import TFIDF
from Initial import Initial
from Mymodel import Mymodel
from SaveLog import SaveLog
from ContentParser import ContentParser
from bert_serving.client import BertClient

corpus_amount = [1296, 1254, 1500, 1500, 1500, 1500]
def main():
    # initial dataset
    dataset, dataType, model = Initial().InitialMain()
    #TF_words, TF_scores = TFIDF(dataset).getTFIDFWeigths()
    TF_words, TF_scores = "", ""
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
                correct, count, ind, tTime, correct_list = 0, 0, 0, time.time(), []
                if isinstance(single_dataset, str):
                    typeChange+=1
                    Process_dataset = "Start processing dataset: " + single_dataset + "\n"
                    print(Process_dataset)
                    continue
                for single_data in single_dataset:
                    s_string, q_string, options, answer = ContentParser(single_data).getContent()
                    #print(single_data['storyName'])
                    guessAnswer = Mymodel(bc, s_string, q_string, options, m, TF_words, TF_scores, constant).MymodelMain()
                    if guessAnswer == answer:
                        correct += 1
                    count += 1
                    if count == corpus_amount[ind]:
                        print(count)
                        ind += 1
                        correct_list.append(correct)
                        correct, count = 0, 0
                Accuracy = "Accuracy: "
                for c in range(len(correct_list)):
                    accuracy = round(correct_list[c] / corpus_amount[c],3)
                    Accuracy += str(accuracy) + ", "
                    AccuracyList.append(accuracy)
                CostTime = "\nTotal cost time: "+ str(time.time()-tTime) + "\n\n"
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
