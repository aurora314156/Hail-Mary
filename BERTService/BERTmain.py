import os, time, json
from numpy import array
from Initial import Initial
from Mymodel import Mymodel
from bert_serving.client import BertClient
import xlwt
from xlwt import Workbook
from xlrd import open_workbook
from xlutils.copy import copy

def getContent(single_data):
    
    s_string, q_string, o_string, options = "", "", "", []

    for story_element in single_data['story']:
        s_string += story_element + " "
    for question_element in single_data['question']:
        q_string += question_element + " "
    for option in single_data['options']:
        o_string = ""
        for single_option_element in option:
            o_string += single_option_element + " "
        options.append(o_string[:-1])
        
    answer = single_data['answer']
    return s_string[:-1], q_string[:-1], options, answer

def saveLog(dataType, Process_dataset, model, Accuracy, CostTime):
    with open('log.txt', 'a') as log:
        log.write(dataType)
        log.write(Process_dataset)
        log.write(model)
        log.write(Accuracy)
        log.write(CostTime)

def saveLogExcel(AccuracyList):
    rb = open_workbook("experiment.xls")
    wb = copy(rb)

    sheet = wb.get_sheet(0)

    sheet.write(0, 0, 'model') 
    sheet.write(0, 1, 'test') 
    sheet.write(0, 2, 'train') 
    sheet.write(0, 3, 'dev') 
    sheet.write(0, 4, '')
    sheet.write(0, 5, 'test')
    sheet.write(0, 6, 'train')
    sheet.write(0, 7, 'dev')
    for i in range(1,11):
        sheet.write(i, 0, str(i))

    ind_x, ind_y = 1, 1
    for a in AccuracyList:
        sheet.write(ind_y, ind_x, str(a))
        ind_x += 1
        if ind_x % 8 == 0:
            ind_y += 1
            ind_x = 1
            continue
        if ind_x % 4 == 0:
            ind_x +=1
            continue
        

    wb.save('experiment.xls')

def main():
    # initial dataset
    dataset, dataType, model, d_model = Initial().InitialMain()
    # initial BERT model
    bc = BertClient()
    AccuracyList = []
    for m in model:
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
                print(single_data['storyName'])
                s_string, q_string, options, answer = getContent(single_data)
                guessAnswer = Mymodel(bc, s_string, q_string, options, m).MymodelMain()
                if guessAnswer == answer:
                    correct += 1
            Accuracy = "Accuracy: " + str(correct/len(single_dataset)) + "\n"
            CostTime = "Total cost time: "+ str(time.time()-tTime) + "\n\n"
            AccuracyList.append(correct/len(single_dataset))
            print(Accuracy)
            print(CostTime)
            if typeChange <4:
                dataTypeLog = "Data type: " + dataType[0] + "\n"
            else:
                dataTypeLog = "Data type: " + dataType[1] + "\n"

            saveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime)
    saveLogExcel(AccuracyList)
            

if __name__ == "__main__":
    main()
