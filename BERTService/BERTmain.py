import os, time, json
from numpy import array
from Initial import Initial
from Mymodel import Mymodel
from bert_serving.client import BertClient

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

def main():
    # initial dataset
    dataset, dataType, model, d_model = Initial().InitialMain()
    # initial BERT model
    bc = BertClient()
    for m in model:
        print(dataType)
        model = "Start run model: " + m
        print(model)
        for single_dataset in dataset:
            correct, tTime = 0, time.time()
            if isinstance(single_dataset, str):
                Process_dataset = "Start processing dataset: " + single_dataset + "\n"
                print(Process_dataset)
                continue
            for single_data in single_dataset:
                print(single_data['storyName'])
                s_string, q_string, options, answer = getContent(single_data)
                guessAnswer = Mymodel(bc, s_string, q_string, options, m).MymodelMain()
                if guessAnswer == answer:
                    correct += 1
            Accuracy = "Accuracy: " + str(correct/len(single_dataset))
            CostTime = "Total cost time: "+ str(time.time()-tTime) + "\n"
            print(Accuracy)
            print(CostTime)
            saveLog(dataType, Process_dataset, model, Accuracy, CostTime)

if __name__ == "__main__":
    main()
