import numpy as np
import keras,os, time, json, sys, codecs
from numpy import array
from keras_bert import load_trained_model_from_checkpoint
from keras_bert.layers import MaskedGlobalMaxPool1D
from Initial import Initial
from Mymodel import Mymodel

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
        options.append(o_string)
        
    answer = single_data['answer']
    return s_string, q_string, options, answer

def saveLog(dataType, Process_dataset, model, Accuracy, CostTime):
    with open('log.txt', 'a') as log:
        log.write(dataType)
        log.write(Process_dataset)
        log.write(model)
        log.write(Accuracy)
        log.write(CostTime)

def getModelAndToken():

    config_path, checkpoint_path, dict_path = tuple(['bertmodel/bert_config.json', 'bertmodel/bert_model.ckpt', 'bertmodel/vocab.txt'])

    model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    pool_layer = MaskedGlobalMaxPool1D(name='Pooling')(model.output)
    model = keras.models.Model(inputs=model.inputs, outputs=pool_layer)
    #model.summary(line_length=120)

    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    return model, token_dict

def main():

    # initial dataset
    dataset, dataType, run_model_list = Initial().InitialMain()
    for m in run_model_list:
        print(dataType)
        model_id = "Start run model: " + m + "\n"
        print(model_id)
        model, token_dict = getModelAndToken()
        for single_dataset in dataset:
            correct, tTime = 0, time.time()
            if isinstance(single_dataset, str):
                Process_dataset = "Start processing dataset: " + single_dataset + "\n"
                print(Process_dataset)
                continue
            for single_data in single_dataset:
                s_string, q_string, options, answer = getContent(single_data)
                guessAnswer = Mymodel(model, token_dict, s_string, q_string, options, model_id).MymodelMain()
                if guessAnswer == answer:
                    correct += 1
            Accuracy = "Accuracy: " + str(correct/len(single_dataset)) + "\n"
            CostTime = "Total cost time: "+ str(time.time()-tTime) + "\n"
            print(Accuracy)
            print(CostTime)
            saveLog(dataType, Process_dataset, model_id, Accuracy, CostTime)

if __name__ == "__main__":
    main()