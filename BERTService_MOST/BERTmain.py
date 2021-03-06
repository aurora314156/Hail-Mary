import os, time, json, shutil
from numpy import array
from TFIDF import TFIDF
from Initial import Initial
from Mymodel import Mymodel
from SaveLog import SaveLog
from ContentParser import ContentParser
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
from random import randint


# setting

lim_start = [7694, 0, 855, 1710, 2565, 3420, 4275, 5130, 5985, 6840]
lim_end = [8549, 854, 1709, 2564, 3419, 4274, 5129, 5984, 6839, 7693]

args_setting_max = ['-model_dir', '/project/Divh/chinese_L-12_H-768_A-12/',
                    '-graph_tmp_dir', '/project/Divh/tmp/',
                                     '-num_worker', '1',
                                     '-gpu_memory_fraction', '0.8',
                                     '-pooling_layer', '-12',
                                     '-pooling_strategy', 'REDUCE_MAX',
                                     '-port', '5557',
                                     '-port_out', '5558',
                                     '-max_seq_len', 'NONE',]
args_setting_mean = ['-model_dir', '/project/Divh/chinese_L-12_H-768_A-12/',
                     '-graph_tmp_dir', '/project/Divh/tmp/',
                                     '-num_worker', '1',
                                     '-gpu_memory_fraction', '0.8',
                                     '-pooling_layer', '-12',
                                     '-pooling_strategy', 'REDUCE_MEAN',
                                     '-port', '5557',
                                     '-port_out', '5558',
                                     '-max_seq_len', 'NONE',]

tmpargs = []
tmpargs.append(args_setting_max)
#tmpargs.append(args_setting_mean)

port, port_out = 5557, 5558
#port, port_out = 4000,4001
#port, port_out = 6006, 6007
showing_result_story_name = ""


def eraseBertTmpFiles():
    print(" ****** Erase bert tmp files ******")
    shutil.rmtree("/project/Divh/tmp")
    os.mkdir("/project/Divh/tmp")
    allFiles = os.listdir(os.getcwd())
    currentPath = os.getcwd()
    for a in allFiles:
        filePath = os.path.join(currentPath, a)
        if a[:3] == "tmp":
            shutil.rmtree(filePath)
            
def showInferenceVector(storyName, inferenceVector, guessAnswer, answer):
    if showing_result_story_name == storyName:
        print("InferenceVector length: ", len(inferenceVector))
        print("guessAns: ", guessAnswer)
        print("correctAns: ", answer)
        # save inferenceVector
        if len(inferenceVector) != 3:
            with open('inferenceVector.txt', 'a') as log:
                tmpV = ""
                for i in inferenceVector:
                    tmpV += str(i) + " "
                tmpV += "\n\n"
                log.write(tmpV)
            return 0
        else:
            with open('inferenceVector.txt', 'a') as log:
                tmpV = ""
                for i in inferenceVector:
                    for element in i:
                        tmpV += str(element) + " "
                    tmpV += "\n"
                log.write(tmpV)
            return 0
    return 1

# def showInferenceVector(storyName, inferenceVector, guessAnswer, answer):
#     if showing_result_story_name == storyName:
#         print("InferenceVector length: ", len(inferenceVector))
#         print("guessAns: ", guessAnswer)
#         print("correctAns: ", answer)
#         print(inferenceVector)
#         # save inferenceVector
#         with open('inferenceVector.txt', 'a') as log:
#             tmpV = ""
#             for i in inferenceVector:
#                 if type(i) == list:
#                     for element in i:
#                         tmpV += str(element) + " "
#                 else:
#                     tmpV += str(i) + " "
#                 tmpV += "\n"
#             log.write(tmpV)
#         return 0

def recordAnswer(recordAns, answer):
    with open('recordAnswer.txt', 'a') as log:
        tmpA =""
        for r in recordAns:
            tmpA += str(r) + ","
        tmpA += "\n"
        for a in answer:
            tmpA += str(a) + ","
        tmpA += "\n"
        log.write(tmpA)

def initialLogFile():
    with open('accuracyList.txt', 'w') as log:
        log.close()
    with open('inferenceVector.txt', 'w') as log:
        log.close()
    with open('recordAnswer.txt', 'w') as log:
        log.close()
    
def main():
    # initial dataset and log file
    dataset, dataType, model = Initial().InitialMain()
    initialLogFile()
    #TF_words, TF_scores = TFIDF(dataset).getTFIDFWeigths()
    TF_words, TF_scores = "", ""
    constant, bestAccuracy, bestStrategy, bestPool = 0, 0, "", ""
    
    for m in model:
        print("***********************************\nStart getting datatype: ")
        print(dataType)
        print("***********************************\n")
        model = "Start run model: " + m + "\n"
        print(model)
        for ps in tmpargs:
            AccuracyList = []
            for pool_layer in range(1,13):
                eraseBertTmpFiles()
                args = get_args_parser().parse_args(ps)
                print(args.pooling_strategy)
                setattr(args, 'pooling_layer', [-pool_layer])
                server = BertServer(args)
                server.start()
                print('wait until server is ready...')
                time.sleep(30)
                print('encoding...')
                # initial BERT model
                bc = BertClient(port=port, port_out=port_out)
                typeChange, wrongNumJson, dataTypeJson = 0, {}, 0
                for single_dataset in dataset:
                    tmpC, count, correct, ind, tTime, flag, correct_list, wrongNumList = 0, 0, 0, 0, time.time(), [], [], []
                    if isinstance(single_dataset, str):
                        typeChange+=1
                        Process_dataset = "Start processing dataset: " + single_dataset + "\n"
                        print(Process_dataset)
                        continue
                    for l in range(len(lim_start)):
                    #for l in range(1,2):
                        correct, count, wrongNum, recordAns, correctAns = 0, 0, {}, [], []
                        for single_data in single_dataset:
                            # storyName = int(single_data['storyName'].split(".")[0][2:])
                            # print(storyName)
                            if count >= lim_start[l] and count <= lim_end[l]:
                                #storyName = int(single_data['storyName'].split(".")[0][2:])
                                s_string, q_string, options, answer = ContentParser(single_data).getContent()
                                print(type(single_data['storyName']))
                                guessAnswer, inferenceVector = Mymodel(bc, s_string, q_string, options, m, TF_words, TF_scores, constant).MymodelMain()
                                # show inference vector
                                status = showInferenceVector(single_data['storyName'], inferenceVector, guessAnswer, answer)
                                if status == 0:
                                    break
                                if guessAnswer == answer:
                                    correct += 1
                                else:
                                    wrongNum[count] = single_data['storyName']
                                tmpC +=1
                                recordAns.append(guessAnswer)
                                correctAns.append(answer)
                            count += 1
                        recordAnswer(recordAns, correctAns)
                        wrongNumList.append(wrongNum)
                        print(str(round( correct / 855, 3)))
                        correct_list.append(str(round( correct / 855, 3)))
                    wrongNumJson[str(dataTypeJson)] = wrongNumList
                    Accuracy = "Accuracy: "
                    for c in correct_list:
                        Accuracy += c + ", "
                    CostTime = "\nTotal cost time: "+ str(time.time()-tTime) + "\n\n"
                    print(Accuracy)
                    print(CostTime)
                    if typeChange <4:
                        dataTypeLog = "Data type: " + dataType[0] + "\n"
                    else:
                        dataTypeLog = "Data type: " + dataType[1] + "\n"
                    #SaveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime).saveLogTxt()
                    dataTypeJson += 1
                with open('Experiment/'+ m +'_wrongNum.json', 'w') as json_file:
                    json.dump(wrongNumJson, json_file)
                # save accuracy list
                with open('accuracyList.txt', 'a') as log:
                    tmpAc = ""
                    for a in correct_list:
                        tmpAc += str(a) + " "
                    tmpAc += "\n"
                    log.write(tmpAc)
                print(Accuracy)
                bc.close()
                server.close()
    #SaveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime, AccuracyList).saveLogExcel()
            
if __name__ == "__main__":
    main()