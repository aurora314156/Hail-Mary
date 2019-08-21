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

# setting
args_setting_max = ['-model_dir', '/project/Divh/uncased_L-24_H-1024_A-16/uncased_L-24_H-1024_A-16/',
                    '-graph_tmp_dir', '/project/Divh/tmp/',
                                     '-num_worker', '1',
                                     '-gpu_memory_fraction', '0.3',
                                     '-pooling_layer', '-12',
                                     '-pooling_strategy', 'REDUCE_MAX',
                                     '-port', '5557',
                                     '-port_out', '5558',
                                     '-max_seq_len', 'NONE',]
args_setting_mean = ['-model_dir', '/project/Divh/uncased_L-24_H-1024_A-16/uncased_L-24_H-1024_A-16/',
                     '-graph_tmp_dir', '/project/Divh/tmp/',
                                     '-num_worker', '1',
                                     '-gpu_memory_fraction', '0.3',
                                     '-pooling_layer', '-12',
                                     '-pooling_strategy', 'REDUCE_MEAN',
                                     '-port', '5557',
                                     '-port_out', '5558',
                                     '-max_seq_len', 'NONE',]

tmpargs = []
#tmpargs.append(args_setting_max)
tmpargs.append(args_setting_mean)

port, port_out = 5557, 5558
showing_result_story_name = ""

def eraseBertTmpFiles():
    shutil.rmtree("/project/Divh/tmp")
    os.mkdir("/project/Divh/tmp")
    allFiles = os.listdir(os.getcwd())
    currentPath = os.getcwd()
    for a in allFiles:
        filePath = os.path.join(currentPath, a)
        if a[:3] == "tmp":
            shutil.rmtree(filePath)
# for 25 26, 18 model
def showInferenceVector(storyName, inferenceVector, guessAnswer, answer, sentences):
    if showing_result_story_name == storyName:
        print("InferenceVector length: ", len(inferenceVector))
        print("guessAns: ", guessAnswer)
        print("correctAns: ", answer)
        print(inferenceVector)
        # save inferenceVector
        if len(inferenceVector) != 3:
            with open('inferenceVector.txt', 'a') as log:
                tmpV = ""
                for i in inferenceVector:
                    tmpV += str(i) + " "
                tmpV += "\n\n"
                for s in sentences:
                    tmpV += s + "\n"
                tmpV += "\n"
                log.write(tmpV)
            return 0
        else:
            with open('inferenceVector.txt', 'a') as log:
                tmpV = ""
                for i in inferenceVector:
                    for element in i:
                        tmpV += str(element) + " "
                    tmpV += "\n"
                for s in sentences:
                    tmpV += s + "\n"
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

def recordAnswer(recordName, recordAns, answer):
    with open('recordAnswer.txt', 'a') as log:
        tmpA =""
        for n in recordName:
            tmpA += str(n) + ","
        tmpA += "\n"
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
    TF_words, TF_scores = TFIDF(dataset).getTFIDFWeigths()
    constant, bestAccuracy, bestStrategy, bestPool = 0, 0, "", ""
    
    for m in model:
        print("***********************************\nStart getting datatype: ")
        print(dataType)
        print("***********************************\n")
        model = "Start run model: " + m + "\n"
        print(model)
        for ps in tmpargs:
            AccuracyList = []
            for pool_layer in range(11,12):
                eraseBertTmpFiles()
                args = get_args_parser().parse_args(ps)
                print(args.pooling_strategy)
                setattr(args, 'pooling_layer', [-pool_layer])
                server = BertServer(args)
                server.start()
                print('wait until server is ready...')
                time.sleep(10)
                print('encoding...')
                # initial BERT model
                bc = BertClient(port=port, port_out=port_out, show_server_config=False)
                typeChange=0
                for single_dataset in dataset:
                    correct, tTime = 0, time.time()
                    if isinstance(single_dataset, str):
                        typeChange+=1
                        Process_dataset = "Start processing dataset: " + single_dataset +"\n"
                        print(Process_dataset)
                        continue
                    recordAns, correctAns, recordName, sentences = [], [], [], []
                    for single_data in single_dataset:
                        s_string, q_string, options, answer = ContentParser(single_data).getContent()
                        #guessAnswer, inferenceVector = Mymodel(bc, s_string, q_string, options, m, TF_words, TF_scores, constant).MymodelMain()
                        guessAnswer, inferenceVector, sentences = Mymodel(bc, s_string, q_string, options, m, TF_words, TF_scores, constant).MymodelMain()
                        storyName = single_data['storyName']
                        print(single_data['storyName'])
                        # show inference vector
                        #status = showInferenceVector(storyName, inferenceVector, guessAnswer, answer)
                        status = showInferenceVector(storyName, inferenceVector, guessAnswer, answer, sentences)
                        if status == 0:
                            break
                        if guessAnswer == answer:
                            #print(single_data['storyName'])
                            correct += 1
                        recordName.append(single_data['storyName'])
                        recordAns.append(guessAnswer)
                        correctAns.append(answer)
                    recordAnswer(recordName, recordAns, correctAns)
                    print(len(single_dataset))
                    accuracy = round(correct/len(single_dataset),3)
                    if accuracy > bestAccuracy:
                        bestAccuracy = accuracy
                        bestPool = pool_layer
                        bestStrategy = args.pooling_strategy
                    Accuracy = "Accuracy: " + str(accuracy) + "\n"
                    CostTime = "Total cost time: "+ str(time.time()-tTime) + "\n"
                    AccuracyList.append(accuracy)
                    print(Accuracy)
                    print(CostTime)
                    if typeChange <4:
                        dataTypeLog = "Data type: " + dataType[0] + "\n"
                    else:
                        dataTypeLog = "Data type: " + dataType[1] + "\n"
                    #SaveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime).saveLogTxt()
                #SaveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime, AccuracyList).saveLogExcel()
                    print("Current pool: ", pool_layer)
                    print("Best Accuracy: ", bestAccuracy)
                    print("Best bestPool: ", bestPool)
                    print("Best bestStrategy: ", bestStrategy)
                    bc.close()
                    server.close()
                # save accuracy list
            with open('accuracyList.txt', 'a') as log:
                tmpAc = ""
                for a in AccuracyList:
                    tmpAc += str(a) + " "
                tmpAc += "\n"
                log.write(tmpAc)
        print(AccuracyList)
        

if __name__ == "__main__":
    main()
