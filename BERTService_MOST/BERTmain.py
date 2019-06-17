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

def randomNum(corpus_amount, flag):
    flag = []
    while(len(flag) < corpus_amount):
        randomNumber = randint(1, corpus_amount)
        if randomNumber not in flag:
            flag.append(randomNumber)
    return flag

fraction = 1
# lim_start = [6840, 0, 1710, 3420, 5130]
# lim_end = [8549, 1709, 3419, 5129, 6839]
lim_start = [7694, 0, 855, 1710, 2565, 3420, 4275, 5130, 5985, 6840]
lim_end = [8549, 854, 1709, 2564, 3419, 4274, 5129, 5984, 6839, 7693]


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

args_setting_max = ['-model_dir', '/project/Divh/chinese_L-12_H-768_A-12/',
                    '-graph_tmp_dir', '/project/Divh/tmp/',
                                     '-num_worker', '1',
                                     '-gpu_memory_fraction', '0.3',
                                     '-pooling_layer', '-12',
                                     '-pooling_strategy', 'REDUCE_MAX',
                                     '-port', '5557',
                                     '-port_out', '5558',
                                     '-max_seq_len', 'NONE',]
args_setting_mean = ['-model_dir', '/project/Divh/chinese_L-12_H-768_A-12/',
                     '-graph_tmp_dir', '/project/Divh/tmp/',
                                     '-num_worker', '1',
                                     '-gpu_memory_fraction', '0.3',
                                     '-pooling_layer', '-12',
                                     '-pooling_strategy', 'REDUCE_MEAN',
                                     '-port', '5557',
                                     '-port_out', '5558',
                                     '-max_seq_len', 'NONE',]


tmpargs = []
tmpargs.append(args_setting_max)
tmpargs.append(args_setting_mean)

port, port_out = 5557, 5558

def main():
    # initial dataset
    dataset, dataType, model = Initial().InitialMain()
    #TF_words, TF_scores = TFIDF(dataset).getTFIDFWeigths()
    TF_words, TF_scores = "", ""
    constant, bestAccuracy, bestStrategy, bestPool = 0, 0, "", ""
    with open('accuracyList.txt', 'w') as log:
        log.close()
    constant = 0
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
                time.sleep(20)
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
                        correct, count, wrongNum = 0, 0, {}
                        for single_data in single_dataset:
                            # storyName = int(single_data['storyName'].split(".")[0][2:])
                            # print(storyName)
                            if count >= lim_start[l] and count <= lim_end[l]:
                                #storyName = int(single_data['storyName'].split(".")[0][2:])
                                s_string, q_string, options, answer = ContentParser(single_data).getContent()
                                #print(single_data['storyName'])
                                guessAnswer = Mymodel(bc, s_string, q_string, options, m, TF_words, TF_scores, constant).MymodelMain()
                                if guessAnswer == answer:
                                    correct += 1
                                else:
                                    wrongNum[count] = guessAnswer
                                tmpC +=1
                            count += 1
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
    #SaveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime, AccuracyList).saveLogExcel()
            
if __name__ == "__main__":
    main()