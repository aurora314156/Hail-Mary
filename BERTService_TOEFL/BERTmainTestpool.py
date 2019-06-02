import os, time, json
from numpy import array
from TFIDF import TFIDF
from Initial import Initial
from Mymodel import Mymodel
from SaveLog import SaveLog
from ContentParser import ContentParser
from bert_serving.client import BertClient


args = get_args_parser().parse_args(['-model_dir', '/project/Divh/cased_L-24_H-1024_A-16/',
                                     '-num_worker', '1',
                                     '-gpu_memory_fraction', '0.9',
                                     '-pooling_layer', '-12',
                                     '-pooling_strategy', 'REDUCE_MAX',
                                     '-port', '6006',
                                     '-port_out', '6007',
                                     '-max_seq_len', 'NONE',])

pooling_strategy = ['REDUCE_MAX', 'REDUCE_MEAN']
def main():
    # initial dataset
    dataset, dataType, model = Initial().InitialMain()
    TF_words, TF_scores = TFIDF(dataset).getTFIDFWeigths()
    AccuracyList = []
    constant, bestAccuracy, bestStrategy, bestPool = 0, 0, "", ""
    for ps in range(2):
        for pool_layer in range(1,13):
            setattr(args, 'pooling_layer', [-pool_layer])
            setattr(args, 'pooling_strategy', [pooling_strategy[ps]])
            server = BertServer(args)
            server.start()
            print('wait until server is ready...')
            time.sleep(50)
            print('encoding...')
            # initial BERT model
            bc = BertClient()
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
                        Process_dataset = "Start processing dataset: " + single_dataset +"\n"
                        print(Process_dataset)
                        continue
                    for single_data in single_dataset:
                        s_string, q_string, options, answer = ContentParser(single_data).getContent()
                        guessAnswer = Mymodel(bc, s_string, q_string, options, m, TF_words, TF_scores, constant).MymodelMain()
                        if guessAnswer == answer:
                            #print(single_data['storyName'])
                            correct += 1
                    accuracy = round(correct/len(single_dataset),3)
                    if accuracy > bestAccuracy:
                        bestAccuracy = accuracy
                        bestPool = pool_layer
                        bestStrategy = pooling_strategy[ps]
                    Accuracy = "Accuracy: " + str(accuracy) + "\n"
                    CostTime = "Total cost time: "+ str(time.time()-tTime) + "\n"
                    AccuracyList.append(accuracy)
                    print(Accuracy)
                    print(CostTime)
                    if typeChange <4:
                        dataTypeLog = "Data type: " + dataType[0] + "\n"
                    else:
                        dataTypeLog = "Data type: " + dataType[1] + "\n"
                    SaveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime).saveLogTxt()
            SaveLog(dataTypeLog, Process_dataset, model, Accuracy, CostTime, AccuracyList).saveLogExcel()
            print("Best Accuracy: ", bestAccuracy)
            print("Best bestPool: ", bestPool)
            print("Best bestStrategy: ", bestStrategy)
            bc.close()
            server.close()
            

if __name__ == "__main__":
    main()
