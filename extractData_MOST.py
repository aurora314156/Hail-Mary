#!/usr/bin/python
 #-*-coding:utf-8 -*-
import os
import json

def removePunctuation(s):
    return 0

cut_punctuation = {"，", "。", "；", "？", "！", "："}
fileName = 'most.json'
with open(fileName, 'w') as CQA:
    data = []
    for r in range(1,7):
        type_path = os.path.join(os.getcwd(),'MOST_original/') + str(r)
        print(type_path)
        count = 0
        for each_file in os.listdir(type_path):
            count += 1
            with open(os.path.join(type_path, each_file), 'r') as f:
                # content = {}
                # content['storyName'] = each_file
                # tmp_words, tmp_list, flag = "", [], 0
                # for each_line in f:
                #     # C
                #     if flag == 1:
                #         for word in each_line:
                #             if word in cut_punctuation:
                #                 tmp_list.append(tmp_words)
                #                 tmp_words = ""    
                #             else:
                #                 tmp_words += word
                #             content['story'] = tmp_list
                #             tmp_list = []
                #     # Q
                #     elif flag == 3:
                #         line_content = removePunctuation(each_line)
                #         content['question'] = tmp_list.append(line_content)
                #         tmp_list = []
                #     # A
                #     elif flag > 5:
                #         options.append(parsed[1:-1])
                #         if parsed[-1] == '1':
                #             content['answer'] = ans
                #             content['options'] = options
                #     flag += 1
                f.close()
        print(count)
    #         data.append(content)
    # CQA.write(json.dumps(data, ensure_ascii=False))
                        
                    
                        
                
                
        
        