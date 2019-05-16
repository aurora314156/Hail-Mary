#!/usr/bin/python
 #-*-coding:utf-8 -*-
import os
import json
import re

def removePunctuation(s):
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+「」——！，。？、~@#￥%……&*（）*()?]+", "", s)
    return string

cut_punctuation = {"，", "。", "；", "？", "！", "："}
fileName = 'most.json'
with open(fileName, 'w') as CQA:
    data = []
    for r in range(1,7):
        type_path = os.path.join(os.getcwd(),'MOST_original/') + str(r)
        with open(str(r)+'.json', 'r') as a:
            aa = json.loads(a.read())
            count = 0
            for each_file in os.listdir(type_path):
                count += 1
                with open(os.path.join(type_path, each_file), 'r') as f:
                    content = {}
                    content['storyName'] = str(r) + '_' + each_file
                    tmp_words, tmp_list, flag = "", [], 0
                    for each_line in f:
                        # C
                        if flag == 1:
                            for word in each_line:
                                if word in cut_punctuation and tmp_words is not "":
                                    tmp_list.append(removePunctuation(tmp_words))
                                    tmp_words = ""
                                else:
                                    tmp_words += word
                            content['story'] = tmp_list
                            tmp_list = []
                        # Q
                        elif flag == 3:
                            line_content = removePunctuation(each_line)
                            tmp_list.append(line_content)
                            content['question'] = tmp_list
                            tmp_list = []
                        # A
                        elif flag > 4:
                            tmp_list.append(each_line.replace("\n", ""))
                        flag += 1
                    
                    content['answer'] = aa[int(each_file[:len(each_file)-4])-1]
                    content['options'] = tmp_list
                    data.append(content)
                    f.close()
            a.close()
        print(count)
    CQA.write(json.dumps(data, ensure_ascii=False))
                        
                    
                        
                
                
        
        