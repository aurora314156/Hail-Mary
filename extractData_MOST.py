#!/usr/bin/python
 #-*-coding:utf-8 -*-
import os
import json
import re

def removePunctuation(s):
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+「」——！，。？、~@#￥%……&*（）*()?]+", "", s)
    return string

fileLen = [0, 1296, 1254, 1500, 1500, 1500, 1500]
cut_punctuation = {"，", "。", "；", "？", "！", "："}
fileName = 'most.json'
with open(fileName, 'w') as CQA:
    data = []
    flag, count = 0, 0
    for r in range(1,7):
        type_path = os.path.join(os.getcwd(),'MOST_original/') + str(r)
        with open(str(r)+'.json', 'r') as a:
            correct_ans = json.loads(a.read())
            for fileLenInd in range(fileLen[r]):
                fileName = str(fileLenInd+1) +".txt"
                with open(os.path.join(type_path, fileName), 'r') as f:
                    content = {}
                    content['storyName'] = str(count)
                    tmp_words, tmp_list, flag = "", [], 0
                    for each_line in f:
                        each_line = each_line.strip()
                        # C
                        if flag == 1:
                            for word in each_line:
                                if word in cut_punctuation and tmp_words is not "":
                                    tmp_words = removePunctuation(tmp_words.replace("\n", ""))
                                    #tmp_list.append(tmp_words.replace("\n", ""))
                                    if tmp_words is "":
                                        continue
                                    else:
                                        tmp_list.append(tmp_words)
                                    tmp_words = ""
                                else:
                                    tmp_words += word
                            if tmp_words is not "" and tmp_words is not "\n":
                                tmp_list.append(tmp_words.replace("\n", ""))
                            content['story'] = tmp_list
                            tmp_list = []
                        # Q
                        elif flag == 3:
                            line_content = removePunctuation(each_line)
                            tmp_list.append(line_content.replace("\n", ""))
                            content['question'] = tmp_list
                            tmp_list = []
                        # A
                        elif flag > 4:
                            tmp_list.append(each_line.replace("\n", ""))
                        flag += 1
                    
                    content['answer'] = correct_ans[fileLenInd]
                    content['options'] = tmp_list
                    data.append(content)
                    f.close()
                    count += 1
            a.close()
        print(count)
    CQA.write(json.dumps(data, ensure_ascii=False))
                        
                    
                        
                
                
        
        