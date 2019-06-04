#!/usr/bin/python
 #-*-coding:utf-8 -*-
import os
import json
import re

# def removePunctuation(s):
#     string = re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+「」『』….:——！，。？、~@#￥%……&*（）*()?=-]+", "", s)
#     return string

def removePunctuation(char):
    char = re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+「」『』《》….:——！，。？、~@#￥%……&*（）*()?=-]+", "", char)
    appendWord = ""
    # reserve zh
    #if '\u4e00' <= char <= '\u9fff' or '\u2e80'<= char <= '\u2fdf' or '\u3400'<= char <= '\u4dbf':
    for cc in char:
        if cc >= u'\u4E00' and cc <= u'\u9FA5':
            appendWord += cc
        # reserve number
        elif cc >= u'\u0030' and cc <= u'\u0039':
            appendWord += cc
        elif ( cc >= u'\u0041' and cc <= u'\u005A' ) or ( cc >= u'\u0061' and cc <= u'\u007A'):
            appendWord += cc
    
    return appendWord

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
            #for fileLenInd in range(10,11):
                fileName = str(fileLenInd+1) + ".txt"
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
                                        #print(tmp_list)
                                    tmp_words = ""
                                else:
                                    tmp_words += word
                            #print(tmp_words)
                            if tmp_words is not "" and tmp_words is not "\n":
                                tmp_list.append(removePunctuation(tmp_words.replace("\n", "")))
                            content['story'] = tmp_list
                            tmp_list = []
                        # Q
                        elif flag == 3:
                            #print(each_line)
                            line_content = removePunctuation(each_line)
                            #print(line_content)
                            tmp_list.append(line_content.replace("\n", ""))
                            #print(tmp_list)
                            content['question'] = tmp_list
                            #print(content['question'])                            
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
                        
                    
                        
                
                
        
        