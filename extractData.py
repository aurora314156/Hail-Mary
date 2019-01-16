import os
import json

type_path = os.path.join(os.getcwd(),'data/dev')
fileName = 'dev_data.json'
with open(fileName, 'w') as CQA:
    data = []
    for each_file in os.listdir(type_path):
        with open(os.path.join(type_path, each_file), 'r') as f:
            content = {}
            question, options, story = [], [], []
            content['storyName'] = each_file
            ans = 0
            for each_line in f:
                parsed = each_line.strip().split()
                if parsed[0] == 'SENTENCE':
                    for p in parsed[1:-1]:
                        story.append(p)
                elif parsed[0] == 'QUESTION':
                    for p in parsed[1:-1]:
                        question.append(p)
                else:
                    options.append(parsed[1:-1])
                    if parsed[-1] == '1':
                        content['answer'] = ans
                    ans += 1
            content['question'] = question
            content['story']= story
            content['options'] = options

        data.append(content)
    print(len(data))
    CQA.write(json.dumps(data, ensure_ascii=False))
    #print(data[tpo_1-conversation_1_1])
        