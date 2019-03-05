import os, time, json
from numpy import array
from Initial import Initial
from Mymodel import Mymodel
from bert_serving.client import BertClient

def getEncodeContent(bc, single_data):
    
    s_string, q_string, o_string = "", "", ""
    options = []

    for story_element in single_data['story']:
        s_string += story_element + " "
    for question_element in single_data['question']:
        q_string += question_element + " "
    for option in single_data['options']:
        o_string = ""
        for single_option_element in option:
            o_string += single_option_element + " "
        options.append(o_string)
    
    story = bc.encode([s_string])
    question = bc.encode([q_string])
    options = bc.encode(options)
    answer = single_data['answer']

    return story, question, options, answer

def main():
    dataset, d_model = Initial().InitialMain()
    tTime = time.time()
    # initial BERT model
    bc = BertClient()
    correct = 0
    for single_data in dataset:
        story, question, options, answer = getEncodeContent(bc, single_data)
        guessAnswer = Mymodel(story, question, options).MymodelMain()
        if guessAnswer == answer:
            correct += 1
        #print("=========================")
        #print("Guess answer is",guessAnswer)
        #print("Actual answer is",answer)
        #print("=========================\n")
    print("correctness ", correct/;len(dataset))
        
    print("Total cost time %.2fs." % (time.time()-tTime))

if __name__ == "__main__":
    main()
