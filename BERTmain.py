import os, time, json
from numpy import array
from Initial import Initial


def getEncodeContent(single_data):
    
    s_string, q_string, o_string = "", "", ""
    options = []

    for story_element in single_data['story']:
        s_string += story_element + " "
    for question_element in single_data['question']:
        q_string += question_element + " "
    for options in single_data['options']:
        o_string = ""
        for single_option_element in options:
            o_string += single_option_element + " "
        options.append(o_string)
    print(options)
    
    answer = single_data['answer']
    story,question = "", ""
    return story, question, options, answer

def main():
    dataset, d_model = Initial().InitialMain()
    tTime = time.time()
    for single_data in dataset:
        print(single_data['storyName'])
        story, question, options, answer = getEncodeContent(single_data)
        break
        
    print("Total cost time %.2fs." % (time.time()-tTime))

if __name__ == "__main__":
    main()
