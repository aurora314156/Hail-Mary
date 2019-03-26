

class ContentParser():
    
    def getContent(self, single_data):
    
        s_string, q_string, o_string, options = "", "", "", []

        for story_element in single_data['story']:
            s_string += story_element + " "
        for question_element in single_data['question']:
            q_string += question_element + " "
        for option in single_data['options']:
            o_string = ""
            for single_option_element in option:
                o_string += single_option_element + " "
            options.append(o_string[:-1])
            
        answer = single_data['answer']
        return s_string[:-1], q_string[:-1], options, answer