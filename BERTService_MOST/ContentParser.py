


class ContentParser():
    def __init__(self, single_data):
        self.single_data = single_data
        
    def getContent(self):
    
        s_string, q_string, o_string, options = "", "", "", []

        for story_element in self.single_data['story']:
            s_string += story_element + " "
        for question_element in self.single_data['question']:
            q_string += question_element
        for option in self.single_data['options']:
            o_string = ""
            for single_option_element in option:
                o_string += single_option_element
            options.append(o_string)
            
        answer = self.single_data['answer']
        return s_string[:-1], q_string[:], options, answer
