from scipy import spatial
import numpy as np

class Mymodel():
    def __init__(self, model, token_dict, s_string, q_string, options, m):
        self.model = model
        self.token_dict = token_dict
        self.model_id = m
        self.s_string = s_string
        self.q_string = q_string
        self.options = options
    
    def MymodelMain(self):
        guessAnswer = ""
        #guessAnswer = self.FirstModel()
        if self.model_id == "ForthModel":
            guessAnswer = self.ForthModel()
        if self.model_id == "FirstModel":
            guessAnswer = self.FirstModel()
        if self.model_id == "SecondModel":
            guessAnswer = self.SecondModel()
        
        return guessAnswer
    
    def FirstModel(self):
        """
        merge story and question vector by add, calculate similarity with merge story and option vector
        """
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        options = bc.encode(self.options)
        merStoryQue = [x + y for x, y in zip(story, question)]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            merStoryOpt = [x + y for x, y in zip(story, option)]
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, merStoryOpt)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def SecondModel(self):
        """
        merge story and question vector by dot, calculate similarity with merge story and option vector
        """
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        options = bc.encode(self.options)
        merStoryQue = [x * y for x, y in zip(story, question)]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            merStoryOpt = [x * y for x, y in zip(story, option)]
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, merStoryOpt)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def SecondModelWithSoftmax(self):
        """
        merge story and question vector by dot, calculate similarity with merge story and option vector
        """
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        options = bc.encode(self.options)
        merStoryQue = [x * y for x, y in zip(story, question)]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            merStoryOpt = [x * y for x, y in zip(story, option)]
            tmpScore = 1 - spatial.distance.cosine(self.softmax(merStoryQue), self.softmax(merStoryOpt))
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def ThirdModel(self):
        """
        implementation original paper method
        """
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        options = bc.encode(self.options)
        
        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        merStoryQue = [x + y for x, y in zip(story, question)]
        for i in range(100):
            tmp = [x + y for x, y in zip(story, merStoryQue)]
            merStoryQue = tmp

        for option in self.options:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def ForthModel(self):
        """
        encode story sentences, then use each story sentences vector to calculate similarity with question
        choose highest score story vector to calculate similarity with options
        """
        print("run forthModel")
        sentences = self.getParserResStrToList()
        sentences, question, options = self.preprocess(sentences, self.q_string, self.options)

        story_sentences, options = [], []
        for s in sentences:
            story_sentences.append(self.getSentenceEmbedWithPool(s))
        for o in options:
            options.append(self.getSentenceEmbedWithPool(o))

        print(story_sentences.shape)
        print(options.shape)

        ind, guessAnswer, highestScore, highestScore_storyVector = 0, 0, 0, []
        for s in story_sentences:
            tmpScore = 1 - spatial.distance.cosine(s, question)
            if tmpScore > highestScore:
                highestScore_storyVector = s
                highestScore = tmpScore

        highestScore = 0
        for option in options:
            tmpScore = 1 - spatial.distance.cosine(option, highestScore_storyVector)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1

        return guessAnswer
    
    
    def getSentenceEmbedWithPool(self, tokens):
        print("get sentence embed with pooling")
        tmp_token = []
        for token in tokens:
            if token in token_dict:
                tmp_token.append(self.token_dict[token])
            
        tokens = tmp_token
            
        token_input = np.asarray([[self.token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
        seg_input = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])

        print('Inputs:', token_input[0][:len(tokens)])
        predicts = self.model.predict([token_input, seg_input])[0]
        print('Pooled:', predicts.tolist()[:5])
        
        return predicts

    def getParserResStrToList(self):
        print("Start parser string to list")
        sentences, tmp_string = [], ""
        for s in self.s_string:
            if s == "." or s == "?":
                sentences.append(tmp_string)
                tmp_string = ""
                continue
            tmp_string += s

        # use whole story structure
        if tmp_string != "":
            sentences.append(sentences[:len(tmp_string)-1])
        
        return sentences
    
    def preprocess(self, sentences, question, options):
        """
        due to original code limit
        """
        print("start preprocess data vector")
        tmp_question = []
        tmp_question.append("[CLS]")
        for q in question.split(" "):
            if q != "":
                tmp_question.append(q)
        tmp_question.append("[SEP]")

        tmp_sentences = []
        for sentence in sentences:
            if len(sentence) ==0:
                continue
            t_sentence = []
            t_sentence.append("[CLS]")
            for s in sentence.split(" "):
                if s != "":
                    t_sentence.append(s)
            t_sentence.append("[SEP]")
            tmp_sentences.append(t_sentence)
        
        tmp_options = []
        for option in options:
            if len(option) ==0:
                continue
            t_option = []
            t_option.append("[CLS]")
            for o in option.split(" "):
                if o != "":
                    t_option.append(o)
            t_option.append("[SEP]")
            tmp_options.append(t_option)

        return tmp_sentences, tmp_question, tmp_options

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)