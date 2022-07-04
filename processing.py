import re
#from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from fastpunct import FastPunct
import pandas as pd 
#stop_words = stopwords.words('english')
import json
import pickle
import time
from keybert import KeyBERT

t1 = time.time()
f = open(r'D:\Document\Python\text_processing\text_analysis_demo\hedging.txt','r')
hedging = [word.strip('\n') for word in list(f)]


filler_words = ['well', "hmm", "Um", "er", "uh", "like", "actually", "basically", 
                "seriously", "you see", "you know", "I mean", "you know what I mean", 
                "at the end of the day", "believe me", "I guess", "I suppose", 
                "or something", "Okay", "so", "Right", "mhm", "uh", "huh"]

def preprocess(res):
    # preprocess text and change 'i' to 'I'
    text = res['text']
    ps = PorterStemmer()
    # Remove non-alphabetic and non-space characters
    pattern = r"[^a-zA-Z\s]"
    text = re.sub(pattern=pattern, string=text, repl=" ")

    # Replace all spaces into whitespace
    pattern = r"\s"
    text = re.sub(pattern=pattern, string=text, repl=" ")

    # Ensure only one whitespace between every two tokens
    text = " ".join(text.split())

    # Lowercase all characters
    text = text.lower()

    # Tokenise
    text = text.split()

    #
    text = [word.upper() if word == 'i' else word for word in text]
    # Remove stopwords 
    #text = list(filter(lambda x: x not in stop_words, text))


    # Remove single-character tokens
    #text = list(filter(lambda x: len(x) > 1, text))

    #with stemming
    text2 = [ps.stem(w) for w in text]
    res['text']  =  ' '.join(text)
    # for i in range(len(text)): 
    #     res['result'][i]['word'] = text[i]
    return res

def check_duplication_in_list(word_list): 
    # find duplication in a list and return a dictionary of the duplicated words and 
    # their positions
    duplicate_list = {}
    n = len(word_list)
    for word in set(word_list):
        if word_list.count(word)>1 : 
            duplicate_list[word] = []
    
    for i in range(n): 
        if word_list[i] in duplicate_list.keys() : 
            duplicate_list[word_list[i]].append(i)
    
    return duplicate_list

def find_filler_and_hedging(res): 
    # find filler and hedging word in text 
    res['filler'] = {}
    res['hedging'] = {}
    for i in range(len(res['result'])): 
        if res['result'][i]['word'] in filler_words: 
            if res['result'][i]['word'] in res['filler'].keys():
                res['filler'][res['result'][i]['word']].append(i)
            else: 
                res['filler'][res['result'][i]['word']] = [i]
        if res['result'][i]['word'] in hedging: 
            if res['result'][i]['word'] in res['hedging'].keys():
                res['hedging'][res['result'][i]['word']].append(i)
            else: 
                res['hedging'][res['result'][i]['word']] = [i]
    return res 

def repetition (res): 
    # find word that repeted in the text 
    # conditions: 
    # 2 dupicate words sit next to each other
    # bigrams and trigrams appear in the whole speech more than once. 
    word_list = res['text'].split(' ')
    repeted_word = {}
    # find repeted word next to each other 
    for i in range(len(word_list)-1):
        if word_list[i] == word_list[i+1]:
            if word_list[i] in repeted_word.keys():
                repeted_word[word_list[i]].append(i)
            else:
                repeted_word[word_list[i]]= [i]

    # bigram and trigram 
    _2gram = [' '.join(e) for e in ngrams(word_list, 2)]
    _3gram = [' '.join(e) for e in ngrams(word_list, 3)]

    _2gram_duplicate = check_duplication_in_list(_2gram)
    _3gram_duplicate = check_duplication_in_list(_3gram)

    repeted_word.update(_2gram_duplicate)
    repeted_word.update(_3gram_duplicate)

    res['repetition'] = repeted_word

    return res

def display(res): 
    # print dictionary in good format  
    json_obj = json.dumps(res,indent=4)
    print(json_obj)
    return 

def speaking_duration(res): 
    # find speaking and stopping time 
    # add a new key for each word in res['result'] as 'stoptime' 

    stoptime = []
    result = res['result']
    speaking_time = -res['result'][-1]['start'] + res['result'][-1]['end']
    for i in range(len(result)-1): 
        res['result'][i]['stoptime'] = res['result'][i+1]['start'] - res['result'][i]['end']
        stoptime.append(res['result'][i+1]['start'] - res['result'][i]['end'])
        speaking_time+= res['result'][i]['end'] -res['result'][i]['start']

    res['stop_time'] = sum(stoptime)
    res['speaking_time'] = speaking_time
    
    return res

def articulation(res):
    # find articulation 
    clear = 0 
    unclear = 0 
    for word in res['result'] : 
        if word['conf'] < 1.0 : 
            unclear += 1 
        else: 
            clear += 1 
    res['articulation'] = clear/len(res['result'])
    
    return res 

def word_speed(res):
    for i in range(len(res['result'])): 
        res['result'][i]['speak_time'] = res['result'][i]['end']-res['result'][i]['start']
    
    res['average_speaking_time'] = res['speaking_time']/len(res['result'])
    res['average_stop_time'] = res['stop_time'] / (len(res['result'])-1) 
    return res 

def spliting_sentences(res): 
    #splitting senteces by simple algorithm relate to the stop time of the speaker 
    sentence = ''
    sentences = {}
    sen_num = 1
    sen_length = 0 
    for i in range(len(res['result'])-1): 
        sentence += res['result'][i]['word']+ ' '
        sen_length+= 1
        if res['result'][i]['stoptime']> res['average_stop_time']*6 and sen_length>=3:
            
            sentences[sen_num]={'text':sentence, 'end' : res['result'][i]['end']}
            sentence = ' '
            sen_length = 0 
            sen_num += 1
    sentence += res['result'][len(res['result'])-1]['word']
    
    sentences[sen_num]={'text': sentence , 'end' : res['result'][i]['end']}
    
    res['sentences'] = sentences; 
    return res 

def create_data_frame(res): 
    wordlen=[]
    word = []
    period = []
    speak_time = []
    stop_time = []
    for i in range(len(res['result'])-1): 
        word.append(res['result'][i]['word'])
        wordlen.append(len(res['result'][i]['word']))
        period.append(0)
        speak_time.append(res['result'][i]['speak_time'])
        stop_time.append(res['result'][i]['stoptime'])
    
    df = pd.DataFrame({
            'word': word ,
            'wordlen': wordlen, 
            'speak_time' : speak_time,
            'stop_time' : stop_time,
            'period': period 
    })
     
    return df


def split_senteces_with_model(res,model,df):
    x_col = ['wordlen','speak_time','stop_time']

    x_test = df[x_col].to_numpy() 

    position = model.predict(x_test)

    #splitting senteces by simple algorithm relate to the stop time of the speaker 
    sentence = ''
    sentences = {}
    sen_num = 1
    sen_length = 0 
    for i in range(len(res['result'])-1): 
        sentence += res['result'][i]['word']+ ' '
        sen_length+= 1
        if position[i]:
            
            sentences[sen_num]={'text':sentence, 'end' : res['result'][i]['end']}
            sentence = ' '
            sen_length = 0 
            sen_num += 1
    sentence += res['result'][len(res['result'])-1]['word']
    
    sentences[sen_num]={'text': sentence , 'end' : res['result'][i]['end']}
    
    res['sentences'] = sentences; 
    return res 
    


def beautify(res): 
    sentences = []
    for i in range(len(res['sentences'])):
        sentences.append(res['sentences'][i+1]['text'])
    fastpunct = FastPunct()
    x = fastpunct.punct(sentences,correct=True)
    text = ''
    for i in range(len(res['sentences'])):
        res['sentences'][i+1]['text'] = x[i]
        text +=' '+ x[i]
    res['text'] = text
    # x =fastpunct.punct([
    #               'well i have equal rights for all except blacks asians hispanics jews gays women muslims',
    #               'verybody is not a white man and i mean white white',
    #               'know italians know polish just people from ireland england and scotland but only certain parts of scotland and ireland',
    #               'just full blooded whites',
    #                'not you know what not even white nobody gets any right'], correct=True)

    #print(x)
    return res

def split_paragraph(res): 
    paragraph = ''
    paragraphs = []
    time = 30 
    i = 1
    print(res['stop_time']+res['speaking_time'])
    # one paragraph will not longer than 30 sec
    if time > res['stop_time']+res['speaking_time']:
        res ['paragraphs'] = [res['text']]
        return res

    while time < res['stop_time']+res['speaking_time']: 
        while res['sentences'][i]['end'] < time : 
            paragraph += res['sentences'][i]['text']
            i += 1 
        
        paragraphs.append(paragraph)
        paragraph = ''
        time += 30 
    n = len(res['sentences'])+1
    for k in range(i,n):
        paragraph += res['sentences'][k]['text'] 
    paragraphs.append(paragraph)
    res['paragraphs'] = paragraphs
    return res 

def get_key_word(res): 
    kw_model = KeyBERT()
    doc = res['text']
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None)
    res['keys_word'] = keywords
    return res

def get_data(res): 
    wordlen=[]
    word = []
    period = []
    speak_time = []
    stop_time = []
    for i in range(len(res['result'])-1): 
        word.append(res['result'][i]['word'])
        wordlen.append(len(res['result'][i]['word']))
        period.append(0)
        speak_time.append(res['result'][i]['speak_time'])
        stop_time.append(res['result'][i]['stoptime'])
    
    df = pd.DataFrame({
            'word': word ,
            'wordlen': wordlen, 
            'speak_time' : speak_time,
            'stop_time' : stop_time,
            'period': period 
    })
    df.to_csv('data3.csv',index= False) 
    return 

def text_anasyslis(test,model): 
    # adding analysis to the input dictionary 
    test = preprocess(test)
    test = speaking_duration(test)
    test = find_filler_and_hedging(test)
    test = articulation(test)
    test= repetition(test)  
    test = word_speed(test) 
    get_data(test)  
    df = create_data_frame(test)
    test = split_senteces_with_model(test,model,df)
    test = beautify(test)
    test = split_paragraph(test)
    test = get_key_word(test)
    
    return test 





with open('model_pickle','rb') as f : 
    model = pickle.load(f)



with open('sample1.json') as json_file:
    test = json.load(json_file)
test = text_anasyslis(test,model)
display(test)
t2 = time.time()

print('total time: ',t2-t1)





#word_list = 'We re no strangers to love You know the rules and so do I A full commitments what Im thinking of You wouldnt get this from any other guy I just wanna tell you how Im feeling Gotta make you understand Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you Weve known each other for so long Your hearts been aching but youre too shy to say it Inside we both know whats been going on We know the game and were gonna play it And if you ask me how Im feeling Dont tell me youre too blind to see Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you Never gonna give, never gonna give (Give you up) (Ooh) Never gonna give, never gonna give (Give you up) Weve known each other for so long Your hearts been aching but '.strip().split(' ')
