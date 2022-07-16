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
from nlprule import Tokenizer, Rules
from transformers import pipeline



t1 = time.time()
f = open(r'hedging.txt','r')
hedging = [word.strip('\n') for word in list(f)]


filler_words = ['well', "hmm", "Um", "er", "uh", "like", "actually", "basically",
                "seriously", "you see", "you know", "I mean", "you know what I mean",
                "at the end of the day", "believe me", "I guess", "I suppose",
                "or something", "Okay", "so", "Right", "mhm", "uh", "huh"]

def preprocess(res):
    #change i to I
    pattern = r'( i((?=\s)|(?=\')))'
    text = re.sub(pattern,' I',res['text'])
    res['text'] = text
    words = text.split(' ')
    for a,b in zip(words,res['result']):
        b['word'] = a
    return res

def check_duplication_in_list(word_list,length):
    # find duplication in a list and return a dictionary of the duplicated words and
    # their positions
    repeted_word = {}
    for i in range(len(word_list)-length):
        if word_list[i] == word_list[i+length]:
            if word_list[i] in repeted_word.keys():
                repeted_word[word_list[i]].append(i)
            else:
                repeted_word[word_list[i]]= [i]

    return repeted_word

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

    _2gram_duplicate = check_duplication_in_list(_2gram,2)
    _3gram_duplicate = check_duplication_in_list(_3gram,3)

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

    res['words_per_minutes'] = len(res['result'])/( res['speaking_time']+res['stop_time'])*60
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

    res['sentences'] = sentences
    return res

def create_data_frame(res):
    # create data frame as a input for the model
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


def split_senteces_with_model(res,df):
    with open('model_pickle','rb') as f :
        model = pickle.load(f)

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

    res['sentences'] = sentences
    return res


def adding_comma_and_split_senteces_with_model(res,df):
    # open saved model to detect comma and period in a paragraph.
    with open('model_predict_comma_and_period','rb') as f :
        model = pickle.load(f)

    x_col = ['wordlen','speak_time','stop_time']
    if res['words_per_minutes']<150 : 
        df['stop_time'] = df['stop_time']*res['words_per_minutes']/150
    x_test = df[x_col].to_numpy()

    position = model.predict(x_test)

    #splitting senteces by simple algorithm relate to the stop time of the speaker
    sentence = ''
    sentences = {}
    sen_num = 1
    sen_length = 0
    res['text'] = ''
    for i in range(len(res['result'])-1):
        if sen_length == 0 :
            res['result'][i]['word'] = res['result'][i]['word'][0].upper() + res['result'][i]['word'][1:]
        sen_length+= 1
        if position[i]==2.0:
            sentence += res['result'][i]['word']+ '. '
            sentences[sen_num]={'text':sentence, 'end' : res['result'][i]['end']}
            res['text'] += sentence
            sentence = ''
            sen_length = 0
            sen_num += 1
        elif position[i] == 1.0 :
            sentence += res['result'][i]['word']+ ', '
        else :
            sentence += res['result'][i]['word']+ ' '
    sentence += res['result'][len(res['result'])-1]['word']+ '. '
    res['text'] += sentence
    sentences[sen_num]={'text': sentence , 'end' : res['result'][i]['end']}

    res['sentences'] = sentences
    return res

def question_mark_check(res): 
    question_word = ['what','when','where','who','how','why']
    question_word_begin = ['Would','Will','Can','Could','Should']
    for i in range(len(res['sentences'])): 
        sentence = res['sentences'][i+1]['text']
        for word in question_word : 
            if word in sentence.lower() : 
                sentence = sentence.replace('.','?')
                break 
        for word in question_word_begin : 
            if word in sentence: 
                sentence = sentence.replace('.','?')
                break 
        res['sentences'][i+1]['text'] = sentence 
    return res 



def grammar_check(res):
    #check for grammar error and suggest the solution
    tokenizer = Tokenizer.load("en")
    rules = Rules.load("en", tokenizer)
    
    for i in range(len(res['sentences'])) :
        res['sentences'][i+1]['suggestion'] = []
        sentence = res['sentences'][i+1]['text'] 
        suggests = rules.suggest(sentence)
        for s in suggests:
            suggestion = {
                'start' : s.start,
                'end'   : s.end ,
                'replacements' : s.replacements,
                'source' : s.source ,
                'message': s.message
            }
            res['sentences'][i+1]['suggestion'].append(suggestion)
    
    return res

def merge_text(res): 
    res['text'] = ''
    for sen_num in res['sentences'].keys(): 
        res['text'] += res['sentences'][sen_num]['text']
    return res 

def beautify(res):
    # no longer use because take too long to respond
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

def get_emotion(res): 
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
    prediction = classifier(res['text'])
    emotions = {}
    for emotion in prediction: 
        emotions[emotion['label']] = emotion['score']
    res['emotions'] = emotions 
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
    # get key words from text
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

def text_analysis(test):
    # adding analysis to the input dictionary
    test = preprocess(test)
    test = speaking_duration(test)
    test = find_filler_and_hedging(test)
    test = articulation(test)
    test= repetition(test)
    test = word_speed(test)
    df = create_data_frame(test)
    test = split_senteces_with_model(test,df)
    test = beautify(test)
    test = split_paragraph(test)
    test = get_key_word(test)

    return test


def text_analysis_2 (test):
    # adding analysis to the input dictionary
    test = preprocess(test)
    test = speaking_duration(test)
    test = find_filler_and_hedging(test)
    test = articulation(test)
    test=  repetition(test)
    test = word_speed(test)
    df = create_data_frame(test)
    test = adding_comma_and_split_senteces_with_model(test,df)
    test = question_mark_check(test)
    test = merge_text(test)
    test = get_emotion(test)
    test = split_paragraph(test)
    test = grammar_check(test)
    test = get_key_word(test)

    return test



with open('model_predict_comma_and_period','rb') as f :
    model = pickle.load(f)



with open('data/sample8.json') as json_file:
    test = json.load(json_file)


test = text_analysis_2(test)
#display(test)
with open('output_sample4.json','w') as f :
    f.write(json.dumps(test,indent=4))
t2 = time.time()
print('total time: ',t2-t1)





#word_list = 'We re no strangers to love You know the rules and so do I A full commitments what Im thinking of You wouldnt get this from any other guy I just wanna tell you how Im feeling Gotta make you understand Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you Weve known each other for so long Your hearts been aching but youre too shy to say it Inside we both know whats been going on We know the game and were gonna play it And if you ask me how Im feeling Dont tell me youre too blind to see Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you Never gonna give, never gonna give (Give you up) (Ooh) Never gonna give, never gonna give (Give you up) Weve known each other for so long Your hearts been aching but '.strip().split(' ')
