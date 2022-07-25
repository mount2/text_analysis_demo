import re
from nltk.util import ngrams
import pandas as pd
import json
import time
from keybert import KeyBERT
from nlprule import Tokenizer, Rules
from transformers import pipeline
import logging
from simpletransformers.ner import NERModel
from word2number import w2n
import re 
import time 

def categorize (num_word): 
    unit = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine"]
    two_digit=[ "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"]
    scales = ["hundred", "thousand", "million", "billion", "trillion"]
    tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    unit_word = 0 
    number_scale = 0 
    
    bow = num_word.split(' ')
    if 'double' in bow or 'o' in bow : 
        return 'year' 
    if 'point' in bow or 'and' in bow: 
        return 'num'
    for i in range(len(bow)) :
        word = bow[i]
        if word in scales:
            number_scale += 1 
    
    if number_scale ==  0 : 
        return 'year'
    
    return 'num'

def year_to_num(word): 
    num = word.split(' ')
    unit_dict = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'zero' : 0,
    'o' : 0 , }
    two_digit_dict={
    "ten":10, 
    "eleven":11, 
    "twelve":12,
    "thirteen":13, 
    "fourteen":14, 
    "fifteen":15,
    "sixteen":16, 
    "seventeen":17, 
    "eighteen":18, 
    "nineteen":19
}
    tens_dict= {"twenty":20, "thirty":30, "forty":40, "fifty":50, "sixty":60, "seventy":70, "eighty":80, "ninety":90}
    number = ''
    i = 0 

    while i < len(num): 
        word = num[i]
        #print('word:',word)

        if word in tens_dict : 
            if i < len(num)-1:
                if num[i+1] in unit_dict.keys(): 
                    number+= str(tens_dict[word]+unit_dict[num[i+1]])
                    i+= 1 
                else : 
                    number += str(tens_dict[word])
            else: 
                number += str(tens_dict[word])
        elif word in two_digit_dict: 
            number+= str(two_digit_dict[word])
        elif word in unit_dict:
            number+= str(unit_dict[word])
        elif word == 'double': 
            if num[i+1] in unit_dict.keys(): 
                i+= 1 
                word = num[i]
                number+=  str(unit_dict[word])*2
            else: 
                pass 
        #print('word:',word)
        #print('number:',number)
        i += 1

    return str(number) +' '

def words2number(text): 
    words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",'double',
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",'dot','point','o',
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", 
        "sixty", "seventy", "eighty", "ninety","hundred", "thousand", "million", "billion", "trillion" ]
    num_words = {}
    bow = text.split(' ')
    and_in = False
    i = 0 
    num_word = ''
    new_text = ''
    while i < len(bow):
        word = bow[i] 
        
        if num_word != '': 
            if word == 'and' and and_in == False: 
                num_word += word +' '
                i+= 1
                and_in = True 
                word = bow[i] 
        #detect word that in the number word list -> convert them to correct form 
        if word in words : 
            print(word)
            if i < len(bow)-1:
                if word == 'point': 
                    if num_word == '' or bow[i+1] not in words : 
                        new_text += word +' '
                    else: 
                        num_word += word +' '
                elif (bow[i+1] not in words  and (bow[i+1] != 'and'or and_in )):
                    and_in = False
                    num_word+= word
                    if categorize(num_word) == 'num':
                        print(2,num_word,w2n.word_to_num(num_word))    
                        new_text += str(w2n.word_to_num(num_word))+' '
                    else : 
                        new_text += year_to_num(num_word)+' '
                    num_word = ''
                elif num_word == '':
                    num_word = word +' '
                else : 
                    num_word += word +' '
            else : 
                if word == 'point': 
                    new_text+= word +' '
                else: 
                    num_word+= word
                    if categorize(num_word) == 'num':
                        new_text += str(w2n.word_to_num(num_word))+' '
                        
                    else : 
                        text = text.replace(num_word,year_to_num(num_word),1) 
                        new_text+= year_to_num(num_word)+' '
                    num_word = ''
        else: 
            new_text += word +' '
        i += 1
    
    # replace miltiple space with 1 space 
    new_text = re.sub("\s+", " ",new_text) 
    return new_text 

class RestorePuncts:
    def __init__(self, wrds_per_pred=250):
        self.wrds_per_pred = wrds_per_pred
        self.overlap_wrds = 30
        self.valid_labels = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']
        self.model = NERModel("bert", "felflare/bert-restore-punctuation",use_cuda=False, labels=self.valid_labels,args={"silent": True, "max_seq_length": 512})

    def punctuate(self, text: str):
        """
        Performs punctuation restoration on arbitrarily large text.
        Detects if input is not English, if non-English was detected terminates predictions.
        Overrride by supplying `lang='en'`
        
        Args:
            - text (str): Text to punctuate, can be few words to as large as you want.
        """


        # plit up large text into bert digestable chunks
        splits = self.split_on_toks(text, self.wrds_per_pred, self.overlap_wrds)
        # predict slices
        # full_preds_lst contains tuple of labels and logits
        full_preds_lst = [self.predict(i['text']) for i in splits]
        # extract predictions, and discard logits
        preds_lst = [i[0][0] for i in full_preds_lst]
        # join text slices
        combined_preds = self.combine_results(text, preds_lst)
        # create punctuated prediction
        punct_text = self.punctuate_texts(combined_preds)
        return punct_text

    def predict(self, input_slice):
        """
        Passes the unpunctuated text to the model for punctuation.
        """
        predictions, raw_outputs = self.model.predict([input_slice])
        return predictions, raw_outputs

    @staticmethod
    def split_on_toks(text, length, overlap):
        """
        Splits text into predefined slices of overlapping text with indexes (offsets)
        that tie-back to original text.
        This is done to bypass 512 token limit on transformer models by sequentially
        feeding chunks of < 512 toks.
        Example output:
        [{...}, {"text": "...", 'start_idx': 31354, 'end_idx': 32648}, {...}]
        """
        wrds = text.replace('\n', ' ').split(" ")
        resp = []
        lst_chunk_idx = 0
        i = 0

        while True:
            # words in the chunk and the overlapping portion
            wrds_len = wrds[(length * i):(length * (i + 1))]
            wrds_ovlp = wrds[(length * (i + 1)):((length * (i + 1)) + overlap)]
            wrds_split = wrds_len + wrds_ovlp

            # Break loop if no more words
            if not wrds_split:
                break

            wrds_str = " ".join(wrds_split)
            nxt_chunk_start_idx = len(" ".join(wrds_len))
            lst_char_idx = len(" ".join(wrds_split))

            resp_obj = {
                "text": wrds_str,
                "start_idx": lst_chunk_idx,
                "end_idx": lst_char_idx + lst_chunk_idx,
            }

            resp.append(resp_obj)
            lst_chunk_idx += nxt_chunk_start_idx + 1
            i += 1
        logging.info(f"Sliced transcript into {len(resp)} slices.")
        return resp

    @staticmethod
    def combine_results(full_text: str, text_slices):
        """
        Given a full text and predictions of each slice combines predictions into a single text again.
        Performs validataion wether text was combined correctly
        """
        split_full_text = full_text.replace('\n', ' ').split(" ")
        split_full_text = [i for i in split_full_text if i]
        split_full_text_len = len(split_full_text)
        output_text = []
        index = 0

        if len(text_slices[-1]) <= 3 and len(text_slices) > 1:
            text_slices = text_slices[:-1]

        for _slice in text_slices:
            slice_wrds = len(_slice)
            for ix, wrd in enumerate(_slice):
                # print(index, "|", str(list(wrd.keys())[0]), "|", split_full_text[index])
                if index == split_full_text_len:
                    break

                if split_full_text[index] == str(list(wrd.keys())[0]) and \
                        ix <= slice_wrds - 3 and text_slices[-1] != _slice:
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)
                elif split_full_text[index] == str(list(wrd.keys())[0]) and text_slices[-1] == _slice:
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)
        assert [i[0] for i in output_text] == split_full_text
        return output_text

    @staticmethod
    def punctuate_texts(full_pred: list):
        """
        Given a list of Predictions from the model, applies the predictions to text,
        thus punctuating it.
        """
        punct_resp = ""
        for i in full_pred:
            word, label = i
            if label[-1] == "U":
                punct_wrd = word.capitalize()
            else:
                punct_wrd = word

            if label[0] != "O":
                punct_wrd += label[0]

            punct_resp += punct_wrd + " "
        punct_resp = punct_resp.strip()
        # Append trailing period if doesnt exist.
        if punct_resp[-1].isalnum():
            punct_resp += "."
        return punct_resp

t1 = time.time()
f = open(r'hedging.txt','r')
hedging = [word.strip('\n') for word in list(f)]


filler_words = ['well', "hmm", "Um", "er", "uh", "like", "actually", "basically",
                "seriously", "you see", "you know", "I mean", "you know what I mean",
                "at the end of the day", "believe me", "I guess", "I suppose",
                "or something", "Okay", "so", "Right", "mhm", "uh", "huh",'ah']


def preprocess(res):
    #change i to I
    pattern = r'( i((?=\s)|(?=\')))'
    text = re.sub(pattern,' I',res['text'])
    res['text'] = text
    words = text.split(' ')
    for a,b in zip(words,res['result']):
        b['word'] = a
    return res

def convert_num(res): 
    res['text'] = words2number(res['text'])
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
    freq = {}
    # find repeted word next to each other
    for i in range(len(word_list)):
        if i < len(word_list)-1:
            if word_list[i] == word_list[i+1]:
                if word_list[i] in repeted_word.keys():
                    repeted_word[word_list[i]].append(i)
                else:
                    repeted_word[word_list[i]]= [i]
        if word_list[i] in freq.keys():
            freq[word_list[i]] += 1 
        else : 
            freq[word_list[i]] = 1 

    # bigram and trigram
    _2gram = [' '.join(e) for e in ngrams(word_list, 2)]
    _3gram = [' '.join(e) for e in ngrams(word_list, 3)]

    _2gram_duplicate = check_duplication_in_list(_2gram,2)
    _3gram_duplicate = check_duplication_in_list(_3gram,3)

    repeted_word.update(_2gram_duplicate)
    repeted_word.update(_3gram_duplicate)
    res['frequency'] = freq 
    res['repetition'] = repeted_word

    return res

def display(res):
    # print dictionary in good format
    json_obj = json.dumps(res,indent=4)
    print(json_obj)
    return

def merge_text(res): 
    result = res['result']
    res['text'] = ''
    for i in range(len(result)-1 ): 
      res['text']+= res['result'][i]['word']+' '
        
    return res 

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
    pausing = []
    for i in range(len(result)-1 ): 
        if res['result'][i]['stoptime'] > (res['stop_time']/(len(result)-1 )*1.5+1.5) :
            res['result'][i]['word']+= ' ...'
            pausing.append(res['result'][i]['stoptime'])
    res = merge_text(res)
    res['pausing'] = pausing
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

def restore_punct(res): 
    rpunct = RestorePuncts()
    res['text'] = rpunct.punctuate(res['text'])
    return res 

def split_sentences(res): 
    words = res['text'].split(' ')
    sentence = ''
    sentences = {}
    i = 0 
    txt = ''
    for word in words : 
        if '!' in word or '.' in word or '?' in word: 
            sentence += word+' '
            sentence = sentence[0].upper() + sentence[1:]
            sentences[i]  = {'text':sentence}
            txt += sentence 
            sentence = ''
            i += 1 
        else: 
            sentence += word+' '
    sentences[i] = {'text':sentence}
    res['sentences'] = sentences
    res['text'] = txt
    return res

def split_paragraph(res): 
    paragraph = ' '
    paragraphs = []
    wrd_speed = res['words_per_minutes']
    i = 0
    time = 30
    if time > res['stop_time']+res['speaking_time']:
        res ['paragraphs'] = [res['text']]
        return res

    
    while time < res['stop_time']+res['speaking_time']:
        while len(paragraph.split(' '))/wrd_speed <0.75 and i <len(res['sentences']) :
            paragraph += res['sentences'][i]['text']
            i += 1
        if paragraph : 
            paragraphs.append(paragraph)
        paragraph = ''
        time += 30
    n = len(res['sentences'])
    for k in range(i,n):
        paragraph += res['sentences'][k]['text']
    if paragraph: 
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

def get_emotion(res): 
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
    words = res['text'].split(' ')
    if len(words)>300 : 
        prediction = classifier(' '.join(words[:300]))
    else:
        prediction = classifier(res['text'])
    emotions = {}
    for emotion in prediction: 
        emotions[emotion['label']] = emotion['score']
    res['emotions'] = emotions 
    return res  

def fix_bug(res): 
    words = res['text'].split(' ')
    for i in range(1,len(words)-1): 
        word = words[i]
        if word =='...'and '.' in words[i-1]: 
            words[i-1]= words[i-1].replace('.','')
    res['text'] = ' '.join(words)
    txt = res['text']
    res['text']= re.sub("\s+", " ",res['text']) 
    return res 
  

def text_analysis(test): 
    test = preprocess(test)
    test = speaking_duration(test)
    test = find_filler_and_hedging(test)
    test = articulation(test)
    test=  repetition(test)
    test = word_speed(test)
    test = convert_num(test)
    test = restore_punct(test)
    test = split_sentences(test)
    test = split_paragraph(test)
    test = get_emotion(test)
    test = get_key_word(test)
    test = fix_bug(test)
    return test 


with open('data/sample2.json') as json_file:
    test = json.load(json_file)


test = text_analysis(test)
#display(test)
with open('output/output_sample7.json','w') as f :
    f.write(json.dumps(test,indent=4))
t2 = time.time()
print('total time: ',t2-t1)