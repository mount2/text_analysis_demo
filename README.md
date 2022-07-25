# text analysis demo
simple text analysis


# Introduction 
Text analysis including finding filler words, finding repeted words, pacing and grading articulation. 

# Analysis

Use the text_analysis function in processing.py with dictionary as input.

# Setup 
1. Install nltk 

2. use  pip install --upgrade fastpunct to install fastpunct 

3. Clone the repo

4. Run processing.py to see the sample.

 # Output 
 
 Json file : 
 'result': contain all words and its information like 'conf','word','stoptime','speak_time' 
 'text' : text of the speech with punctuation restored and with some '...' respresenting speakers pausing.
 "stop_time": total stop time 
 "speaking_time": total speak time 
 "pausing" : pause time at each '...'
 'filler': return list of filler words in the text
 'hedging' : return list of hedging words
 'articulation': measure how clearly the speaker speaks
 'frequency': frequency of each word 
 'repetition' : words or bigram that repeted
 "words_per_minutes" : average number of words per minute 
 "average_speaking_time": average time to speak a word 
 "average_stop_time" : average pausing between words 
 "sentences": dictionary of all sentences with keys are the index of each sentence
 "paragraphs": list of all paragraphs.
 "emotion": emotion of the speech 
 "keys_word": list of 5 key words
 
 
 
