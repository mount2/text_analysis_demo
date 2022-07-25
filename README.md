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
 1. 'result': contain all words and its information like 'conf','word','stoptime','speak_time' 
 1. 'text' : text of the speech with punctuation restored and with some '...' respresenting speakers pausing.
 1. "stop_time": total stop time 
 1. "speaking_time": total speak time 
 1. "pausing" : pause time at each '...'
 1. 'filler': return list of filler words in the text
 1. 'hedging' : return list of hedging words
 1. 'articulation': measure how clearly the speaker speaks
 1. 'frequency': frequency of each word 
 1. 'repetition' : words or bigram that repeted
 1. "words_per_minutes" : average number of words per minute 
 1. "average_speaking_time": average time to speak a word 
 1. "average_stop_time" : average pausing between words 
 1. "sentences": dictionary of all sentences with keys are the index of each sentence
 1. "paragraphs": list of all paragraphs.
 1. "emotion": emotion of the speech 
 1. "keys_word": list of 5 key words
 
 
 
