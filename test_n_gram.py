from nltk.util import ngrams
text = "I am batman and I like coffee"
_1gram = text.split(" ")
_2gram = [' '.join(e) for e in ngrams(_1gram, 2)]
_3gram = [' '.join(e) for e in ngrams(_1gram, 3)]

print(type(_1gram))
print(_2gram)
print(_3gram)