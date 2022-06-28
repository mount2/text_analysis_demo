import json

dictionary = {'directory': '/content/2022-03-20 13-05-39.wav',
 'result': [{'conf': 0.894261, 'end': 0.18, 'start': 0.03, 'word': 'well'},
  {'conf': 0.815786, 'end': 0.209106, 'start': 0.18, 'word': 'i'},
  {'conf': 1.0, 'end': 0.36, 'start': 0.209106, 'word': 'have'},
  {'conf': 1.0, 'end': 0.81, 'start': 0.42, 'word': 'equal'},
  {'conf': 1.0, 'end': 1.11, 'start': 0.81, 'word': 'rights'},
  {'conf': 1.0, 'end': 1.32, 'start': 1.11, 'word': 'for'},
  {'conf': 1.0, 'end': 1.65, 'start': 1.32, 'word': 'all'},
  {'conf': 1.0, 'end': 2.46, 'start': 2.13, 'word': 'except'},
  {'conf': 1.0, 'end': 2.85, 'start': 2.46, 'word': 'blacks'},
  {'conf': 1.0, 'end': 3.3, 'start': 2.88, 'word': 'asians'},
  {'conf': 1.0, 'end': 3.84, 'start': 3.3, 'word': 'hispanics'},
  {'conf': 1.0, 'end': 4.2, 'start': 3.84, 'word': 'jews'},
  {'conf': 0.457167, 'end': 4.53, 'start': 4.2, 'word': 'gays'},
  {'conf': 1.0, 'end': 4.95, 'start': 4.533771, 'word': 'women'},
  {'conf': 1.0, 'end': 5.67, 'start': 4.98, 'word': 'muslims'},
  {'conf': 1.0, 'end': 7.02, 'start': 6.63, 'word': 'everybody'},
  {'conf': 0.739733, 'end': 7.11, 'start': 7.02, 'word': 'is'},
  {'conf': 1.0, 'end': 7.29, 'start': 7.11, 'word': 'not'},
  {'conf': 1.0, 'end': 7.35, 'start': 7.29, 'word': 'a'},
  {'conf': 1.0, 'end': 7.59, 'start': 7.35, 'word': 'white'},
  {'conf': 1.0, 'end': 7.92, 'start': 7.59, 'word': 'man'},
  {'conf': 1.0, 'end': 8.52, 'start': 8.4, 'word': 'and'},
  {'conf': 1.0, 'end': 8.55, 'start': 8.52, 'word': 'i'},
  {'conf': 1.0, 'end': 8.7, 'start': 8.55, 'word': 'mean'},
  {'conf': 1.0, 'end': 9.06, 'start': 8.7, 'word': 'white'},
  {'conf': 0.693898, 'end': 9.39, 'start': 9.06, 'word': 'white'},
  {'conf': 0.436075, 'end': 9.51, 'start': 9.399758, 'word': 'you'},
  {'conf': 0.436075, 'end': 9.63, 'start': 9.51, 'word': 'know'},
  {'conf': 1.0, 'end': 10.229999, 'start': 9.63, 'word': 'italians'},
  {'conf': 0.529101, 'end': 10.409999, 'start': 10.229999, 'word': 'know'},
  {'conf': 1.0, 'end': 10.92, 'start': 10.41, 'word': 'polish'},
  {'conf': 1.0, 'end': 11.34, 'start': 11.16, 'word': 'just'},
  {'conf': 1.0, 'end': 11.61, 'start': 11.34, 'word': 'people'},
  {'conf': 1.0, 'end': 11.79, 'start': 11.61, 'word': 'from'},
  {'conf': 1.0, 'end': 12.3, 'start': 11.82, 'word': 'ireland'},
  {'conf': 1.0, 'end': 12.66, 'start': 12.3, 'word': 'england'},
  {'conf': 1.0, 'end': 12.75, 'start': 12.66, 'word': 'and'},
  {'conf': 1.0, 'end': 13.26, 'start': 12.75, 'word': 'scotland'},
  {'conf': 1.0, 'end': 13.89, 'start': 13.74, 'word': 'but'},
  {'conf': 1.0, 'end': 14.07, 'start': 13.89, 'word': 'only'},
  {'conf': 1.0, 'end': 14.4, 'start': 14.07, 'word': 'certain'},
  {'conf': 1.0, 'end': 14.67, 'start': 14.4, 'word': 'parts'},
  {'conf': 1.0, 'end': 14.76, 'start': 14.67, 'word': 'of'},
  {'conf': 1.0, 'end': 15.18, 'start': 14.76, 'word': 'scotland'},
  {'conf': 1.0, 'end': 15.27, 'start': 15.18, 'word': 'and'},
  {'conf': 1.0, 'end': 15.63, 'start': 15.27, 'word': 'ireland'},
  {'conf': 1.0, 'end': 16.32, 'start': 16.08, 'word': 'just'},
  {'conf': 1.0, 'end': 16.62, 'start': 16.32, 'word': 'full'},
  {'conf': 1.0, 'end': 17.01, 'start': 16.62, 'word': 'blooded'},
  {'conf': 0.437938, 'end': 17.49, 'start': 17.01, 'word': 'whites'},
  {'conf': 0.421962, 'end': 18.086924, 'start': 17.91, 'word': 'not'},
  {'conf': 1.0, 'end': 18.15, 'start': 18.086924, 'word': 'you'},
  {'conf': 1.0, 'end': 18.24, 'start': 18.15, 'word': 'know'},
  {'conf': 1.0, 'end': 18.45, 'start': 18.24, 'word': 'what'},
  {'conf': 1.0, 'end': 18.87, 'start': 18.69, 'word': 'not'},
  {'conf': 1.0, 'end': 19.05, 'start': 18.87, 'word': 'even'},
  {'conf': 0.620904, 'end': 19.41, 'start': 19.05, 'word': 'white'},
  {'conf': 1.0, 'end': 20.07, 'start': 19.77, 'word': 'nobody'},
  {'conf': 1.0, 'end': 20.25, 'start': 20.07, 'word': 'gets'},
  {'conf': 1.0, 'end': 20.4, 'start': 20.25, 'word': 'any'},
  {'conf': 1.0, 'end': 20.76, 'start': 20.4, 'word': 'right'},
  {'conf': 0.438004, 'end': 22.05, 'start': 21.72, 'word': 'ah'},
  {'conf': 1.0, 'end': 22.92, 'start': 22.17, 'word': 'america'}],
 'text': 'well i have equal rights for all except blacks asians hispanics jews gays women muslims everybody is not a white man and i mean white white you know italians know polish just people from ireland england and scotland but only certain parts of scotland and ireland just full blooded whites not you know what not even white nobody gets any right ah america'}

with open("sample.json", "w") as outfile:
    json.dump(dictionary, outfile,indent=4)