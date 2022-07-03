
from deepcorrect import DeepCorrect
corrector = DeepCorrect('params_path', 'checkpoint_path')
corrector.correct('hey')
'Hey!'