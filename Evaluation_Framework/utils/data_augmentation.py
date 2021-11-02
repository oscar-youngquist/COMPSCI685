from typing import List
from googletrans import Translator

default_languages = [
    'cs',  # czech
    'de',  # german
    'es', # spanish
    'fi',  # finnish
    'fr', # french
    'hi', # hindi
    'it', # italian
    'ja', # japanese
    'pt', # portuguese
    'ru', # russian
    'vi', # vietnamese
    'zh-cn',  # chinese
]


class Augmentor:

    def __init__(self):
        self.translator = Translator()

    def back_translate(self, sentence: str, languages: List[str]=default_languages) -> List[str]:
        paraphrases = []
        for language in languages:
            translated = self.translator.translate(sentence, src='en', dest=language)
            backtranslated = self.translator.translate(translated, src=language, dest='en')
            paraphrases.append(backtranslated)
        return paraphrases
