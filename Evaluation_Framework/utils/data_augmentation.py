from typing import List
from googletrans import Translator
from nltk.tokenize import sent_tokenize


default_languages = [
    'cs', # czech
    'de', # german
    'es', # spanish
    'fi', # finnish
    'fr', # french
    'hi', # hindi
    'it', # italian
    'ja', # japanese
    'pt', # portuguese
    'ru', # russian
    'vi', # vietnamese
    'zh-cn',  # chinese
]


class TranslationAugmentor:
    def __init__(self, languages: List[str]=default_languages):
        self.translator = Translator()
        self.languages = languages

    def paraphrase_sentence(self, sentence, language):
        translated = self.translator.translate(sentence, src='en', dest=language).text
        return self.translator.translate(translated, src=language, dest='en').text

    def paraphrase_document(self, doc: str) -> List[str]:
        """
        Args:
            article (str): The entire article in a single string

        Returns:
            List[str]: List of single-string paraphrased articles 
                       (not including the original article)
        """
        pivots = self.languages

        paraphrases = []
        tokenized = sent_tokenize(doc)
        for language in pivots:
            # The following lines should work (without iteration), but the googletrans
            # library doesn't like being passed a list
            # translated = self.translator.translate(tokenized, src='en', dest=language)
            # backtranslated = " ".join(self.translator.translate(translated, src=language, dest='en'))

            backtranslated = " ".join([self.paraphrase_sentence(s, language) for s in tokenized])
            paraphrases.append(backtranslated[:-1])
        return paraphrases

    def paraphrase_documents(self, docs: List[str]) -> List[List[str]]:
        """
        Args:
            docs (List[str]): Each string in the list is a complete document (summary or article)
        """
        paraphrased_docs = []
        for doc in docs:
            paraphrased_docs.append(self.back_translate_document(doc))
        return paraphrased_docs
