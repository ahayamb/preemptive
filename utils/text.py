from typing import List
from keras.preprocessing.text import Tokenizer, hashing_trick


class Tokzr(object):
    def tokenize(self, text: str):
        pass


class TokzTokenizer(Tokzr):
    def __init__(self, tokenizer: Tokenizer):
        self.tkz = tokenizer
    
    def tokenize(self, text: str):
        return self.tkz.texts_to_sequences([text])[0]


class HashTokenizer(Tokzr):
    def __init__(self, max_words):
        self.max_words = max_words
    
    def tokenize(self, text: str):
        return hashing_trick(text, self.max_words, hash_function='md5')


def build_tokenizer(texts: List[str], num_words: int):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)

    return tokenizer
