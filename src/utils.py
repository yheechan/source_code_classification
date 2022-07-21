from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np


def split(str):
    return [char for char in str]

def tokenize(sourceCode):
    """Tokenize texts, build vocabulary and find maximum sentence length.
    
    Args:
        texts (List[str]): List of text data
    
    Returns:
        tokenized_texts (List[List[str]]): List of list of tokens
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum sentence length
    """

    max_len = 0
    tokenized_codes = []
    ch2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    ch2idx['<pad>'] = 0

    # Building our vocab from the corpus starting from index 2
    idx = 1
    for code in sourceCode:
        tokenized_code = split(code)

        # Add `tokenized_sent` to `tokenized_texts`
        tokenized_codes.append(tokenized_code)

        # Add new token to `word2idx`
        for token in tokenized_code:
            if token not in ch2idx:
                ch2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_code))

    return tokenized_codes, ch2idx, max_len

def encode(tokenized_codes, ch2idx, max_len):
    """Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """

    input_ids = []
    for tokenized_code in tokenized_codes:
        # Pad sentences to max_len
        tokenized_code += ['<pad>'] * (max_len - len(tokenized_code))

        # Encode tokens to input_ids
        input_id = [ch2idx.get(token) for token in tokenized_code]
        input_ids.append(input_id)
    
    return np.array(input_ids)

def tokenize_encode_class(classes):
    encoded_class = []
    class2idx = {}
    idx = 0

    for one_class in classes:

        if not one_class in class2idx:
            class2idx[one_class] = idx
            idx += 1
        
        encoded_class.append(class2idx[one_class])

    return np.array(encoded_class), class2idx, len(class2idx)