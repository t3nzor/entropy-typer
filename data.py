import os
from io import open
import torch
import tiktoken


class Corpus(object):
    def __init__(self, path):
        # Choose from: o200k_base cl100k_base p50k_base r50k_base
        self.tokenizer = tiktoken.get_encoding("p50k_base")
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                ids = self.tokenizer.encode(line)
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
