class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """

        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        words = sorted(set(" ".join(texts).lower().split()))
        
        self.vocab_size = 4 + len(words)

        print(words)
        
        for i in range(len(words)):
            self.word_to_id[words[i]] = i + 4
            self.id_to_word[i + 4] = words[i]
    
    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        ids = []
        for word in text.lower().split():
            if word not in self.word_to_id:
                ids.append(self.word_to_id[self.unk_token])
            else:
                ids.append(self.word_to_id[word])
        
        return ids


    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        texts = []
        for id in ids:
            if id in self.id_to_word:
                texts.append(self.id_to_word[id])
            else:
                texts.append(self.unk_token)

        return " ".join(texts)