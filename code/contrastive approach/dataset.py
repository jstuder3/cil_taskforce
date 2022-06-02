import torch

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, labels, tokenizer):
        #tokenize data and put it into datasets
        self.tokenized_texts = []
        self.attention_masks = []
        self.labels = labels

        for i in range(len(text_list)):
            if i% 20000 == 0:
                print(f"Tokenization progress: {i} / {len(text_list)} ({100*i/float(len(text_list)):.2f}%)")
            tokenizer_output = tokenizer(text_list[i], padding="max_length", truncation="longest_first")
            self.tokenized_texts.append(tokenizer_output["input_ids"])
            self.attention_masks.append(tokenizer_output["attention_mask"])

        print(f"Tokenization done. Converting data to tensors...")

        self.tokenized_texts = torch.tensor(self.tokenized_texts)
        self.attention_masks = torch.tensor(self.attention_masks)
        self.labels = torch.tensor(self.labels)

        print(f"Dataset construction done!")

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, index):
        return {"input_ids": self.tokenized_texts[index], "attention_mask": self.attention_masks[index]}, self.labels[index]


