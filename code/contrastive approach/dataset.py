import torch
import re

def remove_multi_spaces(input_data):
    out_data = []
    for elem in input_data:
        elem=re.sub(" +", " ", elem)
        out_data+=[elem]
    return out_data

def replace_usr_and_url(input_data):
    out_data = []
    for elem in input_data:
        elem = re.sub("<user>", "xxuser", elem)
        elem = re.sub("<url>", "xxurl", elem)
        out_data+=[elem]
    return out_data

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, labels, tokenizer):

        #preprocess data: <user> is replaced by xxuser, <url> is replaced by xxurl and multi-spaces are removed
        text_list = replace_usr_and_url(text_list)
        text_list = remove_multi_spaces(text_list)

        #tokenize data and put it into datasets
        self.tokenized_texts = []
        self.attention_masks = []
        self.labels = labels

        for i in range(len(text_list)):
            if i% 20000 == 0:
                print(f"Tokenization progress: {i} / {len(text_list)} ({100*i/float(len(text_list)):.2f}%)")
            tokenizer_output = tokenizer(text_list[i], padding="max_length", truncation="longest_first", max_length=40)
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


