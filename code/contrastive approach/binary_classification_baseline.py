import torch
from torch import nn
import numpy as np
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

from dataset import TweetDataset

#adapted from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f#:~:text=BERT%20base%2C%20which%20is%20a,hidden%20size%2C%20and%20340%20parameters as a baseline
class BinaryClassificationTransformer(nn.Module):
    def __init__(self, model_name, dropout_rate=0.5):
        super(BinaryClassificationTransformer, self).__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(768, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_ouptut = self.transformer(input_ids=input_ids, attention_mask = attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_ouptut)
        logits = self.linear(dropout_output)
        #final_layer = self.relu(linear_output)

        return logits


def read_text_data(infile):
    out_file = []
    with open(infile, encoding="utf-8") as f:
        for line in f:
            out_file += [line]
    return out_file


########Hyperparameters############
num_epochs = 30
temperature = 0.07
learning_rate = 1e-5
train_size=0.7
train_batch_size=64
val_batch_size=256
early_stopping_threshold=3
debug_subsampling = 1
##################################

writer = SummaryWriter(filename_suffix=f"-binary_cls-bs_{train_batch_size}-lr_{learning_rate}-debug_{debug_subsampling}")
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = BinaryClassificationTransformer(model_name).to(device)

print(f"Running on {device}")
print(f"GPU name: {torch.cuda.get_device_name(device)}")
print(f"Device properties: {torch.cuda.get_device_properties(device)}")

pos_data = read_text_data("../../twitter-datasets/train_pos.txt")
neg_data = read_text_data("../../twitter-datasets/train_neg.txt")

pos_data_train, pos_data_val = train_test_split(pos_data[0:int(len(pos_data)*debug_subsampling)], train_size=train_size)
neg_data_train, neg_data_val = train_test_split(neg_data[0:int(len(neg_data)*debug_subsampling)], train_size=train_size)

full_train_dataset = pos_data_train + neg_data_train
train_labels = [1] * len(pos_data_train) + [0] * len(neg_data_train)

full_val_dataset = pos_data_val + neg_data_val
val_labels = [1] * len(pos_data_val) + [0] * len(neg_data_val)

train_data = TweetDataset(full_train_dataset, train_labels, tokenizer)
val_data = TweetDataset(full_val_dataset, val_labels, tokenizer)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=True, pin_memory=True, drop_last=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scaler = torch.cuda.amp.GradScaler() #for fp16

lowest_val_loss = float("inf")
val_accuracy_of_best_model = 0

num_epochs_no_improvement = 0

for epoch in range(num_epochs):
    model.train()
    for iteration, batch in enumerate(train_loader):
        optimizer.zero_grad()

        inputs, labels = batch
        tokens = inputs["input_ids"].to(device)
        attention_masks = inputs["attention_mask"].to(device)
        labels = labels.float().to(device)

        with torch.cuda.amp.autocast():#for fp16
            logits = model(input_ids=tokens, attention_mask=attention_masks).reshape(-1)
            loss = torch.nn.BCEWithLogitsLoss()(logits, labels)

        #loss.backward()
        scaler.scale(loss).backward() #for fp16
        scaler.step(optimizer)
        scaler.update()
        #optimizer.step()

        #LOGGING
        writer.add_scalar("Baseline_Loss/training", loss.item(), iteration + epoch * len(train_loader))

        if iteration%25==0:
            print(f"* epoch {epoch} - iteration {iteration}, loss {loss.item():.6f}, progress: {100 * iteration / len(train_loader):.2f}%")

        if iteration%100==0:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(device)
            a = torch.cuda.memory_allocated(device)
            print(f"* memory reserved: {r/float(1000000):.0f} MB, memory allocated: {a/float(1000000):.0f} MB, free memory: {(t-r)/float(1000000):.0f}")

    #validation loop
    with torch.no_grad():
        model.eval()

        #compute accuracy on validation set
        correct_counter = 0
        total_counter = 0
        loss_sum = 0
        for iteration, batch in enumerate(val_loader):
            inputs, labels = batch
            tokens = inputs["input_ids"].to(device)
            attention_masks = inputs["attention_mask"].to(device)
            labels = labels.float().to(device)

            logits = model(input_ids=tokens, attention_mask=attention_masks).reshape(-1)

            predictions = torch.nn.Sigmoid()(logits)

            loss = torch.nn.BCEWithLogitsLoss()(logits, labels)

            loss_sum += loss.item()

            correct_counter += torch.sum(torch.round(predictions)==labels).item()
            total_counter += tokens.shape[0]
            if iteration%25==0:
                print(f"* Validation iteration {iteration}, progress: {100*iteration/len(val_loader):.2f}%")

        val_accuracy = 100*correct_counter/float(total_counter)

        print(f"Baseline_Validation accuracy: {val_accuracy:.3f}%; Validation loss: {loss_sum:.3f}")
        writer.add_scalar("Baseline_Validation/accuracy", 100*correct_counter/float(total_counter), epoch)
        writer.add_scalar("Baseline_Validation/loss", loss_sum, epoch)

        if (loss_sum < lowest_val_loss): #do early stopping based on the validation loss
            num_epochs_no_improvement = 0
            lowest_val_loss = loss_sum
            val_accuracy_of_best_model = val_accuracy
            print(f"Found new best model with {lowest_val_loss:.6f} validation loss ({val_accuracy_of_best_model:.3f}% validation accuracy). Saving model...")
            torch.save(model.state_dict(), "best_model_parameters.pt")
        else:
            num_epochs_no_improvement += 1

        if num_epochs_no_improvement>=early_stopping_threshold: #if we have not improved for early_stopping_threshold epochs, stop the training
            break

print(f"###### Finished training ######")

print(f"Loading the  best checkpoint which had {lowest_val_loss:.6f} validation loss and {val_accuracy_of_best_model:.3f}% validation accuracy...")
model.load_state_dict(torch.load("best_model_parameters.pt"))

####### TESTING #######
test_data = read_text_data("../../twitter-datasets/test_data.txt")
test_dataset = TweetDataset(test_data, [-1]*len(test_data), tokenizer) #note: we don't use the labels here, we just need a dummy input
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, pin_memory=True, drop_last=False) #note: we don't want to "drop_last", so all batches must be the same size; since there's 10k samples, we can set batch_size to 10 to ensure all batches have full size

with torch.no_grad():
    model.eval()
    #predict
    test_predictions = torch.empty((0), dtype=int).to(device)
    for iteration, batch in enumerate(test_loader):
        inputs, labels = batch
        tokens = inputs["input_ids"].to(device)
        attention_masks = inputs["attention_mask"].to(device)

        logits = model(input_ids=tokens, attention_mask=attention_masks).reshape(-1)

        current_predictions = torch.round(torch.nn.Sigmoid()(logits)).int()

        test_predictions = torch.cat((test_predictions, current_predictions), dim=0)

        if iteration%25==0:
            print(f"* Testing iteration {iteration}, progress: {100*iteration/len(test_loader):.2f}%")

    test_predictions = test_predictions.cpu()

    # Write predictions to file
    with open("test_predictions.csv", "w") as outfile:
        outfile.write("Id,Prediction\n")
        for i, prediction in enumerate(test_predictions):
            outfile.write(f"{i+1},{1 if prediction else -1}\n")

print("###### Finished testing, terminating program ######")




