import pathlib
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel

from torch.utils.tensorboard import SummaryWriter

from dataset import TweetDataset

def read_text_data(infile):
    out_file = []
    with open(infile, encoding="utf-8") as f:
        for line in f:
            out_file+=[line]
    return out_file

###################################
########Hyperparameters############
###################################
num_epochs = 30
temperature = 0.07
learning_rate = 5e-6
train_size=0.8
train_batch_size=64
val_batch_size=256
max_queue_size=0 #putting this to zero disables the momentum encoder queue
momentum_update_weight=0.99
max_collection_size = 0 # putting this and num_hard_..._per_sample to zero disables hard negatives
num_hard_negatives_per_sample=0
num_hard_positives_per_sample=0
early_stopping_threshold=3
debug_subsampling = 1
###################################
###################################
###################################

model_params = {
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
    "model" : "bert-base-uncased",
    "num_grus" : 4,
    "gru_hidden_size" : 100,
    "num_layers" : 1,
    "clf_hidden_size" : 100,
    "dropout" : 0.5,
    # "regularization" : "l1"
}

dataset_dir = str(pathlib.Path(__file__).parent.parent.parent.resolve())

class GruBERT(nn.Module):
    def __init__(self,params) -> None:
        super().__init__()
        
        self.device = params["device"]
        self.model_name = params["model"]
        # self.bert_model = AutoModel.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name,output_hidden_states=True)

        self.bert_model = self.bert_model
        self.num_grus = params["num_grus"]
        self.num_shared = int(12 / self.num_grus)
        
        gru_input_size = 768
        gru_hidden_size = params["gru_hidden_size"]
        clf_hidden_size = params["clf_hidden_size"]
        gru_layers = params["num_layers"]
        dropout_rate = params["dropout"]

        self.weight_shared_gru = nn.GRU(self.num_shared * gru_input_size, gru_hidden_size, num_layers=gru_layers, bidirectional=True)
        self.gru = nn.GRU(2 * self.num_grus * gru_hidden_size, gru_hidden_size, num_layers=gru_layers, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * gru_hidden_size, clf_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(clf_hidden_size,2),
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
        
    def forward(self,tokens,attention):
        embeddings = self.bert_model(input_ids=tokens, attention_mask=attention)
        embeddings = [torch.cat(embeddings[2][i * self.num_shared + 1 : (i+1) * self.num_shared + 1], dim=2) for i in range(self.num_grus)]

        permute_input = [embeddings[i].permute(1,0,2) for i in range(self.num_grus)]
        out_ws_gru = [self.weight_shared_gru(permute_input[i])[0] for i in range(self.num_grus)]
        out_ws_gru = torch.cat(out_ws_gru,2)

        out_gru, _ = self.gru(out_ws_gru)
        out_gru = F.relu(out_gru.permute(1,0,2))
        
        out_clf = (self.classifier(out_gru)).sum(dim=1)
        return out_clf

def train_model():

    writer = SummaryWriter(filename_suffix=f"-contrastive_approach-bs_{train_batch_size}-lr_{learning_rate}-qs_{max_queue_size}-updtwght_{momentum_update_weight}-cs_{max_collection_size}-hs_{num_hard_negatives_per_sample}-debug_{debug_subsampling}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_name = model_params["model"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = GruBERT(model_params).to(device)

    print(f"Running on {device}")
    # print(f"GPU name: {torch.cuda.get_device_name(device)}")
    # print(f"Device capabilities: {torch.cuda.get_device_capability(device)}")
    # print(f"Device properties: {torch.cuda.get_device_properties(device)}")

    pos_data = read_text_data(dataset_dir + "/twitter-datasets/train_pos.txt")
    neg_data = read_text_data(dataset_dir + "/twitter-datasets/train_neg.txt")

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

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

    lowest_val_loss = float("inf")
    val_accuracy_of_best_model = 0
    num_epochs_not_improved = 0

    #####################
    ### TRAINING LOOP ###
    #####################

    for epoch in range(num_epochs):
        model.train()
        for iteration, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            #Prepare input data
            inputs, labels = batch
            tokens = inputs["input_ids"].to(device)
            attention_masks = inputs["attention_mask"].to(device)
            labels = labels.to(device)
            
            #Forward Pass through GruBERT
            pred_logits = model(tokens, attention_masks)

            #Compute loss and perform Backpropagation
            batch_loss = loss_fn(pred_logits.to(device), labels)
            _, pred_classes = pred_logits.max(dim=1)

            train_accuracy = accuracy_score(pred_classes.cpu().numpy(), labels.cpu().numpy())
            train_precision = precision_score(pred_classes.cpu().numpy(), labels.cpu().numpy())
            train_recall = recall_score(pred_classes.cpu().numpy(), labels.cpu().numpy())

            batch_loss.backward()

            #Update weights
            optimizer.step()

            ###############
            ### LOGGING ###
            ###############

            writer.add_scalar("Training/Loss", batch_loss.item(), iteration + epoch * len(train_loader))
            writer.add_scalar("Training/Accuracy", train_accuracy, iteration + epoch * len(train_loader))
            writer.add_scalar("Training/Precision", train_precision, iteration + epoch * len(train_loader))
            writer.add_scalar("Training/Recall", train_recall, iteration + epoch * len(train_loader))

            if iteration%25==0:
                print(f"* epoch {epoch} - iteration {iteration} \t progress: {iteration / len(train_loader):.2%}")
                print(f"(Training) loss: {batch_loss:.3f} \t accuracy: {train_accuracy:.2%} \t precision: {train_precision:.2%} \t recall: {train_recall:.2%}")

            # if iteration%100==0:
            #     t = torch.cuda.get_device_properties(0).total_memory
            #     r = torch.cuda.memory_reserved(device)
            #     a = torch.cuda.memory_allocated(device)
            #     print(f"* memory reserved: {r/float(1000000):.0f} MB, memory allocated: {a/float(1000000):.0f} MB, free memory: {(t-r)/float(1000000):.0f}")

        lr_scheduler.step() #multiply learning rate by 0.9 after every epoch

        #######################
        ### VALIDATION LOOP ###
        #######################

        with torch.no_grad():
            model.eval()
            total_loss = 0.
            all_pred_classes = np.array([])
            all_true_classes = np.array([])

            for iteration, batch in enumerate(val_loader):

                #Prepare input data
                inputs, labels = batch
                tokens = inputs["input_ids"].to(device)
                attention_masks = inputs["attention_mask"].to(device)
                labels = labels.to(device)
                
                #Forward Pass through GruBERT
                pred_logits = model(tokens, attention_masks)

                #Compute validation loss / metrics
                val_batch_loss = loss_fn(pred_logits.to(device), labels)
                total_loss += val_batch_loss.item()

                _, pred_classes = pred_logits.max(dim=1)
                all_pred_classes = np.concatenate((all_pred_classes, pred_classes.cpu().numpy()),axis=0)
                all_true_classes = np.concatenate((all_true_classes,labels.cpu().numpy()),axis=0)

                if iteration%25==0:
                    print(f"* Validation iteration {iteration} \t progress: {iteration/len(val_loader):.2%}")

            val_accuracy = accuracy_score(all_pred_classes, all_true_classes)
            val_precision = precision_score(all_pred_classes, all_true_classes)
            val_recall = recall_score(all_pred_classes, all_true_classes)

            ###############
            ### LOGGING ###
            ###############
            print(f"(Validation) loss: {total_loss:.3f} \t accuracy: {val_accuracy:.2%} \t precision: {val_precision:.2%} \t recall: {val_recall:.2%}")

            writer.add_scalar("Validation/Loss", total_loss, epoch)
            writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)
            writer.add_scalar("Validation/Precision", val_precision, epoch)
            writer.add_scalar("Validation/Recall", val_recall, epoch)

            if (total_loss < lowest_val_loss): #do early stopping based on the validation loss
                num_epochs_not_improved = 0
                lowest_val_loss = total_loss
                val_accuracy_of_best_model = val_accuracy
                print(f"Found new best model with {lowest_val_loss:.6f} validation loss ({val_accuracy_of_best_model:.3f}% validation accuracy). Saving model...")
                torch.save(model.state_dict(), "best_grubert_params.pt")
            else:
                num_epochs_not_improved += 1

            if num_epochs_not_improved>=early_stopping_threshold: #if we have not improved for early_stopping_threshold epochs, stop the training
                break
    
    print(f"###### Finished training ######")

    print(f"Loading the  best checkpoint which had {lowest_val_loss:.6f} validation loss and {val_accuracy_of_best_model:.2%} validation accuracy...")
    model.load_state_dict(torch.load("best_grubert_params.pt"))

    return model

def test_model(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_data = read_text_data(dataset_dir + "/twitter-datasets/test_data.txt")

    tokenizer_name = model_params["model"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    test_dataset = TweetDataset(test_data, [-1]*len(test_data), tokenizer) #note: we don't use the labels here, we just need a dummy input
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, pin_memory=True, drop_last=False)

    #######################
    #### TESTING  LOOP ####
    #######################
    with torch.no_grad():
        model.eval()

        for iteration, batch in enumerate(test_loader):

            #Prepare input data
            inputs, labels = batch
            tokens = inputs["input_ids"].to(device)
            attention_masks = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            #Forward Pass through GruBERT
            pred_logits = model(tokens, attention_masks)
            _, pred_classes = pred_logits.max(dim=1)

            if iteration%25==0:
                print(f"* Testing iteration {iteration}, progress: {iteration/len(test_loader):.2%}")
        
        pred_classes = pred_classes.cpu()

        with open("test_predictions_grubert.csv", "w") as outfile:
            outfile.write("Id,Prediction\n")
            for i, prediction in enumerate(pred_classes):
                outfile.write(f"{i+1},{1 if prediction else -1}\n")

        print("###### Finished testing, terminating program ######")

if __name__ == "__main__":
    model = train_model()
    test_model(model)