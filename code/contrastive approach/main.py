import torch
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
learning_rate = 1e-5
train_size=0.7
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

writer = SummaryWriter(filename_suffix=f"-contrastive_approach-bs_{train_batch_size}-lr_{learning_rate}-qs_{max_queue_size}-updtwght_{momentum_update_weight}-cs_{max_collection_size}-hs_{num_hard_negatives_per_sample}-debug_{debug_subsampling}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name).to(device)

#second encoder whose weights will be a weighted average over time of the main encoder
momentum_encoder = AutoModel.from_pretrained(model_name).to(device)
momentum_encoder.eval()

#queue to hold samples from the momentum encoder
embeddings_queue = torch.empty((0, 768), dtype=torch.float32).to(device)
labels_queue = torch.empty((0), dtype=int).to(device)
current_queue_index = 0

#set of "negative sentiment" samples (effectively another, larger queue, but with only one type of labels)
negative_sentiment_collection = torch.empty((0, 768), dtype=torch.float32).to(device)
negative_sentiment_collection_inputs = {"input_ids": torch.empty((0, 40), dtype=int), "attention_mask": torch.empty((0, 40), dtype=int)}
current_negative_sentiment_collection_index = 0

#set of "positive sentiment" samples (effectively another, larger queue, but with only one type of labels)
positive_sentiment_collection = torch.empty((0, 768), dtype=torch.float32).to(device)
positive_sentiment_collection_inputs = {"input_ids": torch.empty((0, 40), dtype=int), "attention_mask": torch.empty((0, 40), dtype=int)}
current_positive_sentiment_collection_index = 0

print(f"Running on {device}")
print(f"GPU name: {torch.cuda.get_device_name(device)}")
#print(f"Device capabilities: {torch.cuda.get_device_capability(device)}")
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
larger_batch_size_train_loader = torch.utils.data.DataLoader(train_data, batch_size=val_batch_size, shuffle=True, pin_memory=True, drop_last=True) #used for the "complete forward pass" during validation (because changing the batch size in dataloaders is apparently not comletely straightforward)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=True, pin_memory=True, drop_last=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

scaler = torch.cuda.amp.GradScaler() #for fp16

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

        #######################
        ### MOMENTUM UPDATE ###
        #######################

        #momentum update (adapted from https://github.com/jstuder3/nl-pl_moco/blob/main/xMoCo_pl.py#L191)
        for param_fast, param_slow in zip(model.parameters(), momentum_encoder.parameters()):
            param_slow.data = param_slow.data * momentum_update_weight + param_fast.data * (1.0 - momentum_update_weight)

        #############################################
        ### DATA PREPARATION AND LABEL GENERATION ###
        #############################################

        inputs, labels = batch
        tokens = inputs["input_ids"].to(device)
        attention_masks = inputs["attention_mask"].to(device)
        labels = labels.float().to(device)


        #generate labels for the cross_entropy loss w.r.t. in-batch samples
        in_batch_cross_entropy_labels = torch.zeros((train_batch_size, train_batch_size))
        for i in range(train_batch_size):
            line = (labels == labels[i]).int()
            in_batch_cross_entropy_labels[i] = line
        in_batch_cross_entropy_labels = in_batch_cross_entropy_labels.to(device)

        #generate labels for the cross_entropy loss w.r.t the queue
        queue_cross_entropy_labels = torch.zeros((train_batch_size, labels_queue.shape[0]))
        for i in range(train_batch_size):
            line = (labels_queue==labels[i]).int()
            queue_cross_entropy_labels[i] = line
        queue_cross_entropy_labels = queue_cross_entropy_labels.to(device)

        #generate slicing indices which we later use to gather hard negatives from the correct sentiment class
        positive_sentiment_indices = []
        negative_sentiment_indices = []
        for i in range(train_batch_size):
            if labels[i] == 0:
                negative_sentiment_indices.append(i)
            else:
                positive_sentiment_indices.append(i)

        with torch.cuda.amp.autocast(): #for fp16

            ####################
            ### FORWARD PASS ###
            ####################

            embeddings = model(input_ids=tokens, attention_mask=attention_masks)["pooler_output"]
            embeddings = F.normalize(embeddings, p=2, dim=1) #normalize

            in_batch_logits = torch.matmul(embeddings, embeddings.T)
            queue_logits = torch.matmul(embeddings, embeddings_queue.T)

            ######################
            ### HARD NEGATIVES ###
            ######################

            #for each of the in-batch samples, find the sample in the (respectively other sentiment) collection that produces the largest dot product (that means that the encoder has trouble separating these samples)
            _, positive_sentiment_hard_negative_indices = torch.matmul(embeddings[positive_sentiment_indices], negative_sentiment_collection.T).topk(k=min(num_hard_negatives_per_sample, negative_sentiment_collection.shape[0]), largest=True, dim=1) #contains the indices which serve as hard negatives for the positive sentiment in-batch samples
            _, negative_sentiment_hard_negative_indices = torch.matmul(embeddings[negative_sentiment_indices], positive_sentiment_collection.T).topk(k=min(num_hard_negatives_per_sample, positive_sentiment_collection.shape[0]), largest=True, dim=1) #contains the indices which serve as hard negatives for the negative sentiment in-batch samples

            #flatten and take only unique values to prevent wasting computations
            positive_sentiment_hard_negative_indices = positive_sentiment_hard_negative_indices.flatten().unique()
            negative_sentiment_hard_negative_indices = negative_sentiment_hard_negative_indices.flatten().unique()

            #now get the raw values so we can generate updated embeddings
            positive_sentiment_hard_negative_inputs = {"input_ids": negative_sentiment_collection_inputs["input_ids"][positive_sentiment_hard_negative_indices], "attention_mask": negative_sentiment_collection_inputs["attention_mask"][positive_sentiment_hard_negative_indices]}
            negative_sentiment_hard_negative_inputs = {"input_ids": positive_sentiment_collection_inputs["input_ids"][negative_sentiment_hard_negative_indices], "attention_mask": positive_sentiment_collection_inputs["attention_mask"][negative_sentiment_hard_negative_indices]}

            hard_negative_inputs = {"input_ids": torch.vstack((positive_sentiment_hard_negative_inputs["input_ids"], negative_sentiment_hard_negative_inputs["input_ids"])), "attention_mask": torch.vstack((positive_sentiment_hard_negative_inputs["attention_mask"], negative_sentiment_hard_negative_inputs["attention_mask"]))}

            with torch.no_grad():
                #can either use momentum_encoder or the regular model
                hard_negative_embeddings = model(input_ids=hard_negative_inputs["input_ids"].to(device), attention_mask = hard_negative_inputs["attention_mask"].to(device))["pooler_output"]

                #compute updated similarities
                hard_negative_logits = torch.matmul(embeddings, hard_negative_embeddings.T)

                #generate labels for the hard negative part
                hard_negative_cross_entropy_labels = torch.empty((train_batch_size, hard_negative_embeddings.shape[0])).to(device) #note that hard_negative_embeddings.shape[0] can be different from train_batch_size * num_hard_negatives_per_sample because we call .unique() above

                #the first few hard negatives are of positive sentiment and the last few are of negative sentiment
                positive_label = torch.tensor([0] * positive_sentiment_hard_negative_indices.shape[0] + [1] * negative_sentiment_hard_negative_indices.shape[0]).to(device)
                negative_label = torch.tensor([1] * positive_sentiment_hard_negative_indices.shape[0] + [0] * negative_sentiment_hard_negative_indices.shape[0]).to(device)

                #depending on the sentiment of the in-batch samples, insert a different label row
                for i in range(train_batch_size):
                    if labels[i] == 1:
                        hard_negative_cross_entropy_labels[i] = positive_label
                    else:
                        hard_negative_cross_entropy_labels[i] = negative_label

            ######################
            ### HARD POSITIVES ###
            ######################

            # for each of the in-batch samples, find the sample in the (respectively same sentiment) collection that produces the smallest dot product (that means that the encoder has trouble pulling these samples together)
            _, positive_sentiment_hard_positive_indices = torch.matmul(embeddings[positive_sentiment_indices], positive_sentiment_collection.T).topk(k=min(num_hard_positives_per_sample, positive_sentiment_collection.shape[0]), largest=False, dim=1)  # contains the indices which serve as hard negatives for the positive sentiment in-batch samples
            _, negative_sentiment_hard_positive_indices = torch.matmul(embeddings[negative_sentiment_indices], negative_sentiment_collection.T).topk(k=min(num_hard_positives_per_sample, negative_sentiment_collection.shape[0]), largest=False, dim=1)  # contains the indices which serve as hard negatives for the negative sentiment in-batch samples

            #flatten and take only unique values to prevent wasting computations
            positive_sentiment_hard_positive_indices = positive_sentiment_hard_positive_indices.flatten().unique()
            negative_sentiment_hard_positive_indices = negative_sentiment_hard_positive_indices.flatten().unique()

            #now get the raw values so we can generate updated embeddings
            positive_sentiment_hard_positive_inputs = {"input_ids": positive_sentiment_collection_inputs["input_ids"][positive_sentiment_hard_positive_indices], "attention_mask": positive_sentiment_collection_inputs["attention_mask"][positive_sentiment_hard_positive_indices]}
            negative_sentiment_hard_positive_inputs = {"input_ids": negative_sentiment_collection_inputs["input_ids"][negative_sentiment_hard_positive_indices], "attention_mask": negative_sentiment_collection_inputs["attention_mask"][negative_sentiment_hard_positive_indices]}

            hard_positive_inputs = {"input_ids": torch.vstack((positive_sentiment_hard_positive_inputs["input_ids"], negative_sentiment_hard_positive_inputs["input_ids"])), "attention_mask": torch.vstack((positive_sentiment_hard_positive_inputs["attention_mask"], negative_sentiment_hard_positive_inputs["attention_mask"]))}

            with torch.no_grad():
                # can either use momentum_encoder or the regular model
                hard_positive_embeddings = model(input_ids=hard_positive_inputs["input_ids"].to(device), attention_mask=hard_positive_inputs["attention_mask"].to(device))["pooler_output"]

                #compute updated similarities
                hard_positive_logits = torch.matmul(embeddings, hard_positive_embeddings.T)

                # generate labels for the hard positive part
                hard_positive_cross_entropy_labels = torch.empty((train_batch_size, hard_positive_embeddings.shape[0])).to(device)  # note that hard_positive_embeddings.shape[0] can be different from train_batch_size * num_hard_positives_per_sample because we call .unique() above

                #the first few hard negatives are of positive sentiment and the last few are of negative sentiment
                positive_label = torch.tensor([1] * positive_sentiment_hard_positive_indices.shape[0] + [0] * negative_sentiment_hard_positive_indices.shape[0]).to(device)
                negative_label = torch.tensor([0] * positive_sentiment_hard_positive_indices.shape[0] + [1] * negative_sentiment_hard_positive_indices.shape[0]).to(device)

                #depending on the sentiment of the in-batch samples, insert a different label row
                for i in range(train_batch_size):
                    if labels[i] == 1:
                        hard_positive_cross_entropy_labels[i] = positive_label
                    else:
                        hard_positive_cross_entropy_labels[i] = negative_label

            ########################################################################
            ### CONCATENATE ALL OF THE PRODUCED VALUES TO COMPUTE THE FINAL LOSS ###
            ########################################################################

            logits = torch.hstack((in_batch_logits, queue_logits, hard_negative_logits, hard_positive_logits))
            complete_cross_entropy_labels = torch.hstack((in_batch_cross_entropy_labels, queue_cross_entropy_labels, hard_negative_cross_entropy_labels, hard_positive_cross_entropy_labels))

            loss = torch.nn.CrossEntropyLoss()(logits/temperature, complete_cross_entropy_labels)

        scaler.scale(loss).backward() #for fp16
        scaler.step(optimizer) #for fp16
        scaler.update() #for fp16

        #loss.backward()

        #optimizer.step()

        ##################################
        ### QUEUE AND COLLETION UPDATE ###
        ##################################

        #update the queue
        with torch.no_grad():
            momentum_embeddings = momentum_encoder(input_ids=tokens, attention_mask=attention_masks)["pooler_output"]
            momentum_embeddings = F.normalize(momentum_embeddings, p=2, dim=1)
            if (max_queue_size >= train_batch_size):  # allow to disable the queue by only updating it if it can even hold an entire batch
                if embeddings_queue.shape[0] < max_queue_size: #if the queue isn't full already, just append the new embeddings to the end of it
                    embeddings_queue = torch.vstack((embeddings_queue, momentum_embeddings))
                    labels_queue = torch.cat((labels_queue, labels), dim=0)
                    current_queue_index += train_batch_size
                else: #otherwise, we need to replace things that are already in the queue
                    if current_queue_index >= max_queue_size:
                        current_queue_index = 0
                    embeddings_queue[current_queue_index:(current_queue_index+train_batch_size)] = momentum_embeddings
                    labels_queue[current_queue_index:(current_queue_index+train_batch_size)] = labels
                    current_queue_index += train_batch_size

            if(max_collection_size >= train_batch_size):

                #do the same thing for the collections (but only append to the corresponding sentiment class collection)
                positive_sentiment_momentum_embeddings = momentum_embeddings[positive_sentiment_indices]
                negative_sentiment_momentum_embeddings = momentum_embeddings[negative_sentiment_indices]

                #extract inputs because we will need them later when we want to get an updated view of the hard negatives
                positive_sentiment_inputs = {"input_ids": batch[0]["input_ids"][positive_sentiment_indices], "attention_mask": batch[0]["attention_mask"][positive_sentiment_indices]}
                negative_sentiment_inputs = {"input_ids": batch[0]["input_ids"][negative_sentiment_indices], "attention_mask": batch[0]["attention_mask"][negative_sentiment_indices]}

                num_positive_sentiment_samples = len(positive_sentiment_indices)
                num_negative_sentiment_samples = len(negative_sentiment_indices)

                #append the positive sentiment samples to the positive sentiment collection
                if positive_sentiment_collection.shape[0] < max_collection_size: #if the collection isn't already full, just append the new embeddings to the end of it

                    remaining_length = max_collection_size - positive_sentiment_collection.shape[0]

                    positive_sentiment_collection = torch.vstack((positive_sentiment_collection, positive_sentiment_momentum_embeddings[:remaining_length]))
                    positive_sentiment_collection_inputs["input_ids"] = torch.vstack((positive_sentiment_collection_inputs["input_ids"], positive_sentiment_inputs["input_ids"][:remaining_length]))
                    positive_sentiment_collection_inputs["attention_mask"] = torch.vstack((positive_sentiment_collection_inputs["attention_mask"], positive_sentiment_inputs["attention_mask"][:remaining_length]))
                    current_positive_sentiment_collection_index += min(num_positive_sentiment_samples, remaining_length) #make sure not to "overshoot" the max collection size
                else:
                    if current_positive_sentiment_collection_index >= max_collection_size:
                        current_positive_sentiment_collection_index = 0

                    start_index = current_positive_sentiment_collection_index

                    end_index = min(start_index + num_positive_sentiment_samples, max_collection_size)

                    positive_sentiment_collection[start_index:end_index] = positive_sentiment_momentum_embeddings[:(end_index-start_index)] #need to cut off any excess samples

                    positive_sentiment_collection_inputs["input_ids"][start_index:end_index] = positive_sentiment_inputs["input_ids"][:(end_index-start_index)]
                    positive_sentiment_collection_inputs["attention_mask"][start_index:end_index] = positive_sentiment_inputs["attention_mask"][:(end_index-start_index)]

                    current_positive_sentiment_collection_index += (end_index-start_index) #num_positive_sentiment_samples

                #do the same for the negative sentiment samples
                if negative_sentiment_collection.shape[0] < max_collection_size:

                    remaining_length = max_collection_size - negative_sentiment_collection.shape[0]

                    negative_sentiment_collection = torch.vstack((negative_sentiment_collection, negative_sentiment_momentum_embeddings[:remaining_length]))

                    negative_sentiment_collection_inputs["input_ids"] = torch.vstack((negative_sentiment_collection_inputs["input_ids"], negative_sentiment_inputs["input_ids"][:remaining_length]))
                    negative_sentiment_collection_inputs["attention_mask"] = torch.vstack((negative_sentiment_collection_inputs["attention_mask"], negative_sentiment_inputs["attention_mask"][:remaining_length]))

                    current_negative_sentiment_collection_index += min(num_negative_sentiment_samples, remaining_length)
                else:
                    if current_negative_sentiment_collection_index >= max_collection_size:
                        current_negative_sentiment_collection_index = 0

                    start_index = current_negative_sentiment_collection_index
                    end_index = min(start_index + num_negative_sentiment_samples, max_collection_size)

                    negative_sentiment_collection[start_index:end_index] = negative_sentiment_momentum_embeddings[:(end_index-start_index)] #need to cut off any excess samples

                    negative_sentiment_collection_inputs["input_ids"][start_index:end_index] = negative_sentiment_inputs["input_ids"][:(end_index-start_index)]
                    negative_sentiment_collection_inputs["attention_mask"][start_index:end_index] = negative_sentiment_inputs["attention_mask"][:(end_index-start_index)]

                    current_negative_sentiment_collection_index += (end_index-start_index)#num_negative_sentiment_samples

        ###############
        ### LOGGING ###
        ###############

        writer.add_scalar("Loss/training", loss.item(), iteration + epoch * len(train_loader))

        if iteration%25==0:
            print(f"* epoch {epoch} - iteration {iteration}, loss {loss.item():.6f}, progress: {100 * iteration / len(train_loader):.2f}%")

        if iteration%100==0:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(device)
            a = torch.cuda.memory_allocated(device)
            print(f"* memory reserved: {r/float(1000000):.0f} MB, memory allocated: {a/float(1000000):.0f} MB, free memory: {(t-r)/float(1000000):.0f}")

    lr_scheduler.step() #multiply learning rate by 0.9 after every epoch

    #######################
    ### VALIDATION LOOP ###
    #######################

    #validation loop: need to forward entire training and validation corpus to generate embeddings to compare to
    with torch.no_grad():
        model.eval()
        #first: generate embeddings of all training samples
        train_embeddings_matrix = torch.empty((0, 768), dtype=torch.float16).to(device)
        train_labels = torch.empty((0), dtype=int).to(device)
        for iteration, batch in enumerate(larger_batch_size_train_loader):
            inputs, labels = batch
            tokens = inputs["input_ids"].to(device)
            attention_masks = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            embeddings = model(input_ids=tokens, attention_mask=attention_masks)["pooler_output"]

            embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize

            train_embeddings_matrix = torch.vstack((train_embeddings_matrix, embeddings.type(torch.float16)))
            train_labels = torch.cat((train_labels, labels), dim=0)

            if iteration%25==0:
                print(f"* Forwarding all training samples: iteration {iteration}, progress: {100*iteration/len(larger_batch_size_train_loader):.2f}%")

        #second: predict labels based on the average proximity to elements in the training set classes; then compute accuracy using validation labels
        correct_counter = 0
        total_counter = 0
        loss_sum = 0
        for iteration, batch in enumerate(val_loader):
            inputs, labels = batch
            tokens = inputs["input_ids"].to(device)
            attention_masks = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            embeddings = model(input_ids=tokens, attention_mask=attention_masks)["pooler_output"]

            embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize

            similarity_matrix = torch.matmul(embeddings.type(torch.float16), train_embeddings_matrix.T)

            #highest_similarity = torch.argmax(similiarity_matrix, dim=1)

            positive_indices = (train_labels == 1)
            negative_indices = (train_labels == 0)

            average_positive_distance = similarity_matrix[:, positive_indices].mean(dim=1)
            average_negative_distance = similarity_matrix[:, negative_indices].mean(dim=1)

            predictions = (average_positive_distance > average_negative_distance).int()

            cross_entropy_labels = torch.zeros((val_batch_size, train_embeddings_matrix.shape[0]), dtype=torch.float32).to(device)
            for i in range(train_batch_size):
                if labels[i] == 1:
                    cross_entropy_labels[i] = positive_indices.int()
                else:
                    cross_entropy_labels[i] = negative_indices.int()

            loss = torch.nn.CrossEntropyLoss()(similarity_matrix/temperature, cross_entropy_labels)

            loss_sum+=loss.item()

            correct_counter += torch.sum(predictions==labels).item()
            total_counter += tokens.shape[0]
            if iteration%25==0:
                print(f"* Validation iteration {iteration}, progress: {100*iteration/len(val_loader):.2f}%")

        val_accuracy = 100*correct_counter/float(total_counter)

        print(f"Validation accuracy: {val_accuracy:.3f}%; Validation loss: {loss_sum:.3f}")
        writer.add_scalar("Validation/accuracy", 100*correct_counter/float(total_counter), epoch)
        writer.add_scalar("Validation/loss", loss_sum, epoch)

        if (loss_sum < lowest_val_loss): #do early stopping based on the validation loss
            num_epochs_not_improved = 0
            lowest_val_loss = loss_sum
            val_accuracy_of_best_model = val_accuracy
            print(f"Found new best model with {lowest_val_loss:.6f} validation loss ({val_accuracy_of_best_model:.3f}% validation accuracy). Saving model...")
            torch.save(model.state_dict(), "best_model_parameters.pt")
        else:
            num_epochs_not_improved += 1

        if num_epochs_not_improved>=early_stopping_threshold: #if we have not improved for early_stopping_threshold epochs, stop the training
            break


print(f"###### Finished training ######")

print(f"Loading the  best checkpoint which had {lowest_val_loss:.6f} validation loss and {val_accuracy_of_best_model:.3f}% validation accuracy...")
model.load_state_dict(torch.load("best_model_parameters.pt"))

###############
### TESTING ###
###############

test_data = read_text_data("../../twitter-datasets/test_data.txt")
test_dataset = TweetDataset(test_data, [-1]*len(test_data), tokenizer) #note: we don't use the labels here, we just need a dummy input
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, pin_memory=True, drop_last=False) #note: we don't want to "drop_last", so all batches must be the same size; since there's 10k samples, we can set batch_size to 10 to ensure all batches have full size

with torch.no_grad():
    model.eval()
    #first: generate embeddings of all training samples
    train_embeddings_matrix = torch.empty((0, 768), dtype=torch.float16).to(device)
    train_labels = torch.empty((0), dtype=int).to(device)
    for iteration, batch in enumerate(larger_batch_size_train_loader):
        inputs, labels = batch
        tokens = inputs["input_ids"].to(device)
        attention_masks = inputs["attention_mask"].to(device)
        labels = labels.to(device)

        embeddings = model(input_ids=tokens, attention_mask=attention_masks)["pooler_output"]

        embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize

        train_embeddings_matrix = torch.vstack((train_embeddings_matrix, embeddings.type(torch.float16)))
        train_labels = torch.cat((train_labels, labels), dim=0)

        if iteration%25==0:
            print(f"* Forwarding all training samples: iteration {iteration}, progress: {100*iteration/len(train_loader)}%")

    #second: predict labels based on the average proximity to elements in the training set classes
    test_predictions = torch.empty((0), dtype=int).to(device)
    for iteration, batch in enumerate(test_loader):
        inputs, labels = batch
        tokens = inputs["input_ids"].to(device)
        attention_masks = inputs["attention_mask"].to(device)

        embeddings = model(input_ids=tokens, attention_mask=attention_masks)["pooler_output"]

        embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize

        similarity_matrix = torch.matmul(embeddings.type(torch.float16), train_embeddings_matrix.T)

        #highest_similarity = torch.argmax(similiarity_matrix, dim=1)

        #current_predictions = train_labels[highest_similarity]

        positive_indices = (train_labels == 1)
        negative_indices = (train_labels == 0)

        average_positive_distance = similarity_matrix[:, positive_indices].mean(dim=1)
        average_negative_distance = similarity_matrix[:, negative_indices].mean(dim=1)

        current_predictions = (average_positive_distance > average_negative_distance).int()

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




