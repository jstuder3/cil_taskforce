import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

from dataset import TweetDataset

def read_text_data(infile):
    out_file = []
    with open(infile, encoding="utf-8") as f:
        for line in f:
            out_file+=[line]
    return out_file


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "bert-base-uncased"

#step 1: load an untrained model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


#step 2: load and tokenize data

print(f"Running on {device}")
print(f"GPU name: {torch.cuda.get_device_name(device)}")
print(f"Device properties: {torch.cuda.get_device_properties(device)}")

pos_data = read_text_data("../../twitter-datasets/train_pos.txt")
neg_data = read_text_data("../../twitter-datasets/train_neg.txt")

pos_data_train, _ = train_test_split(pos_data[0:int(len(pos_data))], train_size=0.05)
neg_data_train, _ = train_test_split(neg_data[0:int(len(neg_data))], train_size=0.05)

full_train_dataset = pos_data_train + neg_data_train
train_labels = [1] * len(pos_data_train) + [0] * len(neg_data_train)


train_data = TweetDataset(full_train_dataset, train_labels, tokenizer)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)

#step 3: produce untrained embeddings

untrained_embeddings_matrix = torch.empty((0, 768), dtype=torch.float32).to(device)
untrained_labels = []

print("* Forwarding samples through untrained encoder...")

with torch.no_grad():
    for iteration, batch in enumerate(train_loader):
        inputs, labels = batch
        tokens = inputs["input_ids"].to(device)
        attention_masks = inputs["attention_mask"].to(device)

        untrained_labels += labels.tolist()

        embeddings = model(input_ids=tokens, attention_mask=attention_masks)["pooler_output"]
        embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize

        untrained_embeddings_matrix = torch.vstack((untrained_embeddings_matrix, embeddings))

untrained_embeddings_matrix = untrained_embeddings_matrix.cpu().numpy()
untrained_labels = np.array(untrained_labels)

#step 4: load a trained model

model.load_state_dict(torch.load("best_model_parameters.pt"))

#step 5: produce trained embeddings

trained_embeddings_matrix = torch.empty((0, 768), dtype=torch.float32).to(device)
trained_labels = []

print("* Forwarding samples through trained encoder...")

with torch.no_grad():
    for iteration, batch in enumerate(train_loader):
        inputs, labels = batch
        tokens = inputs["input_ids"].to(device)
        attention_masks = inputs["attention_mask"].to(device)

        trained_labels += labels.tolist()

        embeddings = model(input_ids=tokens, attention_mask=attention_masks)["pooler_output"]
        embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize

        trained_embeddings_matrix = torch.vstack((trained_embeddings_matrix, embeddings))

trained_embeddings_matrix = trained_embeddings_matrix.cpu().numpy()
trained_labels = np.array(trained_labels)

#step 6: visualize untrained and trained embeddings by downprojecting them using PCA and t-SNE

pca = PCA(n_components=50)
untrained_pca_projection = pca.fit_transform(untrained_embeddings_matrix)
trained_pca_projection = pca.fit_transform(trained_embeddings_matrix)

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
untrained_downprojection = tsne.fit_transform(untrained_pca_projection)
trained_downprojection = tsne.fit_transform(trained_pca_projection)

markersize=50
fontsize=30
max_num_samples=1000

plt.subplot(1, 2, 1)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.gca().set_title("Pre-training embeddings", fontsize=fontsize)

#plt.scatter(untrained_downprojection[:, 0], untrained_downprojection[:, 1], color="red")

plt.scatter(untrained_downprojection[untrained_labels==0][:, 0][:max_num_samples], untrained_downprojection[untrained_labels==0][:, 1][:max_num_samples], color="indianred", s=markersize, marker="x", zorder=2)
plt.scatter(untrained_downprojection[untrained_labels==1][:, 0][:max_num_samples], untrained_downprojection[untrained_labels==1][:, 1][:max_num_samples], color="limegreen", s=markersize, marker="o", zorder=1)

plt.subplot(1, 2, 2)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.gca().set_title("Post-training embeddings", fontsize=fontsize)

plt.scatter(trained_downprojection[trained_labels==0][:, 0][:max_num_samples], trained_downprojection[trained_labels==0][:, 1][:max_num_samples], color="indianred", s=markersize, marker="x", zorder=2)
plt.scatter(trained_downprojection[trained_labels==1][:, 0][:max_num_samples], trained_downprojection[trained_labels==1][:, 1][:max_num_samples], color="limegreen", s=markersize, marker="o", zorder=1)

plt.show()
