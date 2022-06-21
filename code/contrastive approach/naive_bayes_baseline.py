import pandas as pd
import torch
import numpy as np
import re
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from dataset import remove_multi_spaces, replace_usr_and_url
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download("wordnet")
nltk.download('omw-1.4')

###################################
########Hyperparameters############
###################################
eval_model = "Bernoulli"
train_size = 0.8
num_features = 10000
k_fold = 10
#########Preprocessing#############
stopwords = False
stemming = False
lemmatizing = False
###################################
###################################
###################################

dataset_dir = str(pathlib.Path(__file__).parent.parent.parent.resolve())

def read_text_data(infile):
    out_file = []
    with open(infile, encoding="utf-8") as f:
        for line in f:
            out_file += [line]
    return out_file

def preprocessing(data,use_stop,use_stem,use_lemma):
    
    tokenizer = nltk.word_tokenize
    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()

    #Remove duplicate characters *** whhhyyyyy -> why
    data = list(map(lambda x: re.sub(r'(.)\1+', r'\1', x),data))
    #Remove numbers
    data = list(map(lambda x: re.sub('[0-9]+','',x),data))
    data = replace_usr_and_url(data)
    data = remove_multi_spaces(data)

    data_tokens = list(map(tokenizer,data))
    if use_stop:
        stopwords = nltk.corpus.stopwords.words("english")
        data_tokens = list(map(lambda x: [word for word in x if word.lower() not in stopwords and word.isalpha()],data_tokens))
    else:
        data_tokens = list(map(lambda x: [word.lower() for word in x if word.isalpha()],data_tokens))

    if use_stem:
        data_tokens = [list(map(lambda x: stemmer.stem(x), sentence)) for sentence in data_tokens]

    if use_lemma:
        data_tokens = [list(map(lambda x: lemmatizer.lemmatize(x), sentence)) for sentence in data_tokens]

    return data_tokens

def evaluate(model,train_features, train_labels, test_features, test_labels):
    model.fit(train_features,train_labels)
    predictions = model.predict(test_features)
    store_filename = "model_metrics/" + eval_model + "_" + ("stop_" if stopwords else "") + \
        ("stem_" if stemming else "") + ("lemma_" if lemmatizing else "")

    mean_accuracy = model.score(test_features, test_labels)
    print("Mean Accuracy: {0:.2%}".format(mean_accuracy))
    report = classification_report(test_labels,predictions,output_dict=True)
    df = pd.DataFrame.from_dict(report).transpose()
    report_filename = store_filename + "report.csv"
    df.to_csv(report_filename)
    print("Classification report stored in {0}".format(report_filename))
    
    #For printing purposes
    report = classification_report(test_labels,predictions)
    print(report)
    
    cf_matrix = confusion_matrix(test_labels,predictions)
    categories = ["Negative", "Positive"]
    cf_groups = ["TN", "FP", "FN", "TP"]
    cf_values = ['{0:.2%}'.format(val) for val in cf_matrix.flatten()/np.sum(cf_matrix)]
    plot_labels = [f'{v1} {v2}' for v1, v2 in zip(cf_groups,cf_values)]
    plot_labels = np.asarray(plot_labels).reshape(2,2)
    ax= plt.subplot()
    sns.heatmap(cf_matrix,annot=plot_labels, cmap="Blues", fmt="", xticklabels=categories, yticklabels = categories, ax=ax)
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cf_filename = store_filename + "img.png"
    plt.savefig(cf_filename)
    print("Confusion Matrix stored in {0}".format(cf_filename))


device = "cuda" if torch.cuda.is_available() else "cpu"

pos_data = read_text_data(dataset_dir + "/twitter-datasets/train_pos.txt")
neg_data = read_text_data(dataset_dir + "/twitter-datasets/train_neg.txt")

pos_data_train, pos_data_val = train_test_split(pos_data[0:int(len(pos_data))], train_size=train_size)
neg_data_train, neg_data_val = train_test_split(neg_data[0:int(len(neg_data))], train_size=train_size)

full_train_dataset = pos_data_train + neg_data_train
train_labels = [1] * len(pos_data_train) + [0] * len(neg_data_train)

full_val_dataset = pos_data_val + neg_data_val
val_labels = [1] * len(pos_data_val) + [0] * len(neg_data_val)

full_train_tokens = preprocessing(full_train_dataset,stopwords,stemming,lemmatizing)
full_val_tokens = preprocessing(full_val_dataset,stopwords,stemming,lemmatizing)

full_train_tokens = list(map(lambda x: ' '.join(x), full_train_tokens))
full_val_tokens = list(map(lambda x: ' '.join(x), full_val_tokens))

feature_extractor = TfidfVectorizer(max_features=num_features)
train_features = feature_extractor.fit_transform(full_train_tokens)
val_features = feature_extractor.transform(full_val_tokens)
#print(feature_extractor.get_feature_names())

if(eval_model == "LogReg"):
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
elif(eval_model == "LogRegCV"):
    model = LogisticRegressionCV(cv=k_fold, max_iter=10000, n_jobs=-1)
elif(eval_model == "Bernoulli"):
    model = BernoulliNB()
elif(eval_model == "LinearSVC"):
    model = LinearSVC(max_iter=1000)
else:
    print("Model not specified or does not exist. Using Logistic Regression")
    eval_model = "LogReg"
    model = LogisticRegression(max_iter=1000, n_jobs=-1)

evaluate(model,train_features,train_labels,val_features,val_labels)
