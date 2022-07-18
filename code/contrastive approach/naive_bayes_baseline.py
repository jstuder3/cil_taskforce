import pandas as pd
import numpy as np
import re
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from dataset import remove_multi_spaces, replace_usr_and_url
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download("wordnet")
# nltk.download('omw-1.4')

###################################
########Hyperparameters############
###################################
eval_model = "LogReg"
train_size = 0.8
num_features = 10000
max_ngrams = 1
get_best = False
k_fold = 10
#########Preprocessing#############
stopwords = False
stemming = False
lemmatizing = False
###Metric to evaluate Best Model###
eval_mode = "avg"  # Options = ["avg","max"]
eval_metric = "accuracy"  # Options = ["accuracy","precision","recall"]
###################################
###################################
###################################

all_models = {
    "BernoulliNB": BernoulliNB(),
    # Requires input to be dense => Large memory footprint + Computation time
    # "GaussianNB": GaussianNB(),
    "LinearSVC": LinearSVC(max_iter=1000),
    "LogReg": LogisticRegression(max_iter=1000, n_jobs=-1),
}

dataset_dir = str(pathlib.Path(__file__).parent.parent.parent.resolve())


def read_text_data(infile):
    out_file = []
    with open(infile, encoding="utf-8") as f:
        for line in f:
            out_file += [line]
    return out_file


def preprocessing(data, use_stop, use_stem, use_lemma):

    tokenizer = nltk.word_tokenize
    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()

    # Remove duplicate characters *** whhhyyyyy -> why
    data = list(map(lambda x: re.sub(r'(.)\1+', r'\1', x), data))
    # Remove numbers
    data = list(map(lambda x: re.sub('[0-9]+', '', x), data))
    data = replace_usr_and_url(data)
    data = remove_multi_spaces(data)

    data_tokens = list(map(tokenizer, data))
    if use_stop:
        stopwords = nltk.corpus.stopwords.words("english")
        data_tokens = list(map(lambda x: [word for word in x if word.lower(
        ) not in stopwords and word.isalpha()], data_tokens))
    else:
        data_tokens = list(
            map(lambda x: [word.lower() for word in x if word.isalpha()], data_tokens))

    if use_stem:
        data_tokens = [list(map(lambda x: stemmer.stem(x), sentence))
                       for sentence in data_tokens]

    if use_lemma:
        data_tokens = [list(map(lambda x: lemmatizer.lemmatize(x), sentence))
                       for sentence in data_tokens]

    return data_tokens


def evaluate(model_name, model, train_features, train_labels, test_features, test_labels, out_results=True):
    if model_name == "GaussianNB":
        model.fit(train_features.todense(), train_labels)
        test_features = test_features.todense()
    else:
        model.fit(train_features, train_labels)

    predictions = model.predict(test_features)
    store_filename = "model_metrics/" + eval_model + "_" + ("stop_" if stopwords else "") + \
        ("stem_" if stemming else "") + ("lemma_" if lemmatizing else "")

    mean_accuracy = model.score(test_features, test_labels)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    print("Mean Accuracy: {0:.2%} \t Precision: {1:.2%} \t Recall: {2:.2%}".format(
        mean_accuracy, precision, recall))

    if out_results:
        # For printing purposes
        report = classification_report(test_labels, predictions)
        print(report)

        report = classification_report(
            test_labels, predictions, output_dict=True)
        df = pd.DataFrame.from_dict(report).transpose()
        report_filename = store_filename + "report.csv"
        df.to_csv(report_filename)
        print("Classification report stored in {0}".format(report_filename))

        cf_matrix = confusion_matrix(test_labels, predictions)
        categories = ["Negative", "Positive"]
        cf_groups = ["TN", "FP", "FN", "TP"]
        cf_values = ['{0:.2%}'.format(val)
                     for val in cf_matrix.flatten()/np.sum(cf_matrix)]
        plot_labels = [f'{v1} {v2}' for v1, v2 in zip(cf_groups, cf_values)]
        plot_labels = np.asarray(plot_labels).reshape(2, 2)
        ax = plt.subplot()
        sns.heatmap(cf_matrix, annot=plot_labels, cmap="Blues", fmt="",
                    xticklabels=categories, yticklabels=categories, ax=ax)
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cf_filename = store_filename + "img.png"
        plt.savefig(cf_filename)
        print("Confusion Matrix stored in {0}".format(cf_filename))

    return mean_accuracy, precision, recall

# Performs cross validation to determine the best classifier given the hyperparameters (Model Selection)


def get_best_model(all_models, features, labels, cv_splits, mode, metric):

    results = {k: [] for k in all_models.keys()}
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42069)
    labels = np.array(labels)
    current_split = 1
    for train_idx, val_idx in skf.split(features, labels):
        for model_name in list(all_models.keys()):
            model = all_models[model_name]
            train_features, val_features = features[train_idx], features[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            print("Split {0} - {1}".format(current_split, model_name))
            mean_accuracy, precision, recall = evaluate(
                model_name, model, train_features, train_labels, val_features, val_labels, out_results=False)
            results[model_name] += [mean_accuracy, precision, recall]

        current_split += 1

    avg_metrics, max_metrics = {}, {}

    for model_name in list(results.keys()):
        model_scores = results[model_name]
        avg_accuracy, max_accuracy = np.average(
            model_scores[0::3]), np.max(model_scores[0::3])
        avg_precision, max_precision = np.average(
            model_scores[1::3]), np.max(model_scores[1::3])
        avg_recall, max_recall = np.average(
            model_scores[2::3]), np.max(model_scores[2::3])

        avg_metrics[model_name] = [avg_accuracy, avg_precision, avg_recall]
        max_metrics[model_name] = [max_accuracy, max_precision, max_recall]

    best_avg_metric = [x for xs in list(avg_metrics.values()) for x in xs]
    print(best_avg_metric)
    best_max_metric = [x for xs in list(max_metrics.values()) for x in xs]
    best_model_score = 0
    if mode == "avg":
        if metric == "accuracy":
            best_model = int(np.argmax(best_avg_metric[0::3]))
            best_model_score = np.max(best_avg_metric[0::3])
        elif metric == "precision":
            best_model = int(np.argmax(best_avg_metric[1::3]))
            best_model_score = np.max(best_avg_metric[1::3])
        elif metric == "recall":
            best_model = int(np.argmax(best_avg_metric[2::3]))
            best_model_score = np.max(best_avg_metric[2::3])
        else:
            print("Unknown metric! Using average accuracy to determine best model.")
            best_model = int(np.argmax(best_avg_metric[0::3]))
            best_model_score = np.max(best_avg_metric[1::3])

    elif mode == "max":
        if metric == "accuracy":
            best_model = int(np.argmax(best_max_metric[0::3]))
            best_model_score = np.max(best_max_metric[0::3])
        elif metric == "precision":
            best_model = int(np.argmax(best_max_metric[1::3]))
            best_model_score = np.max(best_max_metric[1::3])
        elif metric == "recall":
            best_model = int(np.argmax(best_max_metric[2::3]))
            best_model_score = np.max(best_max_metric[2::3])
        else:
            print("Unknown metric! Using max accuracy to determine best model.")
            best_model = int(np.argmax(best_max_metric[0::3]))
            best_model_score = np.max(best_max_metric[0::3])

    else:
        print("Unknown mode! Using average accuracy to determine best model.")
        best_model = int(np.argmax(best_avg_metric[0::3]))

    best_model_name = list(all_models.keys())[best_model]
    print("Best model ({0} {1}): {2} ({3:.2%})".format(
        eval_mode, eval_metric, best_model_name, best_model_score))

    return best_model_name, best_model_score


if __name__ == "__main__":

    pos_data = read_text_data(dataset_dir + "/twitter-datasets/train_pos.txt")
    neg_data = read_text_data(dataset_dir + "/twitter-datasets/train_neg.txt")

    # Use full dataset to determine the best classifier model
    if get_best:
        full_train_dataset = pos_data + neg_data
        train_labels = [1] * len(pos_data) + [0] * len(neg_data)

        full_train_tokens = preprocessing(
            full_train_dataset, stopwords, stemming, lemmatizing)
        full_train_tokens = list(map(lambda x: ' '.join(x), full_train_tokens))

        feature_extractor = TfidfVectorizer(
            max_features=num_features, ngram_range=(1, max_ngrams))
        train_features = feature_extractor.fit_transform(full_train_tokens)

        best_model_name, best_model_score = get_best_model(
            all_models, train_features, train_labels, k_fold, eval_mode, eval_metric)

    # Evaluate the performance of a model given the hyperparameters
    else:
        pos_data_train, pos_data_val = train_test_split(
            pos_data[0:int(len(pos_data))], train_size=train_size)
        neg_data_train, neg_data_val = train_test_split(
            neg_data[0:int(len(neg_data))], train_size=train_size)

        full_train_dataset = pos_data_train + neg_data_train
        train_labels = [1] * len(pos_data_train) + [0] * len(neg_data_train)

        full_val_dataset = pos_data_val + neg_data_val
        val_labels = [1] * len(pos_data_val) + [0] * len(neg_data_val)

        full_train_tokens = preprocessing(
            full_train_dataset, stopwords, stemming, lemmatizing)
        full_val_tokens = preprocessing(
            full_val_dataset, stopwords, stemming, lemmatizing)

        full_train_tokens = list(map(lambda x: ' '.join(x), full_train_tokens))
        full_val_tokens = list(map(lambda x: ' '.join(x), full_val_tokens))

        feature_extractor = TfidfVectorizer(
            max_features=num_features, ngram_range=(1, max_ngrams))
        train_features = feature_extractor.fit_transform(full_train_tokens)
        val_features = feature_extractor.transform(full_val_tokens)
        # print(feature_extractor.get_feature_names())

        model = all_models[eval_model]
        evaluate(eval_model, model, train_features, train_labels, val_features, val_labels)
