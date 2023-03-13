import pandas as pd
import numpy as np
import nltk
import csv
import math
import re
import matplotlib.pyplot as plt
import string
import numpy.linalg
nltk.download('punkt')



# Downloads file (url), not yet generalized
def download_file(url):
    file_list = []
    with open(url, 'r') as file:
        reader = csv.reader(file)

        # one word per row, so get every row
        for row in reader:
            # add row to stopwords
            file_list.append(row[0])

    return file_list

# __________________________________ Preprocessing Methods ___________________________________

# Cleans a single string
def remove_punc(s):
    s = str(s)
    # removes standard punc
    s = s.translate(str.maketrans('', '', string.punctuation))

    # catches extra punc not included in string.punctuation
    s = re.sub("'|’|–|—", "", s)
    s = re.sub('"', '', s)

    s = s.lower()
    return s


# Remove all stopwords from string
def remove_stop(s):
    sentence = s.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    words = nltk.word_tokenize(sentence)
    words = filter((lambda word: word not in stopwords), words)
    words = ' '.join(words)
    return words


# Gets just the root of the word
def get_stem(s):
    stemmer = nltk.SnowballStemmer('english')
    stems = [stemmer.stem(word) for word in s.split()]
    stem = ' '.join(stems)
    return stem


# _______________________________________ End Preprocessing Methods ___________________________


# ______________________________________ Start Naive Bayes _____________________________________

# Return count most common words
def get_most_common(data, count):
    words = [word for sent in data for word in sent.split() if word.isalpha()]
    freq = nltk.FreqDist(words)
    return dict(freq.most_common(count))


# Gets the combined frequency of every word in list1 and list2
def get_word_freq(all_list, list1, list2):
    all_freq = {}
    count1, count2 = 0, 0
    for word in all_list:
        freq1 = list1.get(word, 0) # Get word, if not there put 0
        count1 += freq1

        freq2 = list2.get(word, 0) # Get word, if not there put 0
        count2 += freq2

        all_freq[word] = [freq1, freq2]

    return all_freq, count1, count2


# Calculates the Phi value of a word
def calc_phi(obama_freq, trump_freq, obama_total, trump_total):
    a, b = obama_freq, trump_freq
    c, d = obama_total - a, trump_total - b
    return (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))


# Makes the DataFrame containing word, frequency, obama frequency, trump's frequency, and the Phi value
def make_comp_df(all_words, obama_dict, trump_dict):
    all_freq, obama_count, trump_count = get_word_freq(all_words, obama_dict, trump_dict)

    comparison_data = {'Word': [], 'Freq': [], 'Obama': [], 'Trump': [], 'Phi': []}

    for word in all_words:
        o_freq = obama_dict.get(word, 0)
        t_freq = trump_dict.get(word, 0)

        phi = calc_phi(o_freq, t_freq, obama_count, trump_count)
        a_freq = all_freq[word][0] + all_freq[word][1]

        comparison_data['Word'].append(word)
        comparison_data['Freq'].append(a_freq)
        comparison_data['Obama'].append(o_freq)
        comparison_data['Trump'].append(t_freq)
        comparison_data['Phi'].append(phi)

    comp_df = pd.DataFrame(comparison_data)
    return comp_df, obama_count, trump_count


# Performs the Naive Bayes algorithm
def naive_bayes(s, comp_df, obama_count, trump_count):
    sentence = s.split()

    # Calculate total frequencies
    total_freq = comp_df['Freq'].sum()
    # Calculate predicted probabilities for each speaker
    o_pred = obama_count / total_freq   # Probability of obama
    t_pred = trump_count / total_freq   # Probability of Trump

    # Calculate likelihood for each speaker
    o_prob, t_prob = 1.0, 1.0

    for word in sentence:
        if word in comp_df['Word'].values:
            o_word_count = comp_df.loc[comp_df['Word'] == word]['Obama'].values
            t_word_count = comp_df.loc[comp_df['Word'] == word]['Trump'].values
            o_prob *= (o_word_count + 1) / obama_count #Add 1 in case the count is 0
            t_prob *= (t_word_count + 1) / trump_count

    # Calculate probabilities for Obama and Trump

    o_prob = math.log(o_prob * o_pred) # also divides by (o_prob * o_pred + t_prob * t_pred) but that amounts to 1
    t_prob = math.log(t_prob * t_pred)
    return o_prob, t_prob


# Gets the naive bayes predictions for all values in X
# Public facing Naive Bayes method
def get_predictions(X, comp_df, obama_count, trump_count):
    preds = []
    diffs = []
    for sent in X:
        o_prob, t_prob = naive_bayes(sent, comp_df, obama_count, trump_count)
        pred = 1 if t_prob >= o_prob else 0
        diffs.append(t_prob - o_prob)
        preds.append(pred)

    return preds, diffs

# ________________________________________ End Naive Bayes ____________________________-


# _______________________________________ Start Utility Methods ________________________

# Helper method for the metrics method
def get_confusion_matrix(y, y_pred):
    unique_classes = set(y) | set(y_pred)
    n_classes = len(unique_classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    pred_pair = list(zip(y, y_pred))

    for i, j in pred_pair:
        matrix[int(i), int(j)] += 1

    return matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0]


# Calculates five metrics and stores in a dict: Accuracy, Sensitivity, Specificity, Precision, F1
def metrics(y, y_pred):
    n = len(y)

    tn, tp, fp, fn = get_confusion_matrix(y, y_pred)

    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)
    met = {'Accuracy': accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity, 'Precision': precision, 'F1': f1}
    return met


# Helper method for graphing a line
def graph_line(x, y, xlabel, ylabel, title, scatterplot=False, colors=None, marker=None):
    plt.figure(figsize=(8, 8))

    if scatterplot:
        plt.scatter(x, y, c=colors)
    else:
        plt.plot(x, y, marker=marker)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# ___________________________________ End Utility Methods __________________________


# ___________________________________ Start KNN Methods ____________________________


# Calculates cosine distance
def cosine_distance(v1, v2):
    magx = np.sqrt(np.dot(v1, v1))
    magy = np.sqrt(np.dot(v2, v2))
    ret = np.dot(v1, v2) / (magx * magy)
    if math.isnan(ret):
        print(v2)
        print(magx)
        print(magy)
        print(np.dot(v1,v2))
    return ret


# Calculates Euclidean Distance
def euclidean_distance(v1, v2):
    squared_distance = np.sum(np.square(v1 - v2))
    distance = math.sqrt(squared_distance)
    print(distance)
    return distance


# Performs the K-Nearest-Neighbor Algorithm
def knn(vector, vector_list, k, dist_func):
    dists = []
    for i in range(len(vector_list)):
        d = (i, dist_func(vector, vector_list[i]))
        dists.append(d)

    dists.sort(key=lambda x: x[1])
    idxs = [d[0] for d in dists]
    k_idxs = idxs[:k]
    count_o_idxs = len([idx for idx in k_idxs if idx < 500])

    # 0 = Obama,  1 = Trump
    res = 0 if count_o_idxs > k/2 else 1
    return res  # Returns label


# Helper method for making a vector for each sentence
# 'A vector for a sentence' is a list of 0's and 1's where
# each index represents a word, with 0 representing that word
# not in the sentence and 1 representing the word is in the sentence
def make_vectors(list_len, subset_words, og_list):
    vectors = []
    for sent in og_list:
        sent_vector = np.zeros(list_len)
        for word in sent.split():
            if word in subset_words:
                sent_vector[subset_words.index(word)] = 1

        vectors.append(sent_vector)

    return vectors


# Helper method for running KNN algorithm on a set of k_values (1 - 32, skipping every other)
# Returns accuracy at every k-value
def print_knn(vectors, print_graph):
    k_values = list(range(1, 32, 2))
    accs = []

    for k in k_values:
        label_list = []
        for idx, v in enumerate(vectors):
            label = knn(v, vectors[:idx] + vectors[idx + 1:], k, dist_func=euclidean_distance)
            label_list.append(label)

        y = [0 if i < 500 else 1 for i in range(1000)]  # 1 = Trump, 0 = Obama
        mets = metrics(y, label_list)
        if print_graph:
            print("K - Value: ", k)
            print("Metrics: ", mets)
        accs.append(mets['Accuracy'])

    if print_graph:
        xlabel = "K"
        ylabel = "Accuracy"
        title = "KNN Accuracy"
        graph_line(k_values, accs, xlabel=xlabel, ylabel=ylabel, title=title, marker='x')

    return accs


# Public facing method for running KNN Algorithm
# Returns max accuracy for KNN Algorithm
def test_knn_values(list_len, subset, og_list, print_graph=True, cross_validation=0):
    subset = subset.sample(frac=1)
    subset_words = subset['Word'].values.tolist()

    vectors = make_vectors(list_len, subset_words, og_list)
    accs = print_knn(vectors, print_graph)
    acc = max(accs)

    return acc
# ___________________________________ End KNN Methods ______________________________________


# ___________________________________ Start Accuracy Improvement Methods __________________


# Given parameters (At this point hard coded) graphs accuracy of KNN Algorithm over set of parameters
#        Hardcoded to only test accuracy vs vector length
def find_best_vector_length(comp_df):
    x_vect_len_acc = list(range(100, 1000, 50))
    y_vect_len_acc = []
    for vect_len in x_vect_len_acc:
        subset = pd.concat([comp_df.head(vect_len//2), comp_df.tail(vect_len//2)])
        max_acc = test_knn_values(vect_len, subset, og_list, print_graph=False)
        print("Maximum Accuracy at " + str(vect_len) + ": ", max_acc)
        y_vect_len_acc.append(max_acc)

    graph_line(x_vect_len_acc, y_vect_len_acc, xlabel="Vector Length", ylabel="Max Accuracy", title="Accuracy For Vector Length", marker='o')

# ___________________________________ End Methods ____________________________________-

# Main:


# Download necessary info
df = pd.read_csv('/Users/mihir/DS4400 - Spring ML 1/Homework 3/statements.csv')
stopwords = download_file('/Users/mihir/DS4400 - Spring ML 1/Homework 3/NLTK_English_stopwords.txt')

# Clean dataframe
df = df.applymap(remove_punc)
df = df.applymap(remove_stop)
df = df.applymap(get_stem)

og_list = df['text'].values

# Get the most common 1000 words from obama and trump
obama_txt = df[df['speaker'] == 'obama']['text']
trump_txt = df[df['speaker'] == 'trump']['text']
obama_dict = get_most_common(obama_txt, 1000)
trump_dict = get_most_common(trump_txt, 1000)

# Create a dataframe for trump and obama combined
obama_words = set(obama_dict.keys())
trump_words = set(trump_dict.keys())

all_words = obama_words.union(trump_words)

# Phi count positive = more like Obama
comp_df, obama_count, trump_count = make_comp_df(all_words, obama_dict, trump_dict)
comp_df = comp_df.sort_values(by='Freq', ascending=False)
print(comp_df)
comp_df = comp_df.sort_values(by='Phi', ascending=False)
print(comp_df)
print(comp_df.shape)

trump_10 = comp_df[comp_df['Trump'] > 0]['Word'].tail(10).values
obama_10 = comp_df[comp_df['Obama'] > 0]['Word'].head(10).values
print("10 Words Most Commonly Associated with Trump: ", trump_10)
print("10 Words Most Commonly Associated with Obama: ", obama_10)


# Naive Bayes
y_pred, diffs = get_predictions(og_list, comp_df, obama_count, trump_count) # Get the naive bayes labl predictions
y = [0 if i < 500 else 1 for i in range(1000)] # 1 = Trump, 0 = Obama # Create actual labels for data
mets = metrics(y, y_pred) # Calculate the metrics for the Naive bayes Model
print(mets)

# Graph Naive Bayes Scatterplot
colors = ['blue' if i < 500 else 'red' for i in range(len(og_list))]
xlabel = 'Sentence Number (Obama 0 - 499,  Trump 500 - 1000)'
ylabel = 'NBScore (Trump) - NBScore (Obama)'
graph_line(list(range(len(og_list))), diffs, xlabel=xlabel, ylabel=ylabel, title='Score Difference', scatterplot=True, colors=colors)


# KNN Algorithm
vect_len = 850 # Hardcoded at this point but calculated using the Accuracy Improvement Method above
comp_df = comp_df.sort_values(by='Phi', ascending=False)
subset = pd.concat([comp_df.head(vect_len//2), comp_df.tail(vect_len//2)])
test_knn_values(vect_len, subset, og_list, print_graph=True)
#find_best_vector_length(comp_df)


