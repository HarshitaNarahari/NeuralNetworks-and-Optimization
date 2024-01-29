import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

#ssl error - fixed using chat gpt
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#################################################################
# Load Dataset
#################################################################

dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    shuffle=True,
    random_state=42,
)

labels = dataset.target
unique_labels, category_sizes = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]

#################################################################
# Evaluate Fitness
#################################################################
def fit_and_evaluate(km, X, n_runs=5):

    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        km.fit(X)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")

#################################################################
# Vectorize 
#################################################################
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)

X_tfidf = vectorizer.fit_transform(dataset.data)

#################################################################
# (TODO): Implement K-Means  
#################################################################


# dont use scikit learns
# implement k means clustering from scratch
clusters = 8
max_iterations = 100
rand_state = np.random.seed(50)

class KMeans: #given
    def __init__(self, X): #given
        # TODO
        #assigns the variables to the specified values from self variable
        self.n_clusters = clusters
        self.max_iters = max_iterations
        self.rand_state = rand_state
        self.centroids = None
        self.labels_ = np.zeros(X.shape[0])
        #for loop variable
        self.iterations = 100

    def fit(self, X): #given 
        # TODO
        # seed is used to reproduce random values for the state
        np.random.seed(self.rand_state)

        # Randomly initializes centroids with the values given
        self.centroids = X[np.random.choice(len(X), self.n_clusters, replace=False).toarray()]

        for self.iterations in range(self.max_iters):
            # Assigns data points to centroids - derived from towards data science article on kmeans
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids))

            # Updates the centroids points based on average points for each cluster
            for self.j in range(self.n_clusters):
                new_centroids = np.array[X[labels == self.j].mean()]

            # re-initializes the current centroid values to the new centroid values for the next iteration
            self.centroids = new_centroids

        # assigns labels value to self from fit and evaluate function
        self.labels_ = labels
        return labels

    def set_params(self, random_state): # given TODO
        # sets the parameters based on the new random state
        self.rand_state = random_state
 

#TODO:
kmeans = KMeans(X_tfidf) #given
# Feel free to change the number of runs
fit_and_evaluate(kmeans, X_tfidf) #given
