import time
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import seaborn as sns
import pickle
from prettytable import PrettyTable


class ClustersModels:
    @staticmethod
    def trainKmeans(data, n_clusters):
        k = MiniBatchKMeans(n_clusters=n_clusters, random_state=728)
        start = time.time()
        k.fit(data)
        end = time.time()
        print("##### Training took {} seconds".format(round(end - start, 3)))
        return k

    @staticmethod
    def trainAgg(data, n_clusters):
        g = AgglomerativeClustering(n_clusters=n_clusters)
        start = time.time()
        print("fitting")
        g.fit(data)
        end = time.time()
        print("##### Training took {} seconds".format(round(end - start, 3)))
        return g

    @staticmethod
    def clusterLabelCount(clusters, labels):
        count = {}
        unique_clusters = list(set(clusters))
        unique_labels = list(set(labels))
        for cluster in unique_clusters:
            count[cluster] = {}
            for label in unique_labels:
                count[cluster][label] = 0
        for i in range(len(clusters)):
            count[clusters[i]][labels[i]] += 1
        cluster_df = pd.DataFrame(count)
        return cluster_df

    @staticmethod
    def printScores(true, pred):
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average="macro")
        rec = recall_score(true, pred, average='weighted')
        return "\n\tF measure: {0:0.8f} | Accuracy: {0:0.8f} | Recall: {0:0.8f}".format(f1, acc, rec)

    @staticmethod
    def showConfusionMatrix(true, pred):
        labels = ['boat', 'car', 'chandelier', 'motorcycle', 'piano', 'plane', 'tiger', 'tree', 'turtle', 'watch']
        confusionmatrix = confusion_matrix(true, pred)
        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.13, right=1, bottom=0.2, top=0.9, wspace=0.2, hspace=0.2)

        sns.heatmap(confusionmatrix, annot=True, fmt="d", annot_kws={"size": 16}, xticklabels=labels,
                    yticklabels=labels)
        plt.title("Confusion matrix", fontsize=20)
        plt.ylabel('True label', fontsize=16)
        plt.xlabel('Clustering label', fontsize=16)
        print("Close the confusion matrix figure to go ... ")
        plt.show()

    def trainModels(self, dir, features, features_pca, data):
        print("# Training Models")
        print("## VGG16")
        [_, y_train] = data

        print("## KMeans:")
        Kmeans = self.trainKmeans(features, 10)
        pickle.dump(Kmeans, open((dir + "/CL_kmeans.sav"), 'wb'))
        print('## Kmeans PCA:')
        Kmeans_pca = self.trainKmeans(features_pca, 10)
        pickle.dump(Kmeans_pca, open((dir + "/CL_kmeans_pca.sav"), 'wb'))
        print("## Agg: ")
        Agg = self.trainAgg(features, 10)
        pickle.dump(Agg, open((dir + "/CL_agg.sav"), 'wb'))
        print("## Agg PCA:")
        Agg_pca = self.trainAgg(features_pca, 10)
        pickle.dump(Agg_pca, open((dir + "/CL_agg_pca.sav"), 'wb'))
        print("#### Models saved")

        Kmeans_pred = Kmeans.predict(features)
        Kmeans_pred_cnt = self.clusterLabelCount(Kmeans_pred, y_train)
        Kmeans_pca_pred = Kmeans_pca.predict(features_pca)
        Kmeans_pca_pred_cnt = self.clusterLabelCount(Kmeans_pca_pred, y_train)
        Agg_pred = Agg.fit_predict(features)
        Agg_pred_cnt = self.clusterLabelCount(Agg_pred, y_train)
        Agg_pca_pred = Agg_pca.fit_predict(features_pca)
        Agg_pca_pred_cnt = self.clusterLabelCount(Agg_pca_pred, y_train)

        print("### New models characteristics")
        print("### KMeans: ")
        print(Kmeans_pred_cnt)
        print("### KMeans PCA: ")
        print(Kmeans_pca_pred_cnt)
        print("Agg ")
        print(Agg_pred_cnt)
        print("Agg PCA ")
        print(Agg_pca_pred_cnt)

    @staticmethod
    def predictNewData(models_dir, features, filenames):
        Kmeans = pickle.load(open((models_dir + "/CL_kmeans.sav"), 'rb'))
        Kmeans_pred = Kmeans.predict(features)


        t = PrettyTable()
        t.field_names = ['File', 'Prediction (Kmeans)']
        Kmenas_codes = ["tiger", "watch", "motorcycle", "car", "tree", "plane", "piano", "chandelier", "turtle", "boat"]

        for i in range(len(Kmeans_pred)):
            t.add_row([filenames[i], Kmenas_codes[Kmeans_pred[i]] ])

        print("\n\n+++ CLASSIFICATION RESULT +++")
        print(t)

    # # Used only to evaluate my models
    # def predict(self, models_dir, features, features_pca, y_test):
    #
    #     Kmeans = pickle.load(open((models_dir + "/CL_model.sav"), 'rb'))
    #     Kmeans_pca = pickle.load(open((models_dir + "/CL_kmeans_pca.sav"), 'rb'))
    #     Agg = pickle.load(open((models_dir + "/CL_agg.sav"), 'rb'))
    #     Agg_pca = pickle.load(open((models_dir + "/CL_agg_pca.sav"), 'rb'))
    #
    #     Kmeans_pred = Kmeans.predict(features)
    #     Kmeans_pred_cnt = self.clusterLabelCount(Kmeans_pred, y_test)
    #     Kmeans_pca_pred = Kmeans_pca.predict(features_pca)
    #     Kmeans_pca_pred_cnt = self.clusterLabelCount(Kmeans_pca_pred, y_test)
    #     Agg_pred = Agg.predict(features)
    #     Agg_pred_cnt = self.clusterLabelCount(Agg_pred, y_test)
    #     Agg_pca_pred = Agg_pca.predict(features_pca)
    #     Agg_pca_pred_cnt = self.clusterLabelCount(Agg_pca_pred, y_test)
    #
    #     Kmenas_codes = ["tiger", "watch", "motorcycle", "car", "tree", "plane", "piano", "chandelier", "turtle", "boat"]
    #     Kmenas_pca_codes = ["tiger", "watch", "motorcycle", "car", "tree", "plane", "piano", "chandelier", "turtle", "boat"]
    #     Agg_codes = ["plane", "turtle", "motorcycle", "piano", "tree", "chandelier", "boat", "tiger", "watch", "car"]
    #     Agg_pca_codes = ["plane", "turtle", "motorcycle", "piano", "tree", "chandelier", "boat", "tiger", "watch", "car"]
    #
    #     Kmeans_pred_codes = [Kmenas_codes[x] for x in Kmeans_pred]
    #     Agg_pred_codes = [Agg_codes[x] for x in Agg_pred]
    #     Kmeans_pca_pred_codes = [Kmenas_pca_codes[x] for x in Kmeans_pca_pred]
    #     Agg_pca_pred_codes = [Agg_pca_codes[x] for x in Agg_pca_pred]
    #
    #     print("### Results")
    #     print("### KMeans: ")
    #     print(Kmeans_pred_cnt)
    #     print(self.printScores(y_test, Kmeans_pred_codes))
    #     self.showConfusionMatrix(y_test, Kmeans_pred_codes)
    #     print("### KMeans PCA: ")
    #     print(Kmeans_pca_pred_cnt)
    #     print(self.printScores(y_test, Kmeans_pca_pred_codes))
    #     self.showConfusionMatrix(y_test, Kmeans_pca_pred_codes)
    #     print("Agg ")
    #     print(Agg_pred_cnt)
    #     print(self.printScores(y_test, Agg_pred_codes))
    #     self.showConfusionMatrix(y_test, Agg_pred_codes)
    #     print("Agg PCA ")
    #     print(Agg_pca_pred_cnt)
    #     print(self.printScores(y_test, Agg_pca_pred_codes))
    #     self.showConfusionMatrix(y_test, Agg_pca_pred_codes)