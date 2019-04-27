import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.cluster import KMeansClusterer, cosine_distance
import json
import numpy as np

# Define Clustering fuction.
def cluster_kmean(train_file, test_file):
    
    # Import training and testing dataset from json file.
    train_data=json.load(open(train_file,'r'))
    test_data=json.load(open(test_file,'r'))
    
    # Define text and label variables.
    text,label=zip(*test_data)
    test_text=list(text)
    test_label=list(label)
    
    # Select only first class of the dataset as the goal is to one layer classification.
    test_label = [ i[0] for i in test_label]
    
    # Call Tfidf from Sklearn with parameters.
    tfidf_vect = TfidfVectorizer(stop_words="english", min_df = 5) 
    
    # Train model with train dataset.
    dtm= tfidf_vect.fit_transform(train_data)
    
    # define number of clusters
    num_clusters=3
    
    # define K-mean parameters.
    clusterer = KMeansClusterer(num_clusters, cosine_distance, repeats=15)
    
    # implement topic classifications.
    clusterer.cluster(dtm.toarray(), assign_clusters=True)

    # Transform test dataset
    test_dtm = tfidf_vect.transform(test_text)
    
    # Classify test dataset
    predicted = [clusterer.classify(v) for v in test_dtm.toarray()]
    
    # Define a dataframe of test labels and predicted clusters.
    confusion_df = pd.DataFrame(list(zip(test_label, predicted)),\
                            columns = ["label", "cluster"])
                            
     # print Crosstab to see the correctly predicted number of each cluster/                
    print (pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label))
    
    cluster_dict = {}
    
    # store array of column names.
    get_col = pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label).columns.values
    
    # store values in array from created crosstab
    get_cluster_dict  = pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label).values
    
    # get dictionary of cluster number and cluster name (eg. {0 : 'Travel & Transportation'....}) 
    for idx , arr in enumerate(get_cluster_dict) :
        cluster_dict[idx] = get_col[arr.argsort()[::-1][0]]
        print ('Cluster {} : Topic {}'.format(idx, cluster_dict[idx]))
    
    # get list of predicted cluster
    predicted_target=[cluster_dict[i] for i in predicted]
    
    # report performance
    print(metrics.classification_report(test_label, predicted_target))
    

cluster_kmean('train_text.json', 'test_text.json')


    
