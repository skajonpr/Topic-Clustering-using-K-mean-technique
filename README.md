# Topic-Clustering-using-K-mean-technique
This project is to classify news topics using K-mean clustering technique.

Two json files are imported to test and train the model.

File example:

<img width="500" alt="train and test docs exam" src="https://user-images.githubusercontent.com/45326221/56854811-8563a400-690a-11e9-9720-2b1b90ae3925.PNG">


**Here is how the program works:**
- Take two file name strings as inputs, the file path of text_train.json, and the
file path of text_test.json
- Use KMean to cluster documents in train_file.json into 3 clusters by cosine similarity (Can be defined using NLTK K-mean).
- Test the clustering model performance using test_file:
  + Predict the cluster ID for each document in test_file.
  + Use the first label in the ground-truth label list of each test document as one layer clustering is used in this project.
  + Apply majority vote rule to dynamically map the predicted cluster IDs to the ground-truth labels in test_file.
  + Calculate precision/recall/f-score for each label.
