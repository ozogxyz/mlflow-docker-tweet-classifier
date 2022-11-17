import os
import sys
import warnings

import mlflow.sklearn
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features import BoW

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    try:
        data = pd.read_csv('../../datasets/comments_small_dataset/comments.tsv', sep='\t')
    except FileNotFoundError:
        os.system('wget https://raw.githubusercontent.com/girafe-ai/ml-course/master/datasets/comments_small_dataset'
                  '/comments.tsv -nc')
        data = pd.read_csv("comments.tsv", sep='\t')

    texts = data['comment_text'].values
    target = data['should_ban'].values
    texts_train, texts_test, y_train, y_test = train_test_split(texts, target, test_size=0.5, random_state=42)

    tokenizer = TweetTokenizer()
    preprocess = lambda text: ' '.join(tokenizer.tokenize(text.lower()))
    texts_train = [preprocess(text) for text in texts_train]  # <YOUR CODE>
    texts_test = [preprocess(text) for text in texts_test]  # <YOUR CODE>

    text = 'How to be a grown-up at work: replace "I don\'t want to do that" with "Ok, great!".'
    print("before:", text, )
    print("after:", preprocess(text), )

    MLFLOW_SERVER_URI = 'http://web:5000'
    mlflow.set_tracking_uri(MLFLOW_SERVER_URI)

    EXP_NAME = "lsml"
    EXP_ID = mlflow.create_experiment(EXP_NAME)

    mlflow.set_experiment(EXP_NAME)
    with mlflow.start_run() as run:
        assert run.info.experiment_id == EXP_ID
        print("Experiment created successfully".upper())

    k_min = sys.argv[1] if len(sys.argv) > 1 else 1000

    for k in [min(1000, k_min), min(2000, k_min), min(3000, k_min), min(4000, k_min)]:
        with mlflow.start_run():
            bow = BoW(k)
            bow.fit(texts_train)
            print('example features:', sorted(bow.get_vocabulary())[::100])

            X_train_bow = bow.transform(texts_train)
            X_test_bow = bow.transform(texts_test)

            bow_model = LogisticRegression().fit(X_train_bow, y_train)

            for name, X, y, model in [
                ('train', X_train_bow, y_train, bow_model),
                ('test ', X_test_bow, y_test, bow_model)
            ]:
                proba = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(y, proba)

                print(f"{name} AUC: {auc}")

                # mlflow stuff
                mlflow.log_param("k", k)
                mlflow.log_metric("AUC", auc)

                mlflow.sklearn.log_model(bow_model, "model")
