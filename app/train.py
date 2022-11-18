import os
import sys
import warnings
import json

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


    def generate_submission():  # Report generation
        client = mlflow.tracking.MlflowClient(MLFLOW_SERVER_URI)

        runs = {}
        models = [
            {'name': m.name,
             'versions': [
                 {'current_stage': v.current_stage, 'run_id': v.run_id, 'status': v.status}
                 for v in m.latest_versions if m.name == 'sk-learn-model-ci']}
            for m in client.search_registered_models()
        ]
        for e in client.list_experiments():
            if e.name == 'Twitter':
                for run_info in client.search_runs(e.experiment_id):
                    run = mlflow.get_run(run_info.info.run_id)
                    runs[run_info.info.run_id] = {'run_id': run_info.info.run_id, 'tags': run.data.tags,
                                             'params': run.data.params,
                                             'metrics': run.data.metrics}
        versions = [{'version': v.version, 'run_id': v.run_id} for v in
                    client.search_model_versions(f"name='{nlp_model_name}'")]
        with open('submission.json', 'w') as f:
            json.dump({'runs': runs, 'models': models, 'versions': versions}, f)


    MLFLOW_SERVER_URI = 'http://web:5000'

    client = mlflow.tracking.MlflowClient(MLFLOW_SERVER_URI)
    mlflow.set_tracking_uri(MLFLOW_SERVER_URI)

    EXP_NAME = "Twitter-Test"
    EXP_ID = mlflow.create_experiment(EXP_NAME)

    mlflow.set_experiment(EXP_NAME)

    with mlflow.start_run() as run:
        assert run.info.experiment_id == EXP_ID
        print("Experiment created successfully".upper())


    for k in range(1000, 10000, 2000):
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

    experiment = client.get_experiment_by_name(EXP_NAME)
    nlp_model_name = "Twitter-Bow"
    client.create_registered_model(nlp_model_name)

    # staging model
    run_info = client.search_runs(experiment.experiment_id)[0]
    result = client.create_model_version(
        name=nlp_model_name,
        source=f"{run_info.info.artifact_uri}/model",
        run_id=run_info.info.run_id
    )
    client.transition_model_version_stage(
        name=nlp_model_name,
        version=result.version,
        stage="Staging"
    )
    # prod model
    run_info = client.search_runs(experiment.experiment_id)[-1]
    result = client.create_model_version(
        name=nlp_model_name,
        source=f"{run_info.info.artifact_uri}/model",
        run_id=run_info.info.run_id
    )
    client.transition_model_version_stage(
        name=nlp_model_name,
        version=result.version,
        stage="Production"
    )

    client = mlflow.tracking.MlflowClient(MLFLOW_SERVER_URI)
    experiment = client.get_experiment_by_name(EXP_NAME)
    client.search_runs(experiment.experiment_id)

    current_staging = \
    [v for v in client.search_model_versions(f"name='{nlp_model_name}'") if v.current_stage == 'Staging'][
        -1]

    client.set_tag(current_staging.run_id, "staging", "failed")

    current_prod = \
    [v for v in client.search_model_versions(f"name='{nlp_model_name}'") if v.current_stage == 'Production'][
        -1]
    prod_metrics = client.get_run(current_prod.run_id).data.metrics
    current_staging = \
    [v for v in client.search_model_versions(f"name='{nlp_model_name}'") if v.current_stage == 'Staging'][
        -1]
    current_staging_metrics = client.get_run(current_staging.run_id).data.metrics

    # Task 1
    for run_info in client.search_runs(experiment.experiment_id):
        run = mlflow.get_run(run_info.info.run_id)

        # Compare each experiment with prod
        current_metrics = client.get_run(run_info.info.run_id).data.metrics
        current_tags = client.get_run(run_info.info.run_id).data.tags

        client.set_tag(run_info.info.run_id, "compared_with", current_prod.version)

        if all(current_metrics[k] > v for k, v in prod_metrics.items()):
            client.set_tag(run_info.info.run_id, "staging", "rc")
        else:
            client.set_tag(run_info.info.run_id, "staging", "rejected")


    # find production id and metrics
    def get_production(client_):
        for mv in client_.search_model_versions(f"name='{nlp_model_name}'"):
            if dict(mv)['current_stage'] == 'Production':
                return mv


    prod_mv = get_production(client)
    print(f'Production ID: {prod_mv.run_id}')
    print(f'Production Version: {prod_mv.version}')


    # find production id and metrics
    def get_staging(client_):
        for mv in client_.search_model_versions(f"name='{nlp_model_name}'"):
            if dict(mv)['current_stage'] == 'Staging':
                return mv


    staging_mv = get_staging(client)
    print(f'Staging ID: {staging_mv.run_id}')
    print(f'Staging Version: {staging_mv.version}')

    # Task 2
    metrics = {}

    for run_info in client.search_runs(experiment.experiment_id):
        run = mlflow.get_run(run_info.info.run_id)

        # for all models that pass the initial selection (if experiment is a release candidate)
        if 'rc' in run.data.tags['staging']:
            test_metrics = client.get_run(run_info.info.run_id).data.metrics

            if all(test_metrics[k] > v for k, v in prod_metrics.items()):
                client.set_tag(run_info.info.run_id, "staging", "rc")
                prod = client.create_model_version(
                    name=nlp_model_name,
                    source=f"{run_info.info.artifact_uri}/model",
                    run_id=run_info.info.run_id
                )
                client.transition_model_version_stage(
                    name=nlp_model_name,
                    version=prod.version,
                    stage="Production"
                )
                prod_metrics = client.get_run(prod.run_id).data.metrics
            metrics[run_info.info.run_id] = test_metrics.get('AUC')

    print(f'Production ID: {get_production(client).run_id}')
    print(f'Production Version: {get_production(client).version}', end="\n\n")

    print(f'Staging ID: {get_staging(client).run_id}')
    print(f'Staging Version: {get_staging(client).version}')

    print(f'Production Metrics: {prod_metrics}')
    print(f'Production Metrics: {mlflow.get_run(get_production(client).run_id).data.metrics}')

    for run_info in client.search_runs(experiment.experiment_id):
        for tag in ['staging', 'compared_with']:
            client.delete_tag(run_info.info.run_id, tag)

    generate_submission()

