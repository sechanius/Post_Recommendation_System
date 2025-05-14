import os
from typing import List
from fastapi import FastAPI
from schema import PostGet, Response
from datetime import datetime
from pydantic import BaseModel
import hashlib

import pandas as pd
from sqlalchemy import create_engine



#batch loading
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "connectionl"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

#load features from SQL base
def load_features() -> pd.DataFrame:
    user_data = batch_load_sql('SELECT * FROM semenyachenko_aa_user_data')
    pt_1 = batch_load_sql('SELECT * FROM semenyachenko_aa_post_text')
    return user_data, pt_1

#load features from SQL base
def load_features_train() -> pd.DataFrame:
    user_data_t = batch_load_sql('SELECT * FROM semenyachenko_aa_user_data_train')
    pt_1_t = batch_load_sql('SELECT * FROM semenyachenko_aa_post_text_train')
    return user_data_t, pt_1_t

#checking where model
def get_model_path(path: str, model_name: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/' + model_name
    else:
        MODEL_PATH = path + model_name
    return MODEL_PATH

#setting user group
def get_exp_group(user_id: int) -> str:
    salt = "_first_experiment"
    value_str = str(user_id) + salt

    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16) % 2

    return "control" if value_num == 0 else "test"

#loading models
def load_models_t():
    from catboost import CatBoostClassifier
    model_path = get_model_path("Python things/final project/2 part/", "model_control")
    model = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
    model.load_model(model_path)
    return model

def load_models():
    from catboost import CatBoostClassifier
    model_path = get_model_path("Python things/final project/2 part/", "model_test")
    model = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
    model.load_model(model_path)
    return model

exp_group = get_exp_group(id)
if exp_group == "control":
    user_df, text_df = load_features_train()
    model = load_models_t()
elif exp_group == "test":
    user_df, text_df = load_features()
    model = load_models()
else:
    raise ValueError('unknown group')

#recomendation function
def recommendations(user_id: int, n=5):
    user = user_df[user_df['user_id'] == user_id]
    user_and_text = user.merge(text_df.drop(['text'], axis=1), how='cross').drop(['user_id'], axis=1)
    df = user_and_text.set_index('post_id')
    df = df.drop(['topic'], axis=1)
    df['predict_proba'] = model.predict_proba(df)[:, 1]
    recommends = df.sort_values(by='predict_proba', ascending=False).head(n)
    recommends = recommends.merge(text_df, left_index=True, right_on='post_id').reset_index()
    recommends = recommends[['post_id', 'text', 'topic']]
    recommends = recommends.rename(columns={'post_id': 'id'}).to_dict('records')
    return recommends


app = FastAPI()

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
		id: int, 
		time: datetime=datetime(year=2021, month=1, day=3, hour=14), 
		limit: int = 5) -> Response:
    return Response(exp_group=get_exp_group(id), recommendations = recommendations(id, limit))