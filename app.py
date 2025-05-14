import os
from typing import List
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
from pydantic import BaseModel

import pandas as pd
from sqlalchemy import create_engine



#batch loading
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "connection"
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


#checking where model
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

#loading model
def load_models():
    from catboost import CatBoostClassifier
    model_path = get_model_path("catboost_model_50")
    model = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
    model.load_model(model_path)
    return model

user_df, text_df = load_features()
model = load_models()
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

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
		id: int, 
		time: datetime=datetime(year=2021, month=1, day=3, hour=14), 
		limit: int = 5) -> List[PostGet]:
    return recommendations(id, limit)