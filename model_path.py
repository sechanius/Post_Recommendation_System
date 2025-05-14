import os

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    from catboost import CatBoostClassifier
    model_path = get_model_path("catboost_model")
    model = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
    model.load_model(model_path)
    return model