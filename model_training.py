# model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
from feature_extraction import transform_data
from xgboost import XGBClassifier

# Загрузка и подготовка данных
train_data = pd.read_parquet('train.parquet')

# Преобразование тренировочных данных в признаки
df_features = transform_data(train_data)

# Определение признаков (features) и меток (label)
X = df_features.drop(['id'], axis=1)
y = train_data['label']

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Параметры, подобранные через Optuna
best_params = {
    'n_estimators': 335,
    'max_depth': 8,
    'learning_rate': 0.04379696899563761,
    'subsample': 0.7867043294624124,
    'colsample_bytree': 0.6826761290740205,
    'gamma': 2.2017499580218356,
    'reg_alpha': 0.4718109623004104,
    'reg_lambda': 0.5650989123792739,
    'min_child_weight': 6,
    'scale_pos_weight': 4.141945937336672,
    'max_delta_step': 3,
    'colsample_bylevel': 0.7638065259568012,
    'colsample_bynode': 0.7546571849524679,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': 42
}

# Обучение модели с подобранными гиперпараметрами
model = XGBClassifier(**best_params)
model.fit(X, y)

# Сохранение модели и импьютера
joblib.dump(model, 'trained_model.pkl')
joblib.dump(imputer, 'imputer.pkl')

print("Модель обучена на всех данных и сохранена в 'trained_model.pkl'.")

