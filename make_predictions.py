# make_predictions.py

import pandas as pd
import joblib
from feature_extraction import transform_data

# Функция для предсказания на новых данных и сохранения результатов
def make_predictions(input_file='test.parquet', model_path='trained_model.pkl', imputer_path='imputer.pkl', output_csv='predictions.csv'):
    # Загрузка тестовых данных
    test_data = pd.read_parquet(input_file)
    
    # Загрузка обученной модели и импьютера
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    
    # Преобразование данных в признаки
    df_features = transform_data(test_data)
    
    # Выделение идентификаторов
    ids = df_features['id']
    df_features = df_features.drop('id', axis=1)
    
    # Обработка пропущенных значений
    df_features = imputer.transform(df_features)
    
    # Прогнозирование вероятностей
    predictions = model.predict_proba(df_features)[:, 1]
    
    # Сохранение результатов в CSV файл
    result_df = pd.DataFrame({'id': ids, 'score': predictions})
    result_df.to_csv(output_csv, index=False)

    print(f"Предсказания сохранены в '{output_csv}'.")

# Пример использования
if __name__ == "__main__":
    # Генерация предсказаний для test.parquet
    make_predictions()
