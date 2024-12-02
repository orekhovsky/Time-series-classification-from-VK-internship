# feature_extraction.py

import pandas as pd
import numpy as np

# Функция для извлечения статистических признаков из временного ряда
def extract_features(values):
    features = {}
    # Статистические признаки
    features['mean'] = np.mean(values)
    features['std'] = np.std(values)
    features['var'] = np.var(values)
    features['min'] = np.min(values)
    features['max'] = np.max(values)
    features['median'] = np.median(values)
    features['25_quantile'] = np.percentile(values, 25)
    features['50_quantile'] = np.percentile(values, 50)  # То же, что и медиана
    features['75_quantile'] = np.percentile(values, 75)
    features['range'] = np.max(values) - np.min(values)
    features['sum'] = np.sum(values)
    features['last_value'] = values[-1]
    
    # Дополнительные признаки
    features['first_value'] = values[0]
    features['sum_first_5'] = np.sum(values[:5]) if len(values) >= 5 else np.sum(values)
    features['sum_last_5'] = np.sum(values[-5:]) if len(values) >= 5 else np.sum(values)
    
    # Коэффициент вариации
    features['coef_variation'] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
    
    # Автокорреляция с лагом 1 (проверка на нулевое стандартное отклонение)
    if np.std(values) == 0:
        features['autocorr_lag1'] = 0  # Если стандартное отклонение равно нулю, автокорреляция не определена
    else:
        features['autocorr_lag1'] = pd.Series(values).autocorr(lag=1)

    return features

# Функция для извлечения временных признаков из дат
def extract_time_features(dates):
    features = {}
    dates = pd.to_datetime(dates)
    
    features['date_start_month'] = dates[0].month
    features['date_end_month'] = dates[-1].month
    features['date_range_days'] = (dates[-1] - dates[0]).days
    
    features['start_day_of_week'] = dates[0].weekday()
    features['end_day_of_week'] = dates[-1].weekday()
    
    return features

# Функция для преобразования набора данных в признаки
def transform_data(data):
    rows = []
    for _, row in data.iterrows():
        value_features = extract_features(row['values'])
        time_features = extract_time_features(row['dates'])
        combined_features = {**value_features, **time_features, 'id': row['id']}
        rows.append(combined_features)
    return pd.DataFrame(rows)
