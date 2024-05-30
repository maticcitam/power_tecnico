'''
This file contains functions for backend calculations of project 2 (mostly things that were already done in project 1).
This includes:
- loading raw data
- loading the models and parameters
- running the models
- calculating errors of the models
'''

from sklearn import metrics
import numpy as np
import pandas as pd
import pickle
import warnings
import os
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

model_labels = ['Decision Tree Regressor', 'Gradient Boosting Regressor', 'Linear Regression', 'Neural Network MLPRegressor', 'Random Forest Regressor']

def load_data():
    data = pd.read_csv('testData_2019_NorthTower.csv')

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True, drop=True)
    data['Power-1'] = data['North Tower (kWh)'].shift(1)
    data['Power-2'] = data['North Tower (kWh)'].shift(2)
    data['Power-24'] = data['North Tower (kWh)'].shift(24)
    data['hour'] = data.index.hour
    data.dropna(inplace=True)

    return data

def calculate_errors(data, predictions):
    MAE = metrics.mean_absolute_error(data, predictions) 
    MBE = np.mean(data - predictions)
    MSE = metrics.mean_squared_error(data, predictions)  
    RMSE = np.sqrt(metrics.mean_squared_error(data, predictions))
    cvRMSE = RMSE / np.mean(data)
    NMBE = MBE / np.mean(data)
    R2 = metrics.r2_score(data, predictions)
    labels = ['MAE', 
              'MBE', 
              'MSE', 
              'RMSE', 
              'cvRMSE', 
              'NMBE',
              'R2']
    return pd.Series([MAE, MBE, MSE, RMSE, cvRMSE, NMBE, R2], index=labels)


def run_model(data):  
    params = pickle.load(open('params.pkl', 'rb'))
    models = []
    for filename in os.listdir('models_new'):
        file = os.path.join('models_new', filename)
        models.append(pickle.load(open(file, 'rb')))

    # feature selection
    features = ['Power-1', 'Power-2', 'Power-24', 'hour', 'solarRad_W/m2']
    data_features = data[features]
    for feature in features:
        data_features[feature] = (data_features[[feature]] - params.loc[[feature]]['mean']) / params.loc[[feature]]['std']
        
    data_power = (data[['North Tower (kWh)']]  - params.loc[['North Tower (kWh)']]['mean']) / params.loc[['North Tower (kWh)']]['std']
    result_df = data_power.rename(mapper={'North Tower (kWh)': 'Power consumption [kWh]'}, axis=1)
    for model, label in zip(models, model_labels):
        result_df[label] = model.predict(data_features.values[:,1:])
    
    return result_df
    


if __name__ == '__main__':
    data = load_data()
    result_df = run_model(data)
    print(result_df.head())
    # import matplotlib.pyplot as plt
    # plt.plot(data.index[100:500], result_df['Power consumption [kWh]'][100:500], label='data')
    # plt.plot(data.index[100:500], result_df['Linear Regression'][100:500], label='prediction')
    # plt.show()