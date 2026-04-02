import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def load_data():
    df = pd.read_csv("data.csv")
    return df

def prepare_data(df, country):
    df = df[df['Country'] == country]
    df = df[['Date', 'Confirmed']]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    return df

def train_model(df):
    X = df[['Days']]
    y = df['Confirmed']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future(model, last_day, days=7):
    future_days = np.array(range(last_day+1, last_day+days+1)).reshape(-1,1)
    predictions = model.predict(future_days)
    return future_days, predictions