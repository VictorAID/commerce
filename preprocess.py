# preprocess.py
import pandas as pd

def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def preprocess_data(data):
    columns_to_check = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']
    #data = remove_outliers(data, columns_to_check)
    X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    #y = data['Yearly Amount Spent']
    return X
