import pandas as pd
from joblib import load


df = pd.read_csv('data.csv')
X = df.drop(columns='y')
y = df['y']

search = load('search.joblib')

pred = search.predict(X).astype(bool)
pred_df = pd.Series(pred, index=X.index)
print(pred_df)
