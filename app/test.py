import pandas as pd
import requests

df = pd.read_csv('test_data.csv')
row = df.iloc[0].to_dict()

resp = requests.post("http://localhost:8000/predict", json=row)
print(resp.json())
