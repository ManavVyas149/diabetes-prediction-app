# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('diabetes.csv')
print(df.columns)

X = df.drop('Outcome', axis=1) 
y = df['Outcome']

model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully as model.pkl")