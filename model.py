import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from tqdm import tqdm

df = pd.read_csv('jm1.csv')

df['defects'] = df['defects'].astype(int)
features = ['loc', 'v', 'l', 'i']
target = 'defects'
X = df[features].values
y = df[target].values

model = SGDClassifier(max_iter=10, tol=1e-3, random_state=0)

batch_size = 4 
mse_scores = []
accuracy_scores = []

for i in tqdm(range(batch_size, len(X), batch_size)):
    X_train, y_train = X[:i], y[:i]
    X_test, y_test = X[i:i + batch_size], y[i:i + batch_size]
    
    model.partial_fit(X_train, y_train, classes=np.unique(y))
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    mse_scores.append(mse)
    accuracy_scores.append(acc)

avg_mse = np.mean(mse_scores)
avg_acc = np.mean(accuracy_scores)

print(f"Average MSE: {avg_mse:.4f}")
print(f"Average Accuracy: {avg_acc:.4f}")