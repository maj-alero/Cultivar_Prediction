import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load data
data = load_wine()
X, y = data.data, data.target

# 2. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Train KNN Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# 4. Save model and scaler as model.h5
with open('model.h5', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'target_names': data.target_names}, f)

print("Success! 'model.h5' for Wine Classification (KNN) has been created.")