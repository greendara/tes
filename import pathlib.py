import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# 1. Buat data non-linear
X = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
y = np.sin(X).ravel()

# 2. Model A: Linear regression (tanpa fungsi aktivasi)
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# 3. Model B: Neural network dengan 1 hidden layer + aktivasi
nn_model = MLPRegressor(hidden_layer_sizes=(10,), activation='tanh', max_iter=5000)
nn_model.fit(X, y)
y_pred_nn = nn_model.predict(X)

# 4. Plot hasil
plt.figure(figsize=(10, 5))
plt.plot(X, y, label="True Function (sin)", color='black')
plt.plot(X, y_pred_linear, label="Linear Model (tanpa g)", linestyle='--')
plt.plot(X, y_pred_nn, label="Neural Net (dengan g)", linestyle='dotted')
plt.legend()
plt.title("Perbandingan Model: Dengan vs Tanpa Fungsi Aktivasi")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
