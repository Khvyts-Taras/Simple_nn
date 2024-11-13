from sklearn.datasets import fetch_openml
from simple_nn import *
import os.path


mnist = fetch_openml('mnist_784', version=1)
x = mnist.data / 255.0
y = mnist.target.astype(int)
y_one_hot = np.eye(10)[y]

model = Sequential([Linear(784, 64), LeakyReLU(), Linear(64, 32), LeakyReLU(), Linear(32, 10), Sigmoid()])

model_file = "weights.npz"
if os.path.exists(model_file):
    model.load_weights(model_file)

lr = 0.1
epochs = 5
batch_size = 256

for epoch in range(epochs):
    epoch_loss = 0
    for x_batch, y_batch in get_batches(x, y_one_hot, batch_size):
        res = model(x_batch)
        loss, grad, _ = MSELoss(y_batch, res)
        
        model.backward(grad)
        model.step(lr)

        epoch_loss += loss
    
    if epoch % 1 == 0:
        print(f"Эпоха {epoch + 1}, Потери: {epoch_loss}")
        model.save_weights(model_file)

res = model(x)
predictions = np.argmax(res, axis=1)
accuracy = np.mean(predictions == y) * 100
print(f"Точность модели: {accuracy:.2f}%")