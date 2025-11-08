from keras.datasets import mnist
from keras.utils import to_categorical
from sub_nn import build_model
import os


isCLI = False
epochs = 1
batch_size = 64 
output_dir = './models/test_model' 


if isCLI:
    print("===================================")
    print("Training Sub NN on MNIST dataset")
    print("===================================")
    print("How many epochs? :")
    epochs = int(input())
    print("Batch size?")
    batch_size = int(input())
    print("Output directory?")
    output_dir = input().strip()
    print("===================================")
    print("Training with epochs=", epochs, ", batch_size=", batch_size, ", output_dir=", output_dir)

(X_train, y_train), (X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = build_model(num_classes=10, use_softmax=True)
model.fit(X_train, y_train,
          epochs=1,
          batch_size=64,
          validation_data=(X_test, y_test)
    )

json_string = model.to_json()
open('mnist.json','w').write(json_string)

os.makedirs(output_dir, exist_ok=True)
weights_path = os.path.join(output_dir, 'test_mnist.weights.h5')
model.save_weights(weights_path)

score = model.evaluate(X_test, y_test, verbose=1)

print('LOSS=', score[0])
print('ACCURACY=', score[1])
