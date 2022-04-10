from neural_net import TwoLayerNet
import numpy as np
import matplotlib.pyplot as plt
import getdata
import joblib




X_train, y_train, X_val, y_val, X_test, y_test = getdata.get_mnist_data()
best_net = None # store the best model into this
best_acc = 0

############

input_size = 28 * 28 * 1
num_classes = 10

# Train the network
for bs in [200, 400]:
    for lr in [1e-3, 1e-4, 1e-5]:
        for hidden_size in [50, 100, 200]:
            net = TwoLayerNet(input_size, hidden_size, num_classes)
            stats = net.train(X_train, y_train, X_val, y_val,
                        num_iters=2000, batch_size=bs,
                        learning_rate=lr, learning_rate_decay=0.9,
                        reg=0.5, verbose=True)

            # Predict on the validation set
            val_acc = (net.predict(X_val) == y_val).mean()
            print ('batch_size = %d, lr = %f, hidden size = %f, Valid_accuracy: %f' %(bs, lr, hidden_size,val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                best_net = net


joblib.dump(best_net,"best_net.pkl")

