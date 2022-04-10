import joblib
import getdata


X_train, y_train, X_val, y_val, X_test, y_test = getdata.get_mnist_data()
best_net = joblib.load("best_net.pkl")

test_acc = (best_net.predict(X_test) == y_test).mean()
print ('Test accuracy: ', test_acc)