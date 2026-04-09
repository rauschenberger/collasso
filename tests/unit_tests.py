
import numpy as np

x_train, y_train, x_test, y_test, beta = simulate(rho=0.9, prob_com=0.05, prob_sep=0.05)

# test whether CoopLassoCV returns same results if x is broadcast from matrix to array (for fitting or predicting)

x_train_bc = np.broadcast_to(x_train[:,:,None],(x_train.shape[0],x_train.shape[1],y_train.shape[1]))
x_test_bc = np.broadcast_to(x_test[:,:,None],(x_test.shape[0],x_test.shape[1],y_test.shape[1]))
model = CoopLassoCV(random_state=1)
model.fit(x_train,y_train)
y_hat = []
y_hat.append(model.predict(x_test))
y_hat.append(model.predict(x_test_bc))
model.fit(x_train_bc,y_train)
y_hat.append(model.predict(x_test))
y_hat.append(model.predict(x_test_bc))
y_hat = np.array(y_hat)
np.allclose(y_hat, y_hat[0])

# test whether CoopLasso returns same results if alpha=None and alpha is provided from fitted path

model = CoopLasso()
model.fit(X=x_train,y=y_train)
pred1 = model.predict(X=x_test)
alpha = []
for i in range(len(model.model_)):
    alpha.append(model.model_[i][0])
pred2 = model.predict(X=x_test,alpha=alpha)
np.allclose(pred1,pred2)
