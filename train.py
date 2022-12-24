from models.LSTMv1 import LSTMv1

lstmv1 = LSTMv1(shape=(38, 64, 64, 1))

print(lstmv1.model.summary())
