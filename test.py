import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from neural_network import NeuralNetwork

print("caricamento del dataset")

# carico il dataset
df = pd.read_csv("mushroom.csv")

# one hot encoding
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(df[['class']])
one_hot_columns = [f'class_{i}' for i in range(one_hot_encoded.shape[1])]
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)
df = pd.concat([df.drop(columns=["class"]), one_hot_df], axis=1)

df = (df - df.min()) / (df.max() - df.min())

# rimuovo alcune feature del dataset per fare dei test
#samples_to_remove = df.sample(n=30000).index
#df = df.drop(samples_to_remove)

test_df = pd.read_csv("mushroom_test.csv")

print("dataset caricato")

# estraggo le feature e le etichette per il set di test
X_test = test_df.drop(columns=["class_0","class_1"])
y_test = test_df[["class_0","class_1"]]

W_dict = np.load("W.npy",allow_pickle=True).item()
b_dict = np.load("b.npy",allow_pickle=True).item()

W = [W_dict[str(i)] for i in range(len(W_dict))]
b = [b_dict[str(i)] for i in range(len(b_dict))]

# inizializzo la rete neurale
NN = NeuralNetwork([8,100,50,2])
NN.initialize(0.01)
NN.import_weights(W,b)

runningLoss = 0
runningAccuracy = 0
for i in range(X_test.shape[0]):
    NN.forward_propagation(X_test.iloc[[i]].values.reshape(-1,1))

    # calcolo cross-entropy loss
    NN.compute_loss(y_test.iloc[[i]].values)
    runningLoss += NN.loss

    # calcolo accuratezza
    y_pred = np.argmax(NN.z[NN.layers-1].flatten())
    y_true = np.argmax(y_test.iloc[[i]].values.flatten())
    runningAccuracy +=  (1 - abs(y_pred - y_true))

    # ogni 1000 campioni stampo la loss corrente
    if (i+1) % 1000 == 0:
        print(f"Sample {i+1} - Current Loss: {NN.loss}")

print(f"avg loss: {runningLoss / X_test.shape[0]}")
print(f"avg accuracy: {runningAccuracy / X_test.shape[0]}")
