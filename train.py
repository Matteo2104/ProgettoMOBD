import pandas as pd
import numpy as np
from neural_network import NeuralNetwork
import signal
import sys

def signal_handler(sig, frame):
    print('Premuto Ctrl+C! Uscita dal programma...')
    print("Addestramento terminato. Salvataggio dei pesi.")
    W_dict = {str(i): NN.W[i] for i in range(len(NN.W))}
    b_dict = {str(i): NN.b[i] for i in range(len(NN.b))}
    np.save('W.npy', W_dict)
    np.save("b.npy", b_dict)
    print("Pesi salvati correttamente")
    sys.exit(0)

print("caricamento del dataset...")

# carico il dataset
train_df = pd.read_csv("mushroom_train.csv")

print("dataset caricato.")

# inizializzo la rete neurale
NN = NeuralNetwork([8,100,50,2],"adam")
NN.initialize(0.01)

# Associa il gestore di segnale alla segnalazione di interruzione (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

epochs = 50
for e in range(epochs):
    print(f"Epoca {e+1}")

    # mescolo il set di addestramento e lo divido in features e etichette
    # lo faccio ad ogni inizio epoca per evitare che la rete apprenda l'ordine dei dati presentati
    train_df = train_df.sample(frac=1)
    X_train = train_df.drop(columns=["class_0","class_1"])
    y_train = train_df[["class_0","class_1"]]

    runningLoss = 0
    for i in range(X_train.shape[0]):
        NN.forward_propagation(X_train.iloc[[i]].values.reshape(-1,1))
        NN.back_propagation(y_train.iloc[[i]].values)
        NN.counter(e,X_train.shape[0],i)
        NN.update()

        NN.compute_loss(y_train.iloc[[i]].values)
        runningLoss += NN.loss

        if (i+1) % 1000 == 0:
            print(f"Sample {i+1} - Current Loss: {NN.loss}")
    
    print(f"avg loss: {runningLoss / X_train.shape[0]}")

    if runningLoss / X_train.shape[0] < 0.1:
        print("Addestramento terminato. Salvataggio dei pesi.")
        W_dict = {str(i): NN.W[i] for i in range(len(NN.W))}
        b_dict = {str(i): NN.b[i] for i in range(len(NN.b))}
        np.save('W.npy', W_dict)
        np.save("b.npy", b_dict)
        print("Pesi salvati correttamente")
        sys.exit(0)
