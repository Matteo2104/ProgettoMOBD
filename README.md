# Manuale d'utilizzo



### Requisiti: Windows, Python 3.7 + pip

Per installare le librerie e le versioni necessarie digitare:

`.\install.bat`

In caso di errori consultare il file `requirements.txt` per verificare manualmente le versioni e le librerie necessarie al funzionamento del programma.

### Addestramento

Per avviare l'addestramento sul train-set digitare:

`py train.py `

È possibile modificare l'architettura della rete alla riga 25 del file  `train.py`, settando opportunamente i parametri del costruttore NeuralNetwork(structure, optimizer)

**NeuralNetwork(structure, optimizer)**

- **structure**: un array contenente gli strati della rete (e.g. [5,10,2] è una rete con 3 strati rispettivamente di 5,10 e 2 neuroni)

- **optimizer**: una stringa nell'insieme {"sdg","adam"} che indica l'algoritmo di ottimizzazione utilizzato dalla rete. "sdg" è default.

È sempre possibile interrompere prematuramente la fase di addestramento premendo la combinazione `CTRL+C`, e i pesi correnti del modello verranno comunque salvati sulla directory principale sotto i nomi `W.npy` e `b.npy`. 

### Test & performance

Per valutare le prestazioni del modello sul test-set digitare:

`py test.py`
