import numpy as np
from adam_optimizer import ADAMOptimizer

class NeuralNetwork:
    def __init__(self,structure,optimizer="sdg"):
        self.structure = structure
        self.layers = len(structure)

        if optimizer == "adam":
            self.optimizer = "adam"
            self.Adam = ADAMOptimizer(0.99,0.999,self.layers)
        else:
            self.optimizer = "sdg"

        self.epoch = 0
        self.iterations = 0


    def softmax(self,x):
        e_x = np.exp(x-np.max(x))
        return e_x / e_x.sum()
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self,x):
        return np.maximum(0, x)
    
    def relu_derivative(self,x):
        return np.where(x <= 0, 0, 1)

    def initialize(self,alpha):
        self.alpha = alpha
        self.loss = 0

        # inizializzo le matrici dei pesi e dei bias - utilizzo inizializzazione di He
        self.W = [np.random.randn(self.structure[layer+1],self.structure[layer]) * np.sqrt(2 / self.structure[layer]) for layer in range(self.layers-1)]
        self.b = [np.random.randn(self.structure[layer+1],1) * np.sqrt(2 / self.structure[layer]) for layer in range(self.layers-1)]
        
        # inizializzo le matrici dei gradienti
        self.dE = [np.zeros((self.structure[layer+1],self.structure[layer])) for layer in range(self.layers-1)]
        self.db = [np.zeros((self.structure[layer+1],1)) for layer in range(self.layers-1)]
        
        # inizializzo i vettori delle comb. lineari e degli output
        self.a = [np.zeros((n,1)) for n in self.structure] # a = Wz + b
        self.z = [np.zeros((n,1)) for n in self.structure] # z = g(a)

    # importa i pesi dall'esterno
    def import_weights(self,W,b):
        # TODO: inserire controllo sulle dimensioni
        self.W = W.copy()
        self.b = b.copy()

    def counter(self,epoch,tot_samples,current_sample):
        self.epoch = epoch
        self.iterations = epoch*tot_samples + (current_sample+1)

    def forward_propagation(self,x):
        g = self.relu
        self.z[0] = x.reshape(len(x),1)
        for layer,w in enumerate(self.W):
            # se sto nel penultimo strato uso una funzione sigmoide
            if layer == self.layers-2: 
                g = self.sigmoid
            self.a[layer+1] = np.dot(w,self.z[layer]) + self.b[layer]
            self.z[layer+1] = g(self.a[layer+1])
    
    def back_propagation(self,y):
        # strato finale
        l = self.layers-2
        dz = [0 for layer in range(self.layers-1)]

        dz[l] = (self.z[l+1] - y.T)
        self.dE[l] = np.dot(dz[l], self.z[l].T)
        self.db[l] = dz[l]

        # strati nascosti
        for l in range(self.layers-3,-1,-1):
            term1 = np.dot( self.W[l+1].T, dz[l+1] )
            dz[l] = term1 * self.relu_derivative(self.a[l+1])
            self.dE[l] = np.dot(dz[l], self.z[l].T)
            self.db[l] = dz[l]

     # Stochastic Gradient Descent
    def sdg(self):
        for layer in range(self.layers-1):
            self.W[layer] = self.W[layer] - self.alpha*self.dE[layer] 
            self.b[layer] = self.b[layer] - self.alpha*self.db[layer]

    # ADAM
    def adam(self):
        for layer in range(self.layers-1):
            update1,update2 = self.Adam.update(self.dE[layer],self.db[layer],layer,self.iterations)

            self.W[layer] = self.W[layer] - self.alpha*update1
            self.b[layer] = self.b[layer] - self.alpha*update2

    def update(self):
        if self.optimizer == "adam":
            self.adam()
        else:
            self.sdg() 
    
    # calcola la cross-entropy loss
    def compute_loss(self,y):
        self.loss = -np.sum(y * np.log(self.z[self.layers-1].T))


                


    