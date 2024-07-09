import numpy as np

class ADAMOptimizer:
    def __init__(self,beta1,beta2,layers):
        self.m_dE = [0 for layer in range(layers-1)]
        self.m_db = [0 for layer in range(layers-1)]
        self.v_dE = [0 for layer in range(layers-1)]
        self.v_db = [0 for layer in range(layers-1)]

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-10

    def update(self,dE,db,layer,t):
        self.m_dE[layer] = self.beta1*self.m_dE[layer] + (1-self.beta2)*dE 
        self.m_db[layer] = self.beta1*self.m_db[layer] + (1-self.beta2)*db

        self.v_dE[layer] = self.beta2*self.v_dE[layer] + (1-self.beta2)*(dE**2)
        self.v_db[layer] = self.beta2*self.v_db[layer] + (1-self.beta2)*(db**2)

        # bias correction
        m_dw_corr = self.m_dE[layer] / (1-self.beta1**t)
        m_db_corr = self.m_db[layer] / (1-self.beta1**t)
        v_dw_corr = self.v_dE[layer] / (1-self.beta2**t)
        v_db_corr = self.v_db[layer] / (1-self.beta2**t)

        update_dE = (m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        update_db = (m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))

        return (update_dE,update_db)


        