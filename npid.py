
import numpy as np

class nPID:
	def __init__(self,n=3,beta=1):
		
		self.w = np.random.rand(n,1)
		self.beta = beta
		self.y = 0
		
		self.P = np.diag(np.ones(n))*10
		self.Q = np.diag(np.ones(n))
		self.R = 0.1

	def control_u(self, x,alpha=0.09):
		self.alpha = alpha #Guarda el valor de alpha
		self.v = np.dot(self.w.T,x) #producto punto entre W ( 3 pesos PID) Transpuesto por el error --
		#De las ganancias PID  []
		self.y = np.tanh(self.v*self.alpha) # W (pesos) * entrada (x los Valores pid)
		return self.y*self.beta #Retorna tres datos despues de la funcion de activacion * learning rate
		
	def fit(self,error,x,eta=0.01):
		H = self.get_H(x)
		H = H.reshape((3,1))
		PH = np.dot(self.P,H)
		matriz = self.R + np.dot(H.T,PH)
		inv = np.linalg.inv(matriz)
		k = np.dot(PH,inv)
		
		delta_w = eta*np.dot(k,error)
		self.w = self.w+ delta_w
		self.P = self.P - np.dot(k,np.dot(H.T, self.P)) + self.Q
		
	def get_H(self,x):
		del_phi = (1 - self.y*self.y)*x*self.beta*self.alpha #Derivada de la tangente
		return del_phi
