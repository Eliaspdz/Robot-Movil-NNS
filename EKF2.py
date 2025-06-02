import numpy as np


class EKF:
    def __init__(self, n=3, p=1, q=1, r=1, g=1):
        self.z = None
        self.w = np.random.rand(n, 1) * 0.05
        self.y = 0

        self.p = np.diag(np.ones(n)) * p
        self.q = np.diag(np.ones(n)) * q
        self.r = r
        self.g = g

    def training(self, e1, s, sp):
        h = self.get_z(s, sp)
     #   h = h.reshape(((len(s)), 1))
        ph = np.dot(self.p, h)
        matriz = self.r + np.dot(h.T, ph)
        inv = np.linalg.inv(matriz)
        k = np.dot(ph, inv)
        delta_w = self.g * np.dot(k, e1)
        self.w = self.w + delta_w
        self.p = self.p - np.dot(k, np.dot(h.T, self.p)) + self.q

        return self.w

    def __neg__(self, x, sp):
       # print(np.power((1 / (1 + np.exp(-x))), sp))
        return np.power((1 / (1 + np.exp(-x))), sp)  # funcion sigmoide con el dato en negativo

    def get_z(self, s, sp):
        # Verificar que los dos arrays tienen el mismo tamaño
        if len(s) != len(sp):
            return print("Los arrays deben tener el mismo tamaño")
        self.z = []
        for k in range(len(sp)):
            self.z.append([self.__neg__(s[k], sp[k])])

        # Convertir self.z en un array de NumPy
        self.z = np.array(self.z)

        return self.z  # convertimos una lista en un array de NumPy
