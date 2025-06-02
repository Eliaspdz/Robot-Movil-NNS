import matplotlib.pyplot as plt
import numpy as np
from EKF2 import EKF

# Initialization
k = 0  # Counter

# Inicialización de filtro de Kalman de cada subsistema
p1, p2, p3, p4, p5, p6, p7 = 1e8, 1e4, 8e6, 4e7, 4e9, 1e8, 1e8
q1, q2, q3, q4, q5, q6,q7= 5e5, 5e5, 8e10, 1e-3, 5e-1, 5e0, 5e0
r1, r2, r3, r4, r5, r6, r7 = 1e4, 5e7, 2e10, 1e4, 2e4, 1.1e4, 1.1e4
g1, g2, g3, g4, g5, g6, g7 = 1, 11, 1, 1, 1, 1, 1  # learning rate

# Creacion de las Neuronas que seran los pesos
neurona_1 = EKF(3, p1, q1, r1, g1)
neurona_2 = EKF(3, p2, q2, r2, g2)
neurona_3 = EKF(3, p3, q3, r3, g3)
neurona_4 = EKF(2, p4, q4, r4, g4)
neurona_5 = EKF(2, p5, q5, r5, g5)
neurona_6 = EKF(1, p6, q6, r6, g6)
neurona_7 = EKF(1, p7, q7, r7, g7)

# Inicialización de estados y pesos (randomizados simulación donde se adquieren los datos)
X1 = np.random.rand()
X2 = np.random.rand()
X3 = np.random.rand()
X4 = np.random.rand()
X5 = np.random.rand()
X6 = np.random.rand()
X7 = np.random.rand()




# Variables del sistema
M = np.array([[0.3795, -0.0145],  # Matriz de inercia, definida positiva por parametros fisicos
              [-0.0145, 0.3795]])

M_inv = np.linalg.inv(M)

D = np.array([[5, 0],
              [0, 5]])

Disturb = np.array([[0.1],
                    [0.1]])

N = np.array([[62.55, 0],
              [0, 62.55]])

KT = np.array([[0.2613, 0],  #
               [0, 0.2613]])

KE = np.array([[0.0200, 0],  #Coeficiente  de fuerza electromotices
               [0, 0.0200]])

La = np.array([[0.0480, 0],  # inductancio de los motores
               [0, 0.0480]])

La_inv = np.linalg.inv(La)

Ra = np.array([[2.5, 0],  # resistencia del actuador
               [0, 2.5]])

td = np.array([[0.1],  # perturbaciones
               [0.1]])

R1 = 0.31  # Ancho del Robot del primer sistema
r1 = 0.354  # Radio de la rueda Del primer sistema
R = 0.75  # Ancho del Robot del segundo sistema
r = 0.15  # Radio de la rueda Del segundo sistema
mc = 30  # Masa del Robot
d = 0.3  # Distancia del centro de masa al eje del Motor

NN = 50000
t = 0.001  # Paso de tiempo
u = np.array([[10], [10]])

# Inicializacion de Variables
x_1 = np.array([[0], [0], [0]]) # X, Y , theta
x_2 = np.array([[0], [0]])  # V2 y V2
x_3 = np.array([[0], [0]]) #I1 y I2

#store values
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []


x_history = []
y_history = []
theta_history = []
v1_history = []
v2_history = []
i1_history = []
i2_history = []

#Creamos valores para la graficacion
esp = np.linspace(1, NN, NN)

for k in range(NN):
    ############# Primer Subsistema ###################

    # Matriz de transformación
    dq = np.array([[np.cos(x_1[2, 0]), np.cos(x_1[2, 0])],
                   [np.sin(x_1[2, 0]), np.sin(x_1[2, 0])],
                   [(1 / R1), (-1 / R1)]])
    # Producto qdot
    dx_1 = 0.5 * r1 * np.dot(dq, x_2)
    x_1 = x_1 + dx_1 * t

    ############### Segundo Subsistema ############
    # Cálculo de C
    C = 0.5 * (1 / R) * r ** 2 * mc * d * np.array([[0.0, dx_1[2,0]], [-dx_1[2,0], 0.0]])

    # Cálculo de x2dot
    # M = np.array([[x_1[0, 0], x_1[1, 0]],  # Matriz de inercia, definida positiva por parametros fisicos
    #               [x_1[1, 0], x_1[0, 0]]])
    #
    # M_inv = np.linalg.inv(M)

    dx_2 = np.dot(M_inv, (-np.dot(C, x_2) - np.dot(D, x_2) - td + np.dot(N, np.dot(KT, x_3)) ))
    x_2 = x_2 + dx_2 * t

    ################## Tercer Subsistema ###################

    dx_3 = np.dot(La_inv, (u - np.dot(Ra, x_3) - np.dot(N, np.dot(KE, x_2))))
    x_3 = x_3 + dx_3 * t

    #Valores del sistema simulado
    x_medida = np.squeeze(x_1[0])
    y_medida = np.squeeze(x_1[1])
    theta_medida = np.squeeze(x_1[2])
    v1_medida = np.squeeze(x_2[0])
    v2_medida = np.squeeze(x_2[1])
    i1_medida = np.squeeze(x_3[0])
    i2_medida = np.squeeze(x_3[1])



    # Neural network
    # Initialize state vector for each subsystem

    s1 = [x_medida, y_medida, theta_medida]
    s2 = [x_medida, y_medida, theta_medida]
    s3 = [x_medida, y_medida, theta_medida]
    s4 = [v1_medida, i1_medida]
    s5 = [v2_medida, i2_medida]
    s6 = [i1_medida]
    s7 = [i2_medida]

    sp1 = [1, 1, 1]
    sp2 = [1, 1, 1]
    sp3 = [1, 1, 1]
    sp4 = [1, 1]
    sp5 = [1, 1]
    sp6 = [1]
    sp7 = [1]

    # Cálculo de las matrices Z
    Z1 = neurona_1.get_z(s1, sp1)
    Z2 = neurona_2.get_z(s2, sp2)
    Z3 = neurona_3.get_z(s3, sp3)
    Z4 = neurona_4.get_z(s4, sp4)
    Z5 = neurona_5.get_z(s5, sp5)
    Z6 = neurona_6.get_z(s6, sp6)
    Z7 = neurona_7.get_z(s7, sp7)

    e1 = x_medida - X1
    e2 = y_medida - X2
    e3 = theta_medida - X3
    e4 = v1_medida - X4
    e5 = v2_medida - X5
    e6 = i1_medida - X6
    e7 = i2_medida - X7

    W1 = neurona_1.training(e1, s1, sp1)
    W2 = neurona_2.training(e2, s2, sp2)
    W3 = neurona_3.training(e3, s3, sp3)
    W4 = neurona_4.training(e4, s4, sp4)
    W5 = neurona_5.training(e5, s5, sp5)
    W6 = neurona_6.training(e6, s6, sp6)
    W7 = neurona_7.training(e7, s7, sp7)

    # Estructura de los 6 estados con pesos y sigmodes
    X1 = W1.T @ Z1 + 0.001 * v1_medida + 0.001 * v2_medida
    X2 = W2.T @ Z2 + 0.001 * v1_medida + 0.001 * v2_medida
    X3 = W3.T @ Z3 + 0.001 * v1_medida - 0.001 * v2_medida
    X4 = W4.T @ Z4 + 0.001 * i1_medida
    X5 = W5.T @ Z5 + 0.001 * i2_medida
    X6 = W6.T @ Z6 + 0.001 * u[0]
    X7 = W7.T @ Z7 + 0.001 * u[1]

    #Valores de almacenamiento

    x1.append(X1)
    x2.append(X2)
    x3.append(X3)
    x4.append(X4)
    x5.append(X5)
    x6.append(X6)
    x7.append(X7)

    x_history.append(x_1[0])
    y_history.append(x_1[1])
    theta_history.append(x_1[2])
    v1_history.append(x_2[0])
    v2_history.append(x_2[1])
    i1_history.append(x_3[0])
    i2_history.append(x_3[1])


x1_flat = np.squeeze(x1)
x2_flat = np.squeeze(x2)
x3_flat = np.squeeze(x3)
x4_flat = np.squeeze(x4)
x5_flat = np.squeeze(x5)
x6_flat = np.squeeze(x6)
x7_flat = np.squeeze(x7)

#Creamos valores para la graficacion
esp = np.linspace(1, NN, NN)

plt.figure(figsize=(15, 5))

# Gráfico para comparar x_medida con X1
plt.subplot(1, 3, 1)
plt.plot(esp, x_history, label='X')
plt.plot(esp, x1_flat, label='X1')
plt.title('X')
plt.grid(True)
plt.legend()

    # Gráfico para comparar y_medida con X2
plt.subplot(1, 3, 2)
plt.plot(esp, y_history, label='Y')
plt.plot(esp, x2_flat, label='X2')
plt.title('Y')
plt.grid(True)
plt.legend()

    # Gráfico para comparar theta_medida con X3
plt.subplot(1, 3, 3)
plt.plot(esp, theta_history, label='theta')
plt.plot(esp, x3_flat, label='X3')
plt.title('Theta')
plt.grid(True)
plt.legend()



plt.figure(figsize=(15, 5))

# Gráfico para comparar x_medida con X1
plt.subplot(1, 2, 1)
plt.plot(esp, v1_history, label='v1')
plt.plot(esp, x4_flat, label='X4')
plt.title('V1')
plt.grid(True)
plt.legend()

    # Gráfico para comparar y_medida con X2
plt.subplot(1, 2, 2)
plt.plot(esp, v2_history, label='v2')
plt.plot(esp, x5_flat, label='X5')
plt.title(' v2')
plt.grid(True)
plt.legend()


plt.figure(figsize=(15, 5))

# Gráfico para comparar x_medida con X1
plt.subplot(1, 2, 1)
plt.plot(esp, i1_history, label='i1')
plt.plot(esp, x6_flat, label='X6')
plt.title('i1')
plt.grid(True)
plt.legend()

    # Gráfico para comparar y_medida con X2
plt.subplot(1, 2, 2)
plt.plot(esp, i2_history, label='i2')
plt.plot(esp, x7_flat, label='X7')
plt.title(' i2')
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()