
import numpy as np
import matplotlib.pyplot as plt
from EKF2 import EKF
from npid import nPID
import math

#Inicializacion Kalman

# Inicialización de filtro de Kalman de cada subsistema
# p1, p2, p3 = 1e8, 1e4, 8e6
# q1, q2, q3 = 5e5, 9e8, 7e10
# r1, r2, r3 = 1e4, 5e7, 2e10
# g1, g2, g3 = 1, 1, 1                    # learning rate


p1, p2, p3 = 1000000, 1000000, 1000000
q1, q2, q3 = 1000000, 1000000, 1000000
r1, r2, r3 = 1e4, 1e4, 1e4
g1, g2, g3 = 0.3, 0.3, 0.3                    # learning rate



# Creacion de las Neuronas que seran los pesos
neurona_1 = EKF(3, p1, q1, r1, g1)
neurona_2 = EKF(3, p2, q2, r2, g2)
neurona_3 = EKF(3, p3, q3, r3, g3)


neurona_x = nPID(3,1)
neurona_y = nPID(3,1)

# Inicializacion de las ganancias del controlador PID

ex = 0
ew = 0
eAx = 0
d_ex = 0
i_ex = 0
etax = 0
ey = 0
eAy = 0
d_ey = 0
i_ey = 0
etay = 0
tt = 2
d = 0.08


#Pesos fijos de cada subsistema
g11, g12 = 0.001, 0.001
g21, g22 = 0.001, 0.001
g31, g32 = 0.001, 0.001

gs1 = np.array([[g11], [g12]])
gs2 = np.array([[g21], [g22]])
gs3 = np.array([[g31], [g32]])



# Inicialización de estados y pesos (randomizados simulación donde se adquieren los datos)

X1 = np.random.rand() * 0.1
X2 = np.random.rand() * 0.1
X3 = np.random.rand() * 0.1


L = 0.25                                    # Distancia entre el centro del carro al centro de cada una de sus llantas
p = np.array([0, 0, 0])             # p inicial [x,y,theta]
R = 1                                       # Radio de la llanta
t = 0.05                                    # Paso de tiempo
S = 2000                               # Iteraciones

u = np.array([[0], [0]])                        # Entradas acomodadas para el sistema equivalente de la red neuronal [2,1]

radio = 3
pasos_por_circulo = 500
#Pesos fijos de cada subsistema

# g11, g12 = 0.001, 0.001
# g21, g22 = 0.001, 0.001
# g31, g32 = 0.001, 0.001

# Inicialización de las variables

x1 = []
x2 = []
x3 = []
x_history = []
y_history = []
theta_history = []
ucontrol = []
Xreferencia = []
Yreferencia = []
ThetaReferencia = []
xdpro = []
ydpro = []

E1 = []
E2 = []
Ex = []
Ey = []

# Creamos valores para la graficacion

esp = np.linspace(1, S, S)

for i in range(S):

    dX = u[0] * np.cos(p[2])
    dY = u[0] * np.sin(p[2])
    dTheta = u[1]

    dP = np.array([dX[0], dY[0], dTheta[0]])
    p = p + dP * t

    x_medida = p[0]
    y_medida = p[1]
    theta_medida = p[2]

    # Neural network
    # Initialize state vector for each subsystem

    s1 = [x_medida, y_medida, theta_medida]
    s2 = [x_medida, y_medida, theta_medida]
    s3 = [x_medida, y_medida, theta_medida]

    sp1 = [1, 1, 1]
    sp2 = [1, 1, 1]
    sp3 = [1, 1, 1]

    # Cálculo de las matrices Z
    Z1 = neurona_1.get_z(s1, sp1)
    Z2 = neurona_2.get_z(s2, sp2)
    Z3 = neurona_3.get_z(s3, sp3)

    e1 = x_medida - X1
    e2 = y_medida - X2
    e3 = theta_medida - X3

    W1 = neurona_1.training(e1, s1, sp1)
    W2 = neurona_2.training(e2, s2, sp2)
    W3 = neurona_3.training(e3, s3, sp3)

    f1 = np.dot(W1.T, Z1)
    f2 = np.dot(W2.T, Z2)
    f3 = np.dot(W3.T, Z3)

    # Pesos fijos de cada subsistema

    # g11, g12 = np.cos(f3), np.cos(f3)
    # g21, g22 = np.sin(f3), np.sin(f3)
    # g31, g32 = (1 / L), (1 / L)
    #
    # gs1 = np.array([g11, g12]) * (t * 0.5 * R)
    # gs2 = np.array([g21, g22]) * (t * 0.5 * R)
    # gs3 = np.array([g31, g32]) * (t * 0.5 * R)

    gss1 = gs1.reshape(2, )
    gss2 = gs2.reshape(2, )
    gss3 = gs3.reshape(2, )

    # Estructura de los 6 estados con pesos y sigmodes

    X1 = f1 + np.dot(gss1.T, u)
    X2 = f2 + np.dot(gss2.T, u)
    X3 = f3 + np.dot(gss3.T, u)

    # Calcular el ángulo para el paso actual
    theta = 2 * np.pi * (i % pasos_por_circulo) / pasos_por_circulo

    # Calcular las coordenadas x e y usando las funciones coseno y seno
    xd = radio * np.cos(theta)
    yd = radio * np.sin(theta)

    # xd = 5.0  # Punto deseado
    # yd = 2.0

    xp = X1 + (d * np.cos(X3))
    yp = X2 + (d * np.sin(X3))

    # error de posicion

    ex = xd - xp
    ey = yd - yp

    # PID de la nuerona x
    i_ex = (i_ex + ex) * (1 / tt)
    d_ex = ex - eAx
    eAx = ex
    error_x = np.array([ex, i_ex, d_ex])  # arrar con las 3 Ganancias adaptables

    # PID de la neurona Y
    i_ey = (i_ey + ey) * (1 / tt)
    d_ey = ey - eAy
    eAy = ey
    error_y = np.array([ey, i_ey, d_ey])  # arrar con las 3 Ganancias adaptables

    error_xs = np.reshape(error_x,(3,))
    error_ys = np.reshape(error_y,(3,))

    #
    kx = neurona_x.control_u(error_xs)  # Control u hacer
    neurona_x.fit(ex, error_xs, 0.06)

    ky = neurona_y.control_u(error_ys)
    neurona_y.fit(ey, error_ys, 0.07)

    #errores = np.array([kx, ky])

    matriz_modelos = np.array([[np.cos(X3), -d * np.sin(X3)], [np.sin(X3), d * np.cos(X3)]])
    matriz_modelo = np.reshape(matriz_modelos,(2,2))
    u = np.dot(np.linalg.inv(matriz_modelo), np.array([kx, ky]))

    # us = np.array([kx, ky])
     #u = np.reshape(us,(2,))
    # Supongamos que tenemos un vector X de tamaño 2x1

    # Definir los umbrales
    # umbral_min = -5
    # umbral_max = 5
    #
    # # Limitar los datos del vector X a los umbrales
    # u = np.array([[max(umbral_min, min(umbral_max, u[0]))], [max(-8, min(8, u[1]))]])
    # u = np.reshape(u,(2,1))


    #print(u)




    x1.append(X1)
    x2.append(X2)
    x3.append(X3)
    x_history.append(x_medida)
    y_history.append(y_medida)
    theta_history.append(theta_medida)
    ucontrol.append(u)
    xdpro.append(xd)
    ydpro.append(yd)

    x1_flatPRO = np.squeeze(x1)
    x2_flatPRO = np.squeeze(x2)
    #x3_flat = np.squeeze(x3)

    E1.append(e1)
    E2.append(e2)
    Ex.append(ex)
    Ey.append(ey)

    if isinstance(E1[0], float):  # Si el primer elemento es un escalar
        E1[0] = np.array([[E1[0]]])  # Convierte el primer dato en un array con la misma estructura

    if isinstance(E2[0], float):  # Si el primer elemento es un escalar
        E2[0] = np.array([[E2[0]]])  # Convierte el primer dato en un array con la misma estructura

    E1_array = np.array(E1)
    E2_array = np.array(E2)

    ufinal = np.array(ucontrol)


    #Graficar la trayectoria hasta el paso actual
    plt.clf()
    plt.plot(x1_flatPRO, x2_flatPRO, marker='o', markersize=2, linestyle='-')

    # Graficar el punto en movimiento
    plt.plot(X1, X2, 'ro')  # Punto rojo en la posición actual

    plt.title('Trayectoria Circular Repetida en 2000 Pasos')
    plt.xlabel('xd')
    plt.ylabel('yd')
    plt.grid(True)
    plt.axis('equal')
    plt.pause(0.000001)  # Pausar brevemente para actualizar la gráfica



ufinal = np.reshape(ufinal,(2,S))
# E1_array = np.reshape(E1_array,(1,S))
# E2_array = np.reshape(E2_array,(1,S))

x1_flat = np.squeeze(x1)
x2_flat = np.squeeze(x2)
x3_flat = np.squeeze(x3)

#E1_flat = np.squeeze(E1)
#E2_flat = np.squeeze(E2)

E1_flat = np.squeeze(E1_array)
E2_flat = np.squeeze(E2_array)
Ex_flat = np.squeeze(Ex)
Ey_flat = np.squeeze(Ey)

# Guardamos los arrays en un archivo de texto
np.savetxt('arrays.txt', (x1_flat, x2_flat, x3_flat, esp))
# Guardamos los arrays en archivos de texto individuales

#np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation
# np.savetxt('array_a.out', x1_flat)
# np.savetxt('array_b.out', x2_flat)
# np.savetxt('array_c.out', x3_flat)
# np.savetxt('array_d.out', esp)

# Calcular el RMS de cada variable
def calculate_rms(data):
    return np.sqrt(np.mean(np.square(data)))

# Asegúrate de que las variables no estén vacías
if len(E1_flat) > 0 and len(E2_flat) > 0 and len(Ex_flat) > 0 and len(Ey_flat) > 0:
    rms_E1 = calculate_rms(E1_flat)
    rms_E2 = calculate_rms(E2_flat)
    rms_Ex = calculate_rms(Ex_flat)
    rms_Ey = calculate_rms(Ey_flat)

    # Imprime los valores RMS
    print("RMS de las variables:")
    print("RMS E1:", rms_E1)
    print("RMS E2:", rms_E2)
    print("RMS Ex:", rms_Ex)
    print("RMS Ey:", rms_Ey)
else:
    print("Error: Algunos arrays están vacíos.")

plt.figure(figsize=(8, 6))
plt.plot(esp, theta_history, label='theta')
plt.plot(esp, x3_flat, 'r--', label='Theta identificador')
plt.title('theta and x3')
plt.grid(True)
plt.legend()

plt.figure(figsize=(8, 6))
plt.plot(esp, ufinal[0,:], label='  v1 Velocidad lineal')
plt.plot(esp, ufinal[1,:], 'r--', label='W1 Velocidad angular')
plt.title('Control')
plt.grid(True)
plt.legend()

# Crear una figura y dos subplots (gráficas)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot results
ax1.plot(esp, x_history, label='x medida')
ax1.plot(esp, x1_flat, 'r--', label='x identificador')
ax1.plot(esp, xdpro, 'g--', label='X deseada')
ax1.set_title('x')
ax1.grid(True)
ax1.legend()

ax2.plot(esp, y_history, label='i beta')
ax2.plot(esp, x2_flat, 'r--', label='Y identificador')
ax2.plot(esp, ydpro, 'g--', label='Y deseada')
ax2.set_title('Y')
ax2.grid(True)
ax2.legend()


#errores
plt.figure(figsize=(8, 6))
plt.plot(esp, Ex_flat, label='error de trayectoria X')
plt.plot(esp, Ey_flat, 'r--', label='error de trayectoria Y')
plt.title('Error de trayectoria X y Y')
plt.grid(True)
plt.legend()


# Crear una figura y dos subplots (gráficas)

plt.figure(figsize=(8, 6))
plt.plot(esp, E1_flat, label='Error variable identificada X')
plt.plot(esp, E2_flat, 'r--', label='Error variable identificada Y')
plt.title('Error de identificación ')
plt.grid(True)
plt.legend()

print("Últimos valores:")
print("E1:", E1_flat[-1])
print("E2:", E2_flat[-1])
print("Ex:", Ex_flat[-1])
print("Ey:", Ey_flat[-1])

# Ajustar el espaciado entre las gráficas
plt.tight_layout()

# Mostrar las gráficas
plt.show()
