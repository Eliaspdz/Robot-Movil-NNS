#!/usr/bin/env python2
# coding=utf-8
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
from npid import nPID


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # funcion sigmoide con el dato en engativo


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
t = 1
d = 0.08

#


p1 = 1000000
p2 = 1000000
p3 = 1000000 
q1 = 1000000
q2 = 1000000
q3 = 1000000
r1 = 1e4
r2 = 1e4
r3 = 1e4 
g1 = 0.3
g2 = 0.3
g3 = 0.3   

g11 = 0.001
g12 = 0.001 
g21 = 0.001
g22 = 0.001
g31 = 0.001
g32 = 0.001 

gs1 = np.array([[g11], [g12]])
gs2 = np.array([[g21], [g22]])
gs3 = np.array([[g31], [g32]])

# Inicialización de estados y pesos (randomizados simulación donde se adquieren los datos)

X1 = np.random.rand() * 0.1
X2 = np.random.rand() * 0.1
X3 = np.random.rand() * 0.1

W1 = np.random.rand(3, 1) - 0.5
W2 = np.random.rand(3, 1) - 0.5
W3 = np.random.rand(3, 1) - 0.5

# Creando la matriz P y Q de cada subsistema
P1 = p1 * np.eye(3)
P2 = p2 * np.eye(3)
P3 = p3 * np.eye(3)

Q1 = q1 * np.eye(3)
Q2 = q2 * np.eye(3)
Q3 = q3 * np.eye(3)

u = np.array([[0], [0]])

neurona_x = nPID(3,0.5)
neurona_y = nPID(3,0.3)
#Inicializacion para Guardar variables
S=2000 #                                   Tiempo del ciclo
vector_uv=np.zeros(S)
vector_uw=np.zeros(S)
vector_x=np.zeros(S)
vector_y=np.zeros(S)
vector_theta=np.zeros(S)
vector_errorx=np.zeros(S)
vector_errory=np.zeros(S)
#Vectores de variables identificadas
vector_x_id=np.zeros(S)
vector_y_id=np.zeros(S)
vector_theta_id=np.zeros(S)
vector_t=np.zeros(S)
#vector_error=np.zeros(S,2)
def Callback(msg_in):
    global ex
    global eAx
    global d_ex
    global i_ex
    global ey
    global eAy
    global t
    global d_ey
    global i_ey
    global d
    global vector_uv
    global vector_uw
    global vector_x
    global vector_y
    global vector_theta
    global vector_errorx
    global vector_errory
    global X1
    global X2
    global X3

    global W1
    global W2
    global W3

    # Creando la matriz P y Q de cada subsistema
    global P1
    global P2
    global P3

    global Q1
    global Q2
    global Q3

    global vector_x_id
    global vector_y_id
    global vector_theta_id

    global u
    global vector_t

    x = msg_in.pose.pose.position.x
    y = msg_in.pose.pose.position.y
    quater = np.zeros(4)
    quater[0] = msg_in.pose.pose.orientation.w
    quater[1] = msg_in.pose.pose.orientation.x
    quater[2] = msg_in.pose.pose.orientation.y
    quater[3] = msg_in.pose.pose.orientation.z
    theta = math.atan2(2*(quater[0]*quater[3]+quater[1]*quater[2]),1-2*(quater[2]*quater[2]+quater[3]*quater[3])) #yaw en robotica

    x_medida = x
    y_medida = y
    theta_medida = theta

    s1 = np.array([[x_medida], [y_medida], [theta_medida]])
    s2 = np.array([[x_medida], [y_medida], [theta_medida]])
    s3 = np.array([[x_medida], [y_medida], [theta_medida]])

    sp1 = [1, 1, 1]
    sp2 = [1, 1, 1]
    sp3 = [1, 1, 1]

    # Cálculo de las matrices Z
    Z1 = np.array([[sigmoid(s1[0, 0]) ** 1],
                   [sigmoid(s1[1, 0]) ** 1],
                   [sigmoid(s1[2, 0]) ** 1]])

    Z2 = np.array([[sigmoid(s2[0, 0]) ** 1],
                   [sigmoid(s2[1, 0]) ** 1],
                   [sigmoid(s2[2, 0]) ** 1]])

    Z3 = np.array([[sigmoid(s3[0, 0]) ** 1],
                   [sigmoid(s3[1, 0]) ** 1],
                   [sigmoid(s3[2, 0]) ** 1]])

    e1 = x_medida - X1
    e2 = y_medida - X2
    e3 = theta_medida - X3

    H1 = Z1
    K1 = np.dot(np.dot(P1, H1), np.linalg.inv(r1 + np.dot(np.dot(H1.T, P1), H1)))
    W1 = W1 + g1 * np.dot(K1, e1)
    P1 = P1 - np.dot(np.dot(K1, H1.T), P1) + Q1

    H2 = Z2
    K2 = np.dot(np.dot(P2, H2), np.linalg.inv(r2 + np.dot(np.dot(H2.T, P2), H2)))
    W2 = W2 + g2 * np.dot(K2, e2)
    P2 = P2 - np.dot(np.dot(K2, H2.T), P2) + Q2

    H3 = Z3
    K3 = np.dot(np.dot(P3, H3), np.linalg.inv(r3 + np.dot(np.dot(H3.T, P3), H3)))
    W3 = W3 + g3 * np.dot(K3, e3)
    P3 = P3 - np.dot(np.dot(K3, H3.T), P3) + Q3

    f1 = np.dot(W1.T, Z1)
    f2 = np.dot(W2.T, Z2)
    f3 = np.dot(W3.T, Z3)

    gss1 = gs1.reshape(2, 1)
    gss2 = gs2.reshape(2, 1)
    gss3 = gs3.reshape(2, 1)

    # Estructura de los 6 estados con pesos y sigmodes

    X1 = f1 + np.dot(gss1.T, u)
    X2 = f2 + np.dot(gss2.T, u)
    X3 = f3 + np.dot(gss3.T, u)

    xd = 1.0                                                                            #Punto deseado
    yd = 2.5

    xp = X1[0,0] + d*math.cos(X3[0,0])
    yp = X2[0,0] + d*math.sin(X3[0,0])

    #error de posicion
    ex = xd - xp
    ey = yd - yp

    #PID de la nuerona x
    i_ex = (i_ex + ex)*(1/t)
    d_ex = ex - eAx
    eAx = ex
    error_x = np.array([ex,i_ex,d_ex]) #arrar con las 3 Ganancias adaptables

    #PID de la neurona Y
    i_ey = (i_ey + ey)*(1/t)
    d_ey = ey - eAy
    eAy = ey
    error_y = np.array([ey,i_ey,d_ey]) #arrar con las 3 Ganancias adaptables

    kx = neurona_x.control_u(error_x) #Control u hacer
    neurona_x.fit(ex,error_x)

    ky   = neurona_y.control_u(error_y)
    neurona_y.fit(ey,error_y,0.03)

    errores = np.array([kx,ky])

    matriz_modelo = np.array([[np.cos(theta),-d*np.sin(theta)],[np.sin(theta),d*np.cos(theta)]])
    u = np.dot(np.linalg.inv(matriz_modelo),np.array([kx,ky]))

    udata1 = u[0]
    udata2 = u[1]

    print(u)

    if t==S:

        guardar_datos()
    #np.savetxt('vectores.txt', [vector_u, vector_x, vector_y, vector_theta])
    #execfile('lectura_y_graficacion.py')
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        pub.publish(msg)
        rospy.signal_shutdown("Objetivo alcanzado")


    msg = Twist()
    msg.linear.x = u[0]
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = u[1]

    if ex < 0.01 and ey < 0.01:
        guardar_datos()
        msg.linear.x = 0
        msg.angular.z = 0
        pub.publish(msg)
        rospy.signal_shutdown("Objetivo alcanzado")

    vector_t[t-1]=t
    t = t+1


    pub.publish(msg)
    print('x=',x,' y=',y)

    vector_uv[t-1]=udata1
    vector_uw[t-1]=udata2
    vector_x[t-1]=x
    vector_y[t-1]=y
    vector_theta[t-1]=theta
    vector_errorx[t-1]=ex
    vector_errory[t-1]=ey
    vector_x_id[t-1] = X1
    vector_y_id[t-1] = X2
    vector_theta_id[t-1] = X3



def guardar_datos():

    global vector_uv
    global vector_uw

    global vector_x
    global vector_y
    global vector_theta

    global vector_errorx
    global vector_errory

    global vector_x_id
    global vector_y_id
    global vector_theta_id
    global vector_t

    # Guardar los vectores en un archivo de texto
    print('Datos guardados con exito!')

    np.savetxt("/home/robopc/npid_ws/src/npid_py/scripts/vectores.txt", [vector_uv,vector_uw, vector_x, vector_y])
    np.savetxt("/home/robopc/npid_ws/src/npid_py/scripts/vectores2.txt", [vector_theta,vector_errorx,vector_errory])
    np.savetxt("/home/robopc/npid_ws/src/npid_py/scripts/vectores3.txt", [vector_theta_id, vector_x_id, vector_y_id,vector_t])
rospy.init_node('control_py', anonymous=True)

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rate = rospy.Rate(10) # 10hz

rospy.Subscriber("/odom", Odometry, Callback)

rospy.on_shutdown(guardar_datos)

while not rospy.is_shutdown() and t < S:
    rate.sleep()


rospy.spin()
