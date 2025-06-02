#!/usr/bin/env python2
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
from npid import nPID

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
neurona_x = nPID(3,0.5)
neurona_y = nPID(3,0.3)
#Inicializacion para Guardar variables
S=1300 #Tiempo del ciclo
vector_uv=np.zeros(S)
vector_uw=np.zeros(S)
vector_x=np.zeros(S)
vector_y=np.zeros(S)
vector_theta=np.zeros(S)
vector_errorx=np.zeros(S)
vector_errory=np.zeros(S)
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

    x = msg_in.pose.pose.position.x
    y = msg_in.pose.pose.position.y
    quater = np.zeros(4)
    quater[0] = msg_in.pose.pose.orientation.w
    quater[1] = msg_in.pose.pose.orientation.x
    quater[2] = msg_in.pose.pose.orientation.y
    quater[3] = msg_in.pose.pose.orientation.z
    theta = math.atan2(2*(quater[0]*quater[3]+quater[1]*quater[2]),1-2*(quater[2]*quater[2]+quater[3]*quater[3])) #yaw en robotica

    xd = 3.0 #Punto deseado
    yd = 3.0

    xp = x + d*math.cos(theta)
    yp = y + d*math.sin(theta)

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
    #vector_error[t-1,:]=errores

def guardar_datos():

    global vector_uv
    global vector_uw
    global vector_x
    global vector_y
    global vector_theta
    global vector_errorx
    global vector_errory

    # Guardar los vectores en un archivo de texto
    print('Datos guardados con exito!')
	
	
    np.savetxt("/home/robopc/npid_ws/src/npid_py/scripts/vectores.txt", [vector_uv,vector_uw, vector_x, vector_y])
    np.savetxt("/home/robopc/npid_ws/src/npid_py/scripts/vectores2.txt", [vector_theta,vector_errorx,vector_errory])

rospy.init_node('control_py', anonymous=True)

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rate = rospy.Rate(10) # 10hz

rospy.Subscriber("/odom", Odometry, Callback)

rospy.on_shutdown(guardar_datos)

while not rospy.is_shutdown() and t < S:
    rate.sleep()


rospy.spin()
