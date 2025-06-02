import numpy as np
import matplotlib.pyplot as plt

# Ruta del archivo de texto
ruta1 = "/home/robopc/npid_ws/src/npid_py/scripts/vectores.txt"
ruta2 = "/home/robopc/npid_ws/src/npid_py/scripts/vectores2.txt"
# Cargar datos desde el archivo de texto
data1 = np.loadtxt(ruta1)
data2 = np.loadtxt(ruta2)

# Asignar los datos a los vectores correspondientes
vector_uv = data1[0]
vector_uw = data1[1]
vector_x = data1[2]
vector_y = data1[3]
vector_theta = data2[0]
vector_errorx = data2[1]
vector_errory = data2[2] 
#vector_error = data[4]


# Graficar los vectores
print(np.shape(vector_uv))
print(np.shape(vector_errory))



plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
plt.plot(vector_uv,label='Uvlineal')
plt.plot(vector_uw,label='UvAngular')
plt.title('Vector U')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(vector_x,label='x')
plt.title('Vector X')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(vector_y,label='y')
plt.title('Vector Y')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(vector_theta,label='theta')
plt.title('Vector Theta')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(vector_errorx,label='errorx')
plt.plot(vector_errory,label='errory')
plt.title('Error X y Y')
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.savefig('graficas')


plt.show()
plt.close
