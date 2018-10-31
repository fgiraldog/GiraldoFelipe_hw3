#Este codigo fue corrido con Python3

#Importacion de los paquetes a usar
import numpy as np
import matplotlib.pylab as plt
#from sklearn.decomposition import PCA #Este paquete se usa para verificar que el PCA se hizo de manera correcta (Descomentar si se quiere verificar)
import os as bash #Para poder usar bash en el codigo

#Adecuacion de la base de datos 
comando_1 = 'sed -i "s/M/1/g" WDBC.dat' 
comando_2 = 'sed -i "s/B/0/g" WDBC.dat'
bash.system(comando_1)
bash.system(comando_2)

#Importacion de los datos
datos = np.genfromtxt('WDBC.dat', delimiter = ',', usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31)) #Los parametros de cada paciente
info_paciente = np.genfromtxt('WDBC.dat', delimiter = ',', usecols = (0,1)) #La informacion de cada paciente

#Funcion para poder normalizar los datos y hacer que cada parametro tenga una media = 0 y una varianza = 1
def normalizacion(datos_1):
	N = np.shape(datos_1)[1]
	for i in range(0, N):
		datos_1[:,i] = (datos_1[:,i]-np.mean(datos_1[:,i]))/np.std(datos_1[:,i])

	return datos_1	

#Funcion para poder determinar la matriz de covarianza de los datos
def matriz_covarianza(datos_1):
	N = np.shape(datos_1)[1]
	cov = np.zeros((N,N))
	for i in range(0, N):
		for j in range(0, N):
			cov[i,j] = np.sum((datos_1[:,i]-np.mean(datos_1[:,i]))*(datos_1[:,j]-np.mean(datos_1[:,j])))/(datos_1[:,0].size-1)

	return cov

#Calculo de la matriz de covarianza de los datos normalizados
matriz_cov = matriz_covarianza(normalizacion(datos))
matriz_numpy = np.cov(np.transpose(normalizacion(datos))) #Aqui se hace uso de los paquetes de numpy para probar que la implementacion propia es correcta

#Funcion para verificar que la matriz propia esta correcta y tener que mirar numero por numero (esto fue completamente opcional)
def verificacion_cov(matriz_mia,matriz_numpy):
	N = np.shape(matriz_mia)[1]
	contador = 0
	for i in range(0, N):
		for j in range(0, N):
			pos_mia = matriz_mia[i,j]
			pos_numpy = matriz_numpy[i,j]
			if(pos_numpy < 0 and pos_mia > 0):
				contador = contador + 1
			elif(pos_numpy > 0 and pos_mia < 0):
				contador = contador + 1
			elif(abs(pos_numpy)-abs(pos_mia) > 10**-5):
				contador = contador + 1
	if contador > 0:
		print('La matriz de covarianza propia esta mal')
	else:
		print('La matriz de covarianza propia esta bien')
			
#Impresion en la consola de la matriz de covarianza propia
print('La matriz de covarianzas es:')
print(matriz_cov)

#print('--------------------------------------------------------------------------------')
#verificacion_cov(matriz_cov,matriz_numpy) #Si se quiere comprobar que la matriz esta bien, descomentar estas lineas

print('--------------------------------------------------------------------------------')

#Calculo de los valores propios y los vectores propios de la matriz de covarianza
valores_propios, vectores_propios = np.linalg.eig(matriz_cov)

#Impresion de cada valor propio con su respectivo vector propio
for i in range(0,len(valores_propios)):
	print ('El valor propio y el vector propio ', i+1, ' son:')
	print (valores_propios[i])
	print (vectores_propios[:,i])

print('--------------------------------------------------------------------------------')

#Comentario de los componentes mas importantes en este analisis
print('Segun los resultados recien presentados, para cada autovalor y cada autovector, se puede ver que los componentes mas importantes son PC1 y PC2, es decir el primer autovector y el segundo autovector. Esto se puede confirmar haciendo uso del porcentaje de la varianza que estos atribuyen a los datos. Entonces, PC1 es responsable del ', valores_propios[0]*100/np.sum(valores_propios), '% de la varianza, y PC2 es responsable del ', valores_propios[1]*100/np.sum(valores_propios), '% de la varianza. Ahora bien, en cuanto a los parametros de cada auto vector, en PC1 se ve claramente que la mayoria de estos tiene el mismo peso, pero el mayor se encuentra en la posicion [7] con un valor de 0.26085376 y en la hoja de los datos, esto corresponde a concave points (number of concave portions of the contour). Ahora bien, en cuanto al autovector correspondiente a PC2, se puede apreciar que se cumple el mismo fenomeno que cada parametro tiene el mismo peso, pero el mayor se encuentra en la posicion [9] con un valor de 0.36657547, y en la hoja de datos corresponde a fractal dimension ("coastline approximation" - 1).')

#Definicion de los vectores propios para PC1 y PC2
vector_prop_1 = vectores_propios[:,0]
vector_prop_2 = vectores_propios[:,1]

#Proyeccion de los datos en los nuevos ejes o vectores (los datos tambien estan normalizados)
PC1 = np.dot(vector_prop_1,np.transpose(normalizacion(datos)))
PC2 = np.dot(vector_prop_2,np.transpose(normalizacion(datos)))

#Funcion para poder determinar que puntos pertenencen a un diagnostico maligno y que puntos pertenecen a un diagostico benigno
def benigno_maligno(b_n,pc1,pc2):
	benigno_pc1 = []
	maligno_pc1 = []
	benigno_pc2 = []
	maligno_pc2 = []
	for i in range(0,len(pc1)):
		if b_n[i] == 1:
			maligno_pc1.append(pc1[i])
			maligno_pc2.append(pc2[i])
		else:
			benigno_pc1.append(pc1[i])
			benigno_pc2.append(pc2[i])
						 
	return np.array(maligno_pc1), np.array(maligno_pc2), np.array(benigno_pc1), np.array(benigno_pc2)

#Determinacion de diagnosticos benignos y malignos
maligno_PC1, maligno_PC2, benigno_PC1, benigno_PC2 = benigno_maligno(info_paciente[:,1], PC1, PC2)

#Grafica de PC1 y PC2 con los diagnoticos en diferentes colores
plt.figure()
plt.scatter(maligno_PC1,maligno_PC2, c = 'g', label = 'Diagnostico Maligno')
plt.scatter(benigno_PC1,benigno_PC2, c = 'b', label = 'Diagnostico Benigno')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.legend()
plt.savefig('GiraldoFelipe_PCA.pdf')

print('--------------------------------------------------------------------------------')

#Comentario acerca de los resultados obtenidos
print('Para termiar con este punto, es importante analizar los resultados obtenidos de la proyeccion de los datos en los ejes de PC1 y PC2. Entonces, como se puede ver en la grafica que ha sido guardada en su computador GiraldoFelipe_PCA.pdf, los diagnosticos benignos estan agrupados a la izquierda del grafico, y tienen una varianza menor en cuanto al eje PC1 que aquella para los diagnosticos malignos. Asi mismo, se puede ver que los diagnostico malignos tienen valores mayores en cuanto al eje PC1 y estan dispersos en cuanto a ambos ejes. Pero de manera general, se puede apreciar una clara diferencia en la agrupacion de estos datos, en donde los diagnosticos benignos estan hacia la izquierda y los malignos esta corridos un poco hacia la derecha. Asi pues, con esto resultados se puede decir que el analisis de PCA, para este caso probo ser exitoso debido a que se pudo diferenciar los casos en donde el diagnostico es benigno o maligno, lo cual podria ayudar a los paciente e incluso a los doctores a tener una mayor precaucion con diagnosticos que se encuentren en la zona de los malignos. ')


#Si se quiere comprobar que el PCA fue hecho de manera correcta, se puede descomentar esta seccion. Aqui no se hace diferencia entre los diagnosticos pero se ve que laforma obtenida en la grafica es igual a aquella encontrada en la implementacion propia
#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform((datos))

#plt.figure()
#plt.scatter(principalComponents[:,0], principalComponents[:,1], label = 'Sklearn')
#plt.ylabel('PC2')
#plt.xlabel('PC1')
#plt.legend()
#plt.show()
