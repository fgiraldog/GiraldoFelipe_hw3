import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
import os

comando_1 = 'sed -i "s/M/1/g" WDBC.dat' 
comando_2 = 'sed -i "s/B/0/g" WDBC.dat'
os.system(comando_1)
os.system(comando_2)

datos = np.genfromtxt('WDBC.dat', delimiter = ',', usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))
info_paciente = np.genfromtxt('WDBC.dat', delimiter = ',', usecols = (0,1))

def normalizacion(datos_1):
	N = np.shape(datos_1)[1]
	for i in range(0, N):
		datos_1[:,i] = (datos_1[:,i]-np.mean(datos_1[:,i]))/np.std(datos_1[:,i])

	return datos_1	

def matriz_covarianza(datos_1):
	N = np.shape(datos_1)[1]
	cov = np.zeros((N,N))
	for i in range(0, N):
		for j in range(0, N):
			cov[i,j] = np.sum((datos_1[:,i]-np.mean(datos_1[:,i]))*(datos_1[:,j]-np.mean(datos_1[:,j])))/(datos_1[:,0].size-1)

	return cov

matriz_cov = matriz_covarianza(normalizacion(datos))
matriz_numpy = np.cov(np.transpose(normalizacion(datos)))

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
			elif(abs(pos_numpy)-abs(pos_mia) > 10**-3):
				contador = contador + 1
	if contador > 0:
		print('La matriz de covarianza propia esta mal')
	else:
		print('La matriz de covarianza propia esta bien')
			

print('La matriz de covarianzas es:')
print(matriz_cov)

print('--------------------------------------------------------------------------------')

verificacion_cov(matriz_cov,matriz_numpy)

print('--------------------------------------------------------------------------------')

valores_propios, vectores_propios = np.linalg.eig(matriz_cov)

for i in range(0,len(valores_propios)):
	print ('El valor propio y el vector propio ', i+1, ' son:')
	print (valores_propios[i])
	print (vectores_propios[:,i])

print('--------------------------------------------------------------------------------')

print('Segun los resultados recien presentados, para cada autovalor y cada autovector, se puede ver que los componentes o parametros mas importantes son PC1 y PC2, es decir el primer autovector y el segundo autovector. Esto se puede confirmar haciendo uso del porcentaje de la varianza que estos atribuyen a los datos. Entonces, PC1 es responsable del ', valores_propios[0]*100/np.sum(valores_propios), '% de la varianza, y PC2 es responsable del ', valores_propios[1]*100/np.sum(valores_propios), '% de la varianza.')

vector_prop_1 = vectores_propios[:,0]
vector_prop_2 = vectores_propios[:,1]

PC1 = np.dot(vector_prop_1,np.transpose(normalizacion(datos)))
PC2 = np.dot(vector_prop_2,np.transpose(normalizacion(datos)))

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

maligno_PC1, maligno_PC2, benigno_PC1, benigno_PC2 = benigno_maligno(info_paciente[:,1], PC1, PC2)

plt.figure()
plt.scatter(maligno_PC1,maligno_PC2, label = 'Diagnostico Maligno')
plt.scatter(benigno_PC1,benigno_PC2, label = 'Diagnostico Benigno')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.legend()
plt.savefig('GiraldoFelipe_PCA.pdf')

print('--------------------------------------------------------------------------------')

print('Para termiar con este punto, es importante analizar los resultados obtenidos de la proyeccion de los datos en los ejes de PC1 y PC2. Entonces, como se puede ver en la grafica que ha sido guardada en su computador GiraldoFelipe_PCA.pdf, los diagnosticos benignos estan agrupados a la derecha del grafico, y tienen una varianza menor en cuanto al eje PC1 que aquella para los diagnosticos malignos. Asi mismo, se puede ver que los diagnostico malignos tienen valores mayores en cuanto al eje PC1 y estan dispersos en cuanto a ambos ejes. Pero de manera general, se puede apreciar una clara diferencia en la agrupacion de estos datos, en donde los diagnosticos benignos estan hacia la izquierda y los malignos esta corridos un poco hacia la derecha. Asi pues, con esto resultados se puede decir que el analisis de PCA, para este caso probo ser exitososo debido a que se pudo diferenciar los casos en donde el diagnostico es benigno o maligno, lo cual podria ayudar a los paciente e incluso a los doctores a tener una mayor precuacion con diagnosticos que se encuentren en la zona de los malignos. ')


#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform((datos))

#plt.figure()
#plt.scatter(principalComponents[:,0], principalComponents[:,1], label = 'Sklearn')
#plt.ylabel('PC2')
#plt.xlabel('PC1')
#plt.legend()
#plt.show()
