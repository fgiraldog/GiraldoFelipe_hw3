import numpy as np
import matplotlib.pylab as plt
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
		datos[:,i] = datos[:,i]/np.std(datos[:,i])

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
correccion = abs(matriz_cov) - abs(matriz_numpy)

print('La matriz de covarianzas es:')
print(matriz_cov)

#print(matriz_numpy)

#print(correccion)

valores_propios, vectores_propios = np.linalg.eig(matriz_cov)

#for i in range(0,len(valores_propios)):
#	print ('El valor propio y el vector propio ', i+1, ' son:')
#	print (valores_propios[i])
#	print (vectores_propios[:,i])

#print ('Los valores propios son: ', max(valores_propios))

#print ('Los vectores propios son: ', vectores_propios)

vector_prop_1 = vectores_propios[:,0]
vector_prop_2 = vectores_propios[:,1]

PC1 = np.dot(vector_prop_1,np.transpose(datos))
PC2 = np.dot(vector_prop_2,np.transpose(datos))

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
plt.show()



