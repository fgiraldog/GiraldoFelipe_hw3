import numpy as np
import matplotlib.pylab as plt

datos = np.genfromtxt('WDBC.dat', delimiter = ',', usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))


print (datos)
def matriz_covarianza(datos_1):
	
	N = np.shape(datos_1)[1]
	cov = np.zeros((N,N))
	for i in range(0, N):
		for j in range(0, N):
			cov[i,j] = np.sum((datos_1[:,i]-np.mean(datos_1[:,i]))*(datos_1[:,j]-np.mean(datos_1[:,j])))/(datos_1[:,0].size-1)

	return cov

matriz_cov = matriz_covarianza(datos)

matriz_numpy = np.cov(np.transpose(datos))

correccion = abs(matriz_cov) - abs(matriz_numpy)

print(matriz_cov)

print(matriz_numpy)

print(correccion)

