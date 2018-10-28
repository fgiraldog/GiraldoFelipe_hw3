import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm

imagen = plt.imread('arbol.png')

imagen_transformada = fftpack.fft2(imagen)

transformada_centrada = fftpack.fftshift(imagen_transformada)

print(np.shape(transformada_centrada))

plt.figure()
plt.imshow(np.abs(transformada_centrada), norm = LogNorm(vmin=4))
plt.colorbar()
plt.title('Transformada de Fourier')
plt.show()

def filtro(matrix):
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if (104 < i < 152) and (104 < j < 152):
				matrix[i,j] = matrix[i,j]
			elif matrix[i,j] > 175:
				matrix[i,j] = 0
				
	return matrix

transformada_centrada2 = transformada_centrada.copy()
transformada_filtrada = filtro(transformada_centrada2)

plt.figure()
plt.imshow(np.abs(transformada_filtrada), norm = LogNorm(vmin=4))
plt.title('Transformada de Fourier Filtrada')
plt.show()

imagen_nueva = fftpack.ifft2(fftpack.ifftshift(transformada_filtrada))

plt.figure()
plt.imshow(np.abs(imagen_nueva),plt.cm.gray)
plt.title('Imagen Filtrada')
plt.show()



