import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm

imagen = plt.imread('arbol.png')


plt.figure()
plt.imshow(imagen,plt.cm.gray)
plt.title('Imagen original')
plt.show()

im_fft = fftpack.fft2(imagen)

print(im_fft)

plt.figure()
plt.imshow(np.log(np.abs(im_fft)))
plt.colorbar()
plt.title('Transformada de Fourier')
plt.show()

def filtro(matrix):
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if np.log(matrix[i,j]) > 7.75:
				matrix[i,j] = 0
	return matrix

transformada_filtrada = filtro(im_fft)

plt.figure()
plt.imshow(transformada_filtrada,norm=LogNorm())
plt.title('Transformada de Fourier Filtrada')
plt.show()

imagen_nueva = fftpack.ifft2(transformada_filtrada)

plt.figure()
plt.imshow(np.abs(imagen_nueva),plt.cm.gray)
plt.colorbar()
plt.title('Imagen Filtrada')
plt.show()



