import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm

imagen = np.array(plt.imread('arbol.png'))

imagen_transformada = fftpack.fft2(imagen)

transformada_centrada = fftpack.fftshift(imagen_transformada)

plt.figure()
plt.imshow(np.log(np.abs(transformada_centrada)))
plt.colorbar()
plt.title('Transformada de Fourier (Escala logaritimica)')
plt.savefig('GiraldoFelipe_FT2D.pdf')

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
plt.colorbar()
plt.savefig('GiraldoFelipe_FT2D_filtrada.pdf')

imagen_nueva = fftpack.ifft2(fftpack.ifftshift(transformada_filtrada))

plt.figure()
plt.imshow(np.abs(imagen_nueva),plt.cm.gray)
plt.title('Imagen Filtrada')
plt.savefig('GiraldoFelipe_Imagen_filtrada.pdf')



