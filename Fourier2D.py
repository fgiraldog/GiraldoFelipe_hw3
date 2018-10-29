#Este codigo fue corrido con Python3

#Importacion de los paquetes a usar
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm

#Importacion de la imagen a un array de numpy
imagen = np.array(plt.imread('arbol.png'))

#Transformacion de fourier en 2D para la imagen
imagen_transformada = fftpack.fft2(imagen)

#Como fft2 no entrega la transformacion centrada, con este paquete se centra la funcion
transformada_centrada = fftpack.fftshift(imagen_transformada)

#Grafica de la transformacion de fourier en 2D para la imagen en escala ln
plt.figure()
plt.imshow(np.log(np.abs(transformada_centrada)))
plt.colorbar()
plt.title('Transformada de Fourier (Escala logaritimica)')
plt.savefig('GiraldoFelipe_FT2D.pdf')

#Funcion para poder filtrar la transformada, aqui se intentaron varios filtros, pasa bajas, pasa altas, pero este fue el que probo tener mejores resultados
def filtro(matrix):
	for i in range(0,np.shape(matrix)[0]):
		for j in range(0,np.shape(matrix)[1]):
			if (104 < i < 152) and (104 < j < 152):
				matrix[i,j] = matrix[i,j]
			elif matrix[i,j] > 175:
				matrix[i,j] = 0
				
	return matrix

#Filtro de la imagen
transformada_centrada2 = transformada_centrada.copy()
transformada_filtrada = filtro(transformada_centrada2)

#Grafica de la transformacion de fourier en 2D para la imagen ya filtrada, en escala LogNorm
plt.figure()
plt.imshow(np.abs(transformada_filtrada), norm = LogNorm(vmin=4))
plt.title('Transformada de Fourier Filtrada')
plt.colorbar()
plt.savefig('GiraldoFelipe_FT2D_filtrada.pdf')

#Recuperacion de la imagen ya filtrada
imagen_nueva = fftpack.ifft2(fftpack.ifftshift(transformada_filtrada))

#Grafica de la imagen ya filtrada
plt.figure()
plt.imshow(np.abs(imagen_nueva),plt.cm.gray)
plt.title('Imagen Filtrada')
plt.savefig('GiraldoFelipe_Imagen_filtrada.pdf')



