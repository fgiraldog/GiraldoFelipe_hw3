import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm

imagen = plt.imread('arbol.png')

print(imagen)

plt.figure()
plt.imshow(imagen,plt.cm.gray)
plt.title('Imagen original')
plt.show()

im_fft = fftpack.fft2(imagen)

print(im_fft)

#im_fft[64,64] = 0
#im_fft[192,192] = 0
#im_fft[246,232] = 0
#im_fft[10,24] = 0

imagen_nueva = fftpack.ifft2(im_fft)
plt.figure()
plt.imshow(np.abs(im_fft),plt.cm.gray)
plt.colorbar()
plt.title('Fourier transform')
plt.show()
