import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d


datos_signal = np.genfromtxt('signal.dat', delimiter = ',')
datos_incompletos = np.genfromtxt('incompletos.dat', delimiter = ',')

datos_signal_x = datos_signal[:,0]
datos_signal_y = datos_signal[:,1]

# datos signal
#plt.figure()
#plt.plot(datos_signal_x,datos_signal_y, label = 'Signal')
#plt.xlabel('$x$')
#plt.ylabel('$y$')
#plt.legend()
#plt.show()

def fourier_discreta(f,m,sampling):
	razon_n_m = []
	transformada = []
	k = np.arange(0,m)
	for n in range (0,m):
		razon_n_m.append(float(n)*(sampling)/float(m)) 
		transformada.append(np.sum(f*np.exp(-1j*2*np.pi*k*n/float(m))))

	for i in range(0,m):
		if razon_n_m[i] >= (float(sampling/2)):
			razon_n_m[i] = razon_n_m[i] - sampling

	razon_n_m = np.array(razon_n_m)
	transformada = np.array(transformada)	

	return razon_n_m,transformada

sampling_rate = 1/(datos_signal_x[1]-datos_signal_x[0])
signal_x_trans, signal_y_trans = fourier_discreta(datos_signal_y,len(datos_signal_y),sampling_rate)

transformada = fft(datos_signal_y)
frecuencia = fftfreq(len(datos_signal_y),(datos_signal_x[1]-datos_signal_x[0]))

print(signal_x_trans,frecuencia)
#print(signal_y_trans-transformada)
#plt.figure()
#plt.plot(signal_x_trans, np.real(signal_y_trans), label = 'Transformada de Fourier propia')
#plt.plot(frecuencia, np.real(transformada), label = 'Scipy')
#plt.xlabel('$Frecuencia (Hz)$')
#plt.ylabel('$Amplitud$')
#plt.legend()
#plt.show()

def filtro(frecuencias,transformadas,n):
	for i in range(0,len(frecuencias)):
		if abs(frecuencias[i])>n:
			transformadas[i] = 0

	return transformadas

#print(filtro(signal_x_trans, signal_y_trans, 1000),signal_x_trans)
signal_y_filtrada1000 = np.real(ifft(filtro(signal_x_trans, signal_y_trans, 1000)))

plt.figure()
plt.plot(datos_signal_x,datos_signal_y, label = 'Signal (Sin filtro)')
plt.plot(datos_signal_x,signal_y_filtrada1000, label = 'Signal (Con filtro)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()

# datos incompletos

def interpolacion(datos_viejos, x_nuevos):
	cuadratica = interp1d(datos_viejos[:,0],datos_viejos[:,1], kind='quadratic')
	cubica = interp1d(datos_viejos[:,0],datos_viejos[:,1], kind='cubic')

	y_cuadratica = cuadratica(x_nuevos)
	y_cubica = cubica(x_nuevos)

	return y_cuadratica, y_cubica


x = np.linspace(0.000390625,0.028515625,512)
inter_q,inter_c = interpolacion(datos_incompletos,x)

#plt.figure()
#plt.plot(datos_incompletos[:,0],datos_incompletos[:,1], 'o',label = 'Incompletos')
#plt.plot(x,inter_q, label = 'Cuadratica')
#plt.plot(x,inter_c, label = 'Cubica')
#plt.legend()
#plt.show()

cuadratica_x_trans, cuadratica_y_trans = fourier_discreta(inter_q,len(inter_q))
cubica_x_trans, cubica_y_trans = fourier_discreta(inter_c,len(inter_c))


#plt.figure()
#plt.plot(signal_x_trans, signal_y_trans, label = 'Transformada de Fourier (Signal)')
#plt.plot(cuadratica_x_trans, cuadratica_y_trans, label = 'Transformada de Fourier (Cuadratica)')
#plt.plot(cubica_x_trans, cubica_y_trans, label = 'Transformada de Fourier (Cubica)')
#plt.xlabel('$Frecuencia (Hz)$')
#plt.ylabel('$Amplitud$')
#plt.legend()
#plt.show()

signal_y_filtrada500 = np.real(ifft(filtro(signal_x_trans, signal_y_trans, 500)))
cuadratica_y_filtrada1000 = np.real(ifft(filtro(cuadratica_x_trans, cuadratica_y_trans, 1000)))
cuadratica_y_filtrada500 = np.real(ifft(filtro(cuadratica_x_trans, cuadratica_y_trans, 500)))
cubica_y_filtrada1000 = np.real(ifft(filtro(cubica_x_trans, cubica_y_trans, 1000)))
cuadratica_y_filtrada500 = np.real(ifft(filtro(cubica_x_trans, cubica_y_trans, 5000)))

#plt.figure()
#plt.plot(datos_signal_x,datos_signal_y, label = 'Signal (Sin filtro)')
#plt.plot(datos_signal_x,signal_y_filtrada1000, label = 'Signal (1000 Hz)')
#plt.plot(datos_signal_x,signal_y_filtrada500, label = 'Signal (500 Hz)')
#plt.xlabel('$x$')
#plt.ylabel('$y$')
#plt.legend()
#plt.show()
