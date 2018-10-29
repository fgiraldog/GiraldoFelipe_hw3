import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d


datos_signal = np.genfromtxt('signal.dat', delimiter = ',')
datos_incompletos = np.genfromtxt('incompletos.dat', delimiter = ',')

datos_signal_x = datos_signal[:,0]
datos_signal_y = datos_signal[:,1]

# datos signal
plt.figure()
plt.plot(datos_signal_x,datos_signal_y, label = 'Signal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.savefig('GiraldoFelipe_signal.pdf')

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

plt.figure()
plt.plot(signal_x_trans, np.real(signal_y_trans), label = 'Transformada de Fourier propia')
plt.xlabel('$Frecuencia (Hz)$')
plt.ylabel('$Amplitud$')
plt.legend()
plt.savefig('GiraldoFelipe_TF.pdf')

print('Para la transformada de Fourier, no se uso el paquete de fftfreq.')

print('Las frecuencias principales de la señal, claramente son aquellas que tienen la mayor amplitud. Entonces, teniendo en cuenta la grafica que acaba de ser guardada en su computador GiraldoFelipe_TF.pdf, se puese apreciar que las mayores amplitudes se dan en las frecuencias bajas, es decir menores a 1000 Hz. Mas arriba de estas frecuencias se puede ver amplitudes pequeñas que claramente son las que generan el ruido en la señal. Entonces, es por esta razon que para poder filtrar la señal se debe implementar un filtro pasabajas.')

def filtro(frecuencias,transformadas,n):
	for i in range(0,len(frecuencias)):
		if abs(frecuencias[i])>n:
			transformadas[i] = 0

	return transformadas

signal_y_filtrada1000 = ifft(filtro(signal_x_trans, signal_y_trans, 1000))

plt.figure()
plt.plot(datos_signal_x,datos_signal_y, label = 'Signal (Sin filtro)')
plt.plot(datos_signal_x,np.real(signal_y_filtrada1000), label = 'Signal (Con filtro)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.savefig('GiraldoFelipe_filtrada.pdf')

# datos incompletos

print('La transformada de Fourier no se puede hacer en los datos incompletos, debido a que la tasa de muestreo no es uniforme para cada dato. Es decir, el espaciamiento entre los datos no es el mismo para toda la muestra. Esto, hace que sea imposible crear la transformada de fourien en términos de la frecuencia, que es basicamente lo más importante para poder filtrar una señal con ruido. Esto se puede apreciar de una mejor manera con la funcion tanto propia como fftfreq, en donde, para poder encontrar las frecuencias asociadas a cada amplitud se necesita la tasa de muestreo de los dato, y esta no puede varias en el set de datos.')

def interpolacion(datos_viejos, x_nuevos):
	cuadratica = interp1d(datos_viejos[:,0],datos_viejos[:,1], kind='quadratic')
	cubica = interp1d(datos_viejos[:,0],datos_viejos[:,1], kind='cubic')

	y_cuadratica = cuadratica(x_nuevos)
	y_cubica = cubica(x_nuevos)

	return y_cuadratica, y_cubica


x = np.linspace(0.000390625,0.028515625,512)
inter_q,inter_c = interpolacion(datos_incompletos,x)

cuadratica_x_trans, cuadratica_y_trans = fourier_discreta(inter_q,len(inter_q),sampling_rate)
cubica_x_trans, cubica_y_trans = fourier_discreta(inter_c,len(inter_c),sampling_rate)


plt.figure()
plt.subplot(311)
plt.plot(signal_x_trans, np.real(signal_y_trans), label = 'Transformada de Fourier (Signal)')
plt.xlabel('$Frecuencia (Hz)$')
plt.ylabel('$Amplitud$')
plt.legend()
plt.subplot(312)
plt.plot(cuadratica_x_trans, np.real(cuadratica_y_trans), label = 'Transformada de Fourier (Cuadratica)')
plt.xlabel('$Frecuencia (Hz)$')
plt.ylabel('$Amplitud$')
plt.legend()
plt.subplot(313)
plt.plot(cubica_x_trans, np.real(cubica_y_trans), label = 'Transformada de Fourier (Cubica)')
plt.xlabel('$Frecuencia (Hz)$')
plt.ylabel('$Amplitud$')
plt.legend()
plt.savefig('GiraldoFelipe_TF_interpola.pdf')

signal_y_filtrada500 = ifft(filtro(signal_x_trans, signal_y_trans, 500))
cuadratica_y_filtrada1000 = ifft(filtro(cuadratica_x_trans, cuadratica_y_trans, 1000))
cuadratica_y_filtrada500 = ifft(filtro(cuadratica_x_trans, cuadratica_y_trans, 500))
cubica_y_filtrada1000 = ifft(filtro(cubica_x_trans, cubica_y_trans, 1000))
cubica_y_filtrada500 = ifft(filtro(cubica_x_trans, cubica_y_trans, 500))

plt.figure()
plt.subplot(211)
plt.plot(datos_signal_x, np.real(signal_y_filtrada1000), label = 'Signal ($F_c$=1000 Hz)')
plt.plot(x,np.real(cuadratica_y_filtrada1000), label = 'Cuadratica ($F_c$=1000 Hz)')
plt.plot(x,np.real(cubica_y_filtrada1000), label = 'Cubica ($F_c$=1000 Hz)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.subplot(212)
plt.plot(datos_signal_x, np.real(signal_y_filtrada500), label = 'Signal ($F_c$=500 Hz)')
plt.plot(x,np.real(cuadratica_y_filtrada500), label = 'Cuadratica ($F_c$=500 Hz)')
plt.plot(x,np.real(cubica_y_filtrada500), label = 'Cubica ($F_c$=500 Hz)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.savefig('GiraldoFelipe_2Filtros.pdf')
