#Este codigo fue corrido con Python3
#Prueba data science
#Importacion de los paquetes a usar
import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d

#Importacion de los datos signal.dat y incompletos.dat
datos_signal = np.genfromtxt('signal.dat', delimiter = ',')
datos_incompletos = np.genfromtxt('incompletos.dat', delimiter = ',')

datos_signal_x = datos_signal[:,0]
datos_signal_y = datos_signal[:,1]

#Grafica de los datos en signal.dat
plt.figure()
plt.plot(datos_signal_x,datos_signal_y, label = 'Signal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.savefig('GiraldoFelipe_signal.pdf')

#Funcion para calcular la transformada discreta de fourier
def fourier_discreta(f,m,sampling):
	razon_n_m = []
	transformada = []
	k = np.arange(0,m)
	for n in range (0,m):
		razon_n_m.append(float(n)*(sampling)/float(m)) #Aqui se deja la razon en terminos de la frecuencia
		transformada.append(np.sum(f*np.exp(-1j*2*np.pi*k*n/float(m))))

	for i in range(0,m):
		if razon_n_m[i] >= (float(sampling/2)): #Este paso hace el trabajo del shift en fftfreq 
			razon_n_m[i] = razon_n_m[i] - sampling

	razon_n_m = np.array(razon_n_m)
	transformada = np.array(transformada)	

	return razon_n_m,transformada

#Calculo de la transformada discreta de fourier para los datos de signal.dat
sampling_rate = 1/(datos_signal_x[1]-datos_signal_x[0])
signal_x_trans, signal_y_trans = fourier_discreta(datos_signal_y,len(datos_signal_y),sampling_rate)


#Grafica de la trasformada discreta de fourier para los datos de signal.dat
plt.figure()
plt.plot(signal_x_trans, np.real(signal_y_trans), label = 'Transformada de Fourier propia')
plt.xlabel('$Frecuencia (Hz)$')
plt.ylabel('$Amplitud$')
plt.legend()
plt.savefig('GiraldoFelipe_TF.pdf')

print('Para la transformada de Fourier, no se uso el paquete de fftfreq.')

print('--------------------------------------------------------------------------------')

#Comentario acerca de las frecuencias principales
print('Las frecuencias principales de la senal, claramente son aquellas que tienen la mayor amplitud. Entonces, teniendo en cuenta la grafica que acaba de ser guardada en su computador GiraldoFelipe_TF.pdf, se puede apreciar que las mayores amplitudes se dan en las frecuencias bajas, es decir menores a 1000 Hz. Mas arriba de estas frecuencias se puede ver amplitudes pequenas que claramente son las que generan el ruido en la senal. Entonces, es por esta razon que para poder filtrar la senal se debe implementar un filtro pasabajas.')

print('--------------------------------------------------------------------------------')

#Funcion para poder filtrar la senal
def filtro(frecuencias,transformadas,n):
	for i in range(0,len(frecuencias)):
		if abs(frecuencias[i])>n:
			transformadas[i] = 0

	return transformadas

#Filtro de la senal signal.dat con la f_c = 1000 Hz. Aqui se hace el filtro y se recupera la senal de una vez
signal_y_filtrada1000 = ifft(filtro(signal_x_trans, signal_y_trans, 1000))

#Grafica de la senal filtrada con f_c  = 1000 Hz
plt.figure()
plt.plot(datos_signal_x,datos_signal_y, label = 'Signal (Sin filtro)')
plt.plot(datos_signal_x,np.real(signal_y_filtrada1000), label = 'Signal (Con filtro)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.savefig('GiraldoFelipe_filtrada.pdf')


#Comentario acerca de la transformada discreta de fourier en datos incompletos
print('La transformada de Fourier no se puede hacer en los datos incompletos, debido a que la tasa de muestreo no es uniforme para cada dato. Es decir, el espaciamiento entre los datos no es el mismo para toda la muestra. Esto, hace que sea imposible crear la transformada de fourier en terminos de la frecuencia, que es basicamente lo mas importante para poder filtrar una senal con ruido. Esto se puede apreciar de una mejor manera con la funcion tanto propia como fftfreq, en donde, para poder encontrar las frecuencias asociadas a cada amplitud se necesita la tasa de muestreo de los datos, y esta no puede variar en el set de datos.')

print('--------------------------------------------------------------------------------')

#Funcion para poder interpolar los datos incompletos
def interpolacion(datos_viejos, x_nuevos):
	cuadratica = interp1d(datos_viejos[:,0],datos_viejos[:,1], kind='quadratic')
	cubica = interp1d(datos_viejos[:,0],datos_viejos[:,1], kind='cubic')

	y_cuadratica = cuadratica(x_nuevos)
	y_cubica = cubica(x_nuevos)

	return y_cuadratica, y_cubica

#Interpolacion cuadratica y cubica de los datos incompletos
x = np.linspace(0.000390625,0.028515625,512)
inter_q,inter_c = interpolacion(datos_incompletos,x)

#Transformada discreta de fourier para los datos interpolados
cuadratica_x_trans, cuadratica_y_trans = fourier_discreta(inter_q,len(inter_q),1/(x[1]-x[0]))
cubica_x_trans, cubica_y_trans = fourier_discreta(inter_c,len(inter_c),1/(x[1]-x[0]))
signal_x_trans1, signal_y_trans1 = fourier_discreta(datos_signal_y,len(datos_signal_y),sampling_rate)

#Grafica con tres subplots para las tres transformadas, signal.dat, interpolacion cuadratica, interpolacion cubica
plt.figure(figsize =[11,9])
plt.subplots_adjust(hspace=0.7)
plt.subplot(311)
plt.plot(signal_x_trans1, np.real(signal_y_trans1), label = 'Transformada de Fourier (Signal)')
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

#Comentario acerca de las diferencias en las tres transformadas
print('En la grafica que acaba de ser guardada en su computador GiraldoFelipe_TF_interpola.pdf, se pueden apreciar algunas diferencias entre las transformadas de fourier tanto para las funciones interpoladas como para signal.dat. Entre las funciones interpoladas, existe una gran diferencia para los datos con interpolacion cuadratica ya que tiene mucho mas ruido en las frecuencias bajas. Asi mismo, la transformada de signal.dat tiene mucho mas ruido en las frecuencias altas que la funcion interpolada cubica, y cuenta con un pico alrededor de 500-550Hz, el cual la interpolacion cubica no cuenta. Entonces, estas son algunas de las diferencias entre estas transformaciones, y esto puede ser verificado haciendo uso de la grafica GiradoFelipe_2Filtros.pdf. En esta, se puede ver que la funcion filtrada para la interpolacion cuadratica es muy diferente a las otras dos, y para signal.dat, esta se vuelve mas similar a la funcion interpolada de manera cubica cuando se le hace el filtro con la frecuencia de corte de 500 Hz, debido a que se elimina el pico que se menciono anteriormente.')

#Filtro de las tres senales con una f_c = 500 Hz. Aqui se filtra la senal y se vuelve a recuperar
signal_y_filtrada500 = ifft(filtro(signal_x_trans, signal_y_trans, 500))
cuadratica_y_filtrada500 = ifft(filtro(cuadratica_x_trans, cuadratica_y_trans, 500))
cubica_y_filtrada500 = ifft(filtro(cubica_x_trans, cubica_y_trans, 500))

#Filtro de las dos senales faltantes con una f_c = 1000 Hz. Aqui se filtra la senal y se vuelve a recuperar
cuadratica_y_filtrada1000 = ifft(filtro(cuadratica_x_trans, cuadratica_y_trans, 1000))
cubica_y_filtrada1000 = ifft(filtro(cubica_x_trans, cubica_y_trans, 1000))

#Grafica con dos subplots, uno para cada filtro en donde se ven las tres senales siendo analizadas
plt.figure(figsize =[11,9])
plt.subplots_adjust(hspace=0.7)
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
