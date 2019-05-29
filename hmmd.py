from __future__ import print_function, division
from builtins import range

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from numpy.polynomial.hermite import hermval, hermfit
from scipy.stats import norm

# Calculo del logaritmo del productorio de las probabilidades de emision para un estado concreto
# x son las observaciones asociadas a ese estado
# inicio es el punto, dentro de la secuencia de observaciones, en la que entramos en el estado
# duracion es la duracion del estado en cuestion
# m es el numero de coeficientes que utilizaremos para ajustar a minimos cuadrados la secuencia de observacion
# El ajuste se realiza utilizando polinomios de Hermite
"""def logb(x, inicio, duracion, m):
    coef, aux = hermfit(np.arange(inicio, inicio+duracion+1), x, m, full=True)
    try:
        res = aux[0][0]
    except IndexError:
        res = 0.

    if res != 0:
        sigma = np.sqrt(res/duracion)
        output = np.sum(norm.logpdf(x, hermval(np.arange(inicio, inicio+duracion+1), coef), sigma))
#        y = hermval(np.arange(inicio,inicio+duracion+1),coef)
#        x1 = norm.cdf(x+0.5,y,sigma)
#        x2 = norm.cdf(x-0.5,y,sigma)
#        output = np.sum(np.log(x1-x2))
    else:
        output = 0.

#    if output > 0:
#        print(output)
    return output"""


class HMM:
    def __init__(self, N):
        self.N = N # numero de estados ocultos
        
        
    def logb(self, x, inicio, duracion, m):
        coef, aux = hermfit(np.arange(inicio, inicio+duracion+1), x, m, full=True)
        try:
            res = aux[0][0]
        except IndexError:
            res = 0.

        if res != 0:
            sigma = np.sqrt(res/duracion)
            output = np.sum(norm.logpdf(x, hermval(np.arange(inicio, inicio+duracion+1), coef), sigma))
#        y = hermval(np.arange(inicio,inicio+duracion+1),coef)
#        x1 = norm.cdf(x+0.5,y,sigma)
#        x2 = norm.cdf(x-0.5,y,sigma)
#        output = np.sum(np.log(x1-x2))
        else:
            output = 0.

        return output
    
#    def logb(self, x, inicio, duracion, m):
#        output = 0
#        for i in range(duracion):
#            output = output + self.b[x[i]-self.xmin]
#        return output
#        return np.log(0.2)*duracion

    def fit(self, x, max_iter=30):
        t0 = datetime.now()
        
        np.random.seed(123)
        
        #calculamos la longitud de la secuencia de observacion, T
        T = len(x)
        
#        self.xmin=np.min(x)
#        xmax=np.max(x)
#        self.b = np.zeros(xmax-self.xmin+1)
#        for i in range(T):
#            self.b[x[i]-self.xmin] += 1
#        for i in range(xmax-self.xmin+1):
#            self.b[i] = self.b[i]/T
#        self.b = np.log(self.b)


#        pi = np.ones(self.N)/self.N # Distribucion inicial de estados
#        pi = np.zeros(self.N)
#        pi[0] = 1
        pi = np.random.random(self.N)
        pi = pi/np.sum(pi)
        pi = np.log(pi)

        A = np.random.random((self.N, self.N)) # matriz de probabilidades de transicion entre estados
        for i in range(self.N):
            A[i][i] = 0
#        for i in range(self.N):
#            for j in range(i+1):
#                A[i][j] = 0.
        A = A/A.sum(axis=1, keepdims=True)
#        for i in range(self.N):
#            A[self.N-1][i] = 0
#        A = np.zeros((self.N, self.N))
#        for i in range(self.N-1):
#            A[i][i+1] = 1
        
        A = np.log(A)

        # Fijamos valores iniciales para Dmax, Dmin y m
        # Dmin es la duracion minima para cada estado
        # Dmax es la duracion maxima para cada estado
        # m es el numero de coeficientes utilizados para realizar el ajuste en cada estado
        Dmin = np.zeros(self.N, np.int16)
        Dmax = np.zeros(self.N, np.int16)
        m = np.zeros(self.N, np.int16)
        for i in range(self.N):
            Dmin[i] = 0
            Dmax[i] = T-1
            m[i] = 5
            
#        Dmin[0] = 43 #10-1
#        Dmax[0] = 43 #10-1
#        Dmin[1] = 27 #35-1
#        Dmax[1] = 27 #35-1
#        Dmin[2] = 16 #24-1
#        Dmax[2] = 16 #24-1
#        Dmin[3] = 6 #41-1
#        Dmax[3] = 6 #41-1
#        Dmin[4] = 6 #47-1
#        Dmax[4] = 6 #47-1
#        Dmin[5] = 58 #52-1
#        Dmax[5] = 58# 52-1
#        Dmin[6] = 57 #11-1
#        Dmax[6] = 57 #11-1

#        Dmin[0] = 5
#        Dmax[0] = 15
#        Dmin[1] = 30
#        Dmax[1] = 40
#        Dmin[2] = 20
#        Dmax[2] = 30
#        Dmin[3] = 35
#        Dmax[3] = 45
#        Dmin[4] = 43
#        Dmax[4] = 53
#        Dmin[5] = 45
#        Dmax[5] = 55
#        Dmin[6] = 5
#        Dmax[6] = 15

#        Dmin[0] = 10-1
#        Dmax[0] = 10-1
#        Dmin[1] = 25-1
#        Dmax[1] = 25-1
#        Dmin[2] = 15-1
#        Dmax[2] = 15-1

        #definimos una distribucion de probabilidad no informada para las duraciones
        p = np.zeros((self.N, T))
        for i in range(self.N):
            p[i][Dmin[i]:Dmax[i]+1] = 1./(Dmax[i]-Dmin[i]+1)
        
        p = np.log(p)

        #En logP almacenaremos la probabilidad de la observacion, dado el modelo, para las distintas iteraciones
        #en el proceso de aprendizaje. Realmente se almacena su logaritmo
        logP = []

        #Iniciamos el bucle de entrenamiento
        for it in range(max_iter):
            print("it:", it)

            #Calculamos el logaritmo de los alphas
            lalpha = np.full((T, self.N), np.NINF)
            for t in range(Dmin.min(), T):
                print("Iterando alpha:", t)
                for i in range(self.N):
                    if t >= Dmin[i] and t <= Dmax[i]:
                        lalpha[t][i] = pi[i]+p[i][t]+self.logb(x[0:t+1], 0, t, m[i])
                        for d in range(Dmin[i], t):
                            for j in range(self.N):
                                if i != j:
                                    lalpha[t][i] = np.logaddexp(lalpha[t][i], lalpha[t-d-1][j]+A[j][i]+p[i][d]+self.logb(x[t-d:t+1], t-d, d, m[i]))
                    elif t > Dmax[i]:
                        for d in range(Dmin[i], Dmax[i]+1):
                            for j in range(self.N):
                                if i != j:
                                    lalpha[t][i] = np.logaddexp(lalpha[t][i], lalpha[t-d-1][j]+A[j][i]+p[i][d]+self.logb(x[t-d:t+1], t-d, d, m[i]))

            #A continuacion calculamos el logaritmo de los betas
            lbeta = np.full((T, self.N), np.NINF)
            lbeta[-1] = 0.
            for t in range(T-Dmin.min()-2, -1, -1):
                print("Iterando beta:", t)
                for i in range(self.N):
                    for j in range(self.N):
                        if t >= T-Dmax[j]-1 and t < T-Dmin[j]-1:
                            if i != j:
                                aux = np.NINF
                                for d in range(Dmin[j], T-2-t+1):
                                    aux = np.logaddexp(aux, lbeta[t+d+1][j]+p[j][d]+self.logb(x[t+1:t+d+2], t+1, d, m[j]))
                                lbeta[t][i] = np.logaddexp(lbeta[t][i], A[i][j]+aux)
                        elif t < T-Dmax[j]-1:
                            if i != j:
                                aux = np.NINF
                                for d in range(Dmin[j], Dmax[j]+1):
                                    aux = np.logaddexp(aux, lbeta[t+d+1][j]+p[j][d]+self.logb(x[t+1:t+d+2], t+1, d, m[j]))
                                lbeta[t][i] = np.logaddexp(lbeta[t][i], A[i][j]+aux)

            suma = np.full(T,np.NINF)
            for t in range(T):
                for i in range(self.N):
                    suma[t] = np.logaddexp(suma[t],lalpha[t][i]+lbeta[t][i])
                    
            print(suma)

            #Calculamos el logaritmo de la probabilidad de la observacion, dado el modelo
            logPit = np.NINF
            for i in range(self.N):
                logPit = np.logaddexp(logPit, lalpha[-1][i])

            #lo añadimos al vector logP. Su representacion grafica sirve para ver como evoluciona el entrenamiento
            logP.append(logPit)

            #Actualizamos las probabilidades iniciales de los estados
            for i in range(self.N):
                sum = np.NINF
                for d in range(Dmin[i], Dmax[i]+1):
                    sum = np.logaddexp(sum, lbeta[d][i]+p[i][d]+self.logb(x[0:d+1], 0, d, m[i]))
                pi[i] = pi[i]+sum-logPit

            #Tanto en el calculo de las probabilidades de transicion entre estados como las duraciones, la
            #expresion en el denominador es la misma y la calculamos aqui
#            den = np.full(self.N, np.NINF)
#            for i in range(self.N):
#                for t in range(T):
#                    den[i] = np.logaddexp(den[i], lalpha[t][i]+lbeta[t][i])

            #Calculamos las probabilidades de transicion entre estados
            for i in range(self.N):
                numA = np.full(T, np.NINF)
                for j in range(self.N):
                    print("Calculando A:", i, j)
                    if i != j:
                        for t in range(T):
                            for d in range(Dmin[j], Dmax[j]+1):
                                if d < t:
                                    numA[j] = np.logaddexp(numA[j], lalpha[t-d-1][i]+A[i][j]+p[j][d]+lbeta[t][j]+self.logb(x[t-d:t+1], t-d, d, m[j]))
#                den = np.log(np.sum(np.exp(numA)))
                den = np.NINF
                for j in range(self.N):
                    den = np.logaddexp(den, numA[j])
                for j in range(self.N):
                    if den == np.NINF:
                        A[i][j] = np.NINF
                    else:
                        A[i][j] = numA[j]-den

            #Calculamos las probabilidades de duracion para los distintos estados
            for j in range(self.N):
                nump = np.full(T, np.NINF)
                for d in range(Dmin[j], Dmax[j]+1):
                    print("Calculando p:", j, d)
                    nump[d] = pi[j]+p[j][d]+lbeta[d][j]+self.logb(x[0:d+1], 0, d, m[j])
                    for t in range(T):
                        if d < t:
                            for i in range(self.N):
                                nump[d] = np.logaddexp(nump[d], lalpha[t-d-1][i]+A[i][j]+p[j][d]+lbeta[t][j]+self.logb(x[t-d:t+1], t-d, d, m[j]))
#                den = np.log(np.sum(np.exp(nump)))
                den = np.NINF
                for d in range(Dmin[j], Dmax[j]+1):
                    den = np.logaddexp(den, nump[d])
                for d in range(Dmin[j], Dmax[j]+1):
                    p[j][d] = nump[d]-den

            print("Fit duration iteration:", it, (datetime.now()-t0))
            print("Probabilidad total: ", logPit)




        print("A:", np.exp(A))
        print("pi:", np.exp(pi))

        print("Fit duration:", (datetime.now() - t0))

        plt.plot(logP)
        plt.show()

"""    def likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi*self.B[:,x[0]]
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
        return alpha[-1].sum()

    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])

    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))

    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = self.pi*self.B[:,x[0]]
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * self.B[j, x[t]]
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states"""

def ajusta():
    homedir = os.getenv('HOME')
    mitdbdir = homedir + '/database'
    recordname = mitdbdir +'/105'

#record es un objeto de tipo wfdb.io.record.Record
#Le estamos diciendo que el resultado es en 16 bits por muestra y que no transforme los datos
#a unidades físicas. Queremos directamente el valor de las muestras
    record = wfdb.rdrecord(recordname, return_res=16, physical=False)

#sig_len contiene la longitud del registro en número de muestras
#    sig_len = record.sig_len
#annotation contiene un objeto de tipo wfdb.io.annotation.Annotation
    annotation = wfdb.rdann(recordname, 'xqrs')
#obs es el vector de observaciones. Es un array de 220 muestras
    x = record.d_signal[annotation.sample[0]-87:annotation.sample[0]+133, 0]
#    x = record.d_signal[annotation.sample[0]-20:annotation.sample[0]+30, 0]
    
    hmm = HMM(7)
    hmm.fit(x, max_iter=5)



if __name__ == '__main__':
    ajusta()
