import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import exp, trunc, sin, pi
import math
from time import time
import struct
from scipy.fft import fft, fftfreq
import scipy.io.wavfile

class Functions(object):

    def line(a, b):
        """"x(t) = at + b"""
        return lambda t: a * t + b

    def exp(a, b):
        """x(t) = b * exp(-at)"""
        return lambda t: b * exp(-a * t)
    
    def selfr(a=0, b=1):
        r = Random()
        return lambda x: (b-a)*r.rand() + a
    
    def embedr(a=0, b=1):
        return lambda x: (b-a)*np.random.rand() + a
    
    def garmonic(A, f, dt=1/1000):
        return lambda t: A*sin(2*pi*f*t*dt)

class Latex(object):
    def line(a, b):
        return r'$x(t) = %st + %s$' %(str(a), str(b))
    def exp(a, b):
        if a > 0:
            return r'$x(t) = %s e^{-%st}$'  %(str(b), str(a))
        return r'$x(t) = %s e^{%st}$'  %(str(b), str(-a))

class Model(object):       
    def Zip(T, X):
         return pd.DataFrame(list(zip(T, X)), columns=['t', 'x'])
        
    def shift(data, c):
        dt = data.copy()
        dt['x'] += c
        return dt

    def spikes(dt, n, val, d):
        data = dt.copy()
        positions = np.trunc(np.random.rand(n)*len(data)).astype(int)
        a = val - d
        b = val + d
        for p in positions:
            value = (b-a)*np.random.rand() + a
            if(np.random.rand(1) > 0.5):
                data.loc[p]['x'] = value
            else:
                data.loc[p]['x'] = -value
        return data
    
    def antiTrend(data, L):
        """
        L -- длина интервала
        """
        N = len(data)
        summ = 0
        X = [];
        for m in range(N-L):
            X.append(Model.movingAverage(data, L, m));
        T = [i for i in range(N)]
        diff = data.x - pd.Series(X)
        return pd.DataFrame(list(zip(T, diff)), columns=['t', 'x'])  
    
    def movingAverage(data, L, m):
        """
        L -- длина интервала

        """
        le = len(data)
        summ = 0
        for i in range(m, m + L):
            if (i < le):
                summ += data.iloc[i].x
            else: break;
        return summ/L
    
    def antiSpikes(data, rang):
        spiked = data.copy()
        a = rang[0]
        b = rang[1]
        spikes = []
        for index, row in spiked.iterrows(): 
            if (row.x < a or row.x > b):
                spikes.append(index)
        for index in spikes:
            if (index-1>0 and index+1<len(spikes)):
                mean = (spiked.loc[index - 1] + spiked.loc[index + 1]) / 2
                spiked.loc[index] = mean
            else:
                if (index - 1 <= 0): 
                    spiked.loc[index] = spiked.loc[index + 1]
                else: 
                    spiked.loc[index] = spiked.loc[index - 1]   
        return spiked

    def antishift(data):
        return Model.shift(data, -data.x.mean())
        

    def trend(function, N=1000, dt=1):
        """Возвращает датафрейм со столбцами t и x"""
        T = np.arange(0, N, dt)
        X = [function(t) for t in T]
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x'])
    
    def values(function, N=1000, dt=1):
        """Возвращает датафрейм со столбцами t и x"""
        T = np.arange(0, N, dt)
        X = [function(t) for t in T]
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x'])
    
    def piecesTrend(functionsAndBounds, N=1000, dt=1):
        """
        Возвращает сборный датафрейм из нескольких функций.
        Параметры: 
            functionsAndBounds -- список кортежей состоящий из функции и границ ее аргумента, например:
            
            [(lambda x: x, (1, 2)), (lambda x: 2, [1,2])]
            Границы -- кортежи или списки длины 2, например (1, 5).
            Второй элемент кортежа должен быть больше первого. Начало кортежа включается, конец -- исключается.        
        """
        T = np.arange(0, N, dt)
        X = []
        last = 0
        for FandB in functionsAndBounds:
            func = FandB[0]
            bound = FandB[1]
            
            if(len(bound) != 2) or (bound[1] - bound[0] < 0):
                raise ValueError('Uncorrent bounds')       
            
            start = bound[0]
            stop = bound[1]
            
            b = func(T[start]) - last
            X.extend([func(t) - b for t in T[start:stop]])
            last = X[len(X)-1:len(X)][0]
        
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x'])  
    
    def embedRandom(a=0, b=1, N=1000, dt=1):
        if b==None: a=-S
        T = np.arange(0, N, dt)
        X = [2*(b-a)*np.random.rand()+a for t in T]
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x'])
    
    def selfRandom(a=0, b=1, N=1000, dt=1, sid=(0,0)):
        r = Random(sid)       
        T = np.arange(0, N, dt)
        X = [(2*(b-a)*r.rand()+a) for t in T]      
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x'])          
    
    def FAdditive(functions, N=1000, dt=1):
        Dfs = []
        for f in functions:
            Dfs.append(Model.trend(f, N, dt))
        X = Dfs[0]['x'].copy()
        for i in range(1, len(Dfs)):
            X += Dfs[i]['x']
            
        T = np.arange(0, N, dt)
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x'])
    
    def FMultiplicative(functions, N=1000, dt=1):
        Dfs = []
        for f in functions:
            Dfs.append(Model.trend(f, N, dt))
        X = Dfs[0]['x'].copy()
        for i in range(1, len(Dfs)):
            X *= Dfs[i]['x']
            
        T = np.arange(0, N, dt)
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x']) 
    
    def Multiplicative(trends, N=1000, dt=1):
        X = trends[0]['x'].copy()
        for i in range(1, len(trends)):
            X *= trends[i]['x']
            
        T = np.arange(0, N, dt)
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x']) 
    
    def Additive(trends, N=1000, dt=1):
        X = trends[0]['x'].copy()
        for i in range(1, len(trends)):
            X += trends[i]['x']
            
        T = np.arange(0, N, dt)
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x']) 

class IO_RW(object):
    def display(datas, marker=None, color='Blue', titles=None, labels=None, figsize=(10,10)):
        """Принимает список датафреймов (4) и выводит их"""
        sns.set_style('whitegrid', {'grid.color':'0.1'})
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        for i in range(len(datas)):
            plt.subplot(2, 2, i + 1)
            if (titles!=None): plt.title(titles[i])
            label=None
            if (labels!=None): label = labels[i]
            sns.lineplot(data=datas[i], x='t', y='x', marker=marker, color=color, label=label)
            
    def readBinaryFile(path:str, dt=0.001, N=1000):
        try:
            file = open(path, 'rb')
            X = []
            buffer = file.read(4)
            while buffer:
                [x] = struct.unpack('f', buffer)
                X.append(x)
                buffer = file.read(4)
            T = [i*dt for i in range(N)] 
            return pd.DataFrame(list(zip(T, X)), columns=['t', 'x'])  
        finally:
            file.close()
            
    def __GetFirstChannel(data : np.ndarray):
        n = len(data)
        ret = np.zeros(n)
        for i in range(n):
            ret[i] = data[i][0]
        return ret
        
    def readWaveFile(path :str):
        """Возвращает кортеж из частоты дискретизации и первого канала"""
        freq, data = scipy.io.wavfile.read(path)
        data = IO_RW.__GetFirstChannel(data)
        return (freq, data)
    
    
    def __DuplicateChannel(data):
        n = len(data)
        res = np.zeros((n, 2))
        for i in range(n):
            res[i][0] = data[i]
            res[i][1] = data[i]
        return res
    def writeWaveFile(filename, rate, data):
        data = IO_RW.__DuplicateChannel(data)
        scipy.io.wavfile.write(filename, rate, data.astype(np.int16))
        
    def SpecDisplay(data, title=''):
        plt.subplots(1,1, figsize=(15,7))
        sns.lineplot(data=data, x='F', y='A').set_title(title)
        

class Random(object):
    def sid():
        return trunc((time()*10000000)%10**10)
    
    
    def centr(num):
        if (num <= 10000):
            return num;
        L = len(str(num))
        bounds = L - 5
        M = str(num)
        start = int(bounds/2)
        stop = int(5 + start)
        return int(M[start:stop])
        
    def rand(self):
        if(self.Sid1 * self.Sid2==0):
            self.Sid1 = Random.sid()
            self.Sid2 =  Random.sid()
        a = self.Sid2
        self.Sid2 = Random.centr(self.Sid1*self.Sid2)
        self.Sid1 = a
        return self.Sid2/100000
        
    def __init__(self, sid=(0,0)):     
        if (sid[0] == sid[1] == 0): 
            self.Sid1 = Random.sid()
            self.Sid2 = Random.sid()
        else:
            self.Sid1 = trunc(sid[0]*363269)
            self.Sid2 = trunc(sid[1]*363267 + sin(363267))  
            
from math import sqrt

    
class Analysis(object):
    
    def AutoCorrelation(data):
        T = data.t;
        X = [Analysis.ACF(data, i) for i in range(len(data))]
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x']) 
    
    def AutoCorrelation2(data):
        T = data.t;
        X = [Analysis.ACF2(data, i) for i in range(len(data))]
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x']) 
    
    def CrossCorrelation(data1, data2):
        T = data1.t;
        X = [Analysis.VCF(data1, data2, i) for i in range(len(data1))]
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x']) 
    
    def ACF(trend, L):
        meanx = trend.x.mean()
        x = trend.x

        sum1 = 0
        for k in range(len(x) - L - 1):
            sum1 += (x[k] - meanx)*(x[k+L] - meanx)
        sum2 = 0
        for k in range(len(x) - 1):
            sum2 += (x[k] - meanx)*(x[k] - meanx)
        return sum1/sum2

    def VCF(trend1, trend2, L):
        x = trend1.x
        y = trend2.x
        if (len(x) != len(y)): raise Error()

        meanx = x.mean()
        meany = y.mean()
        sum1 = 0
        for k in range(len(x) - L - 1):
            sum1 += (x[k] - meanx)*(y[k + L] - meany)
        return sum1 / len(x)

    def ACF2(trend, L):
        meanx = trend.x.mean()
        x = trend.x
        for k in range(len(x) - L - 1):
            sum1 += (x[k] - meanx)*(x[k + L] - meanx)
        return sum1 / len(x)

    
    def KDE(data, bins=20 ,figsize=(10,10)):
        plt.subplots(figsize = figsize) 
        plt.title('Гистограммы и ядерные оценки плотности')
        sns.histplot(data=data, kde=True, bins=20)
    
    def ConcatMQ(df):
        ser = pd.Series(name='MQ')
        
        columns = list(df)
        for col in columns: 
            ser = ser.append(pd.Series(Analysis.MQ(df[col]), index=[col], name='q'))
        return ser
    
    def ConcatMQD(df):
        ser = pd.Series(name='MQD')
        
        columns = list(df)
        for col in columns: 
            ser = ser.append(pd.Series(Analysis.MQD(df[col]), index=[col], name='q'))
        return ser

    def MQ(series):
        sum = 0
        for s in series:
            sum += s*s
        return sum/len(series)
    
    def MQD(series):
        return sqrt(Analysis.MQ(series))
    
    def describe(df):
#         desc = df.describe()
#         desc = desc.drop(index='25%')
#         desc = desc.drop(index='50%')
#         desc = desc.drop(index='75%')
        desc = pd.DataFrame()
        desc = desc.append(pd.Series(df.min(), name='min'))
        desc = desc.append(pd.Series(df.max(), name='max'))
        desc = desc.append(pd.Series(df.mean(), name='mean'))
        desc = desc.append(pd.Series(df.var(ddof=0), name='var'))
        #desc = desc.append(pd.Series(df.var(), name='var2'))
        desc = desc.append(pd.Series(df.std(ddof=0), name='std'))
        desc = desc.append(pd.Series(df.skew() * (df.std()**3), name='Асимметрия'))
        desc = desc.append(pd.Series(df.skew(), name='Асимм. коэф.'))
        desc = desc.append(pd.Series((df.kurtosis() + 3) * (df.std()**4), name='Эксцесс'))
        desc = desc.append(pd.Series(df.kurtosis(), name='Эксц. коэф.'))
        desc = desc.append(pd.Series(Analysis.ConcatMQ(df), name='СК'))
        desc = desc.append(pd.Series(Analysis.ConcatMQD(df), name='СКО'))
       
        return desc
    
     
    def analyse(trend, M=10):
        l = len(trend)
        q = l // M
        df = pd.DataFrame()
        start = 0
        for i in range(M):    
            end = start + q
            st = str(start) + '-' + str(end-1)
            df[st] = pd.Series(list(trend.x[start:end]))           
            start = start + q
            
        
        return Analysis.describe(df)
            
    def stationary(trend, M=10):
        df = Analysis.analyse(trend, M=M)
        df.drop(index=['min', 'max', 'std', 'Асимметрия', 'Асимм. коэф.', 
               'Эксцесс', 'Эксц. коэф.', 'СК', 'СКО'], inplace=True)
        
        df.loc['mean'] = df.loc['mean'] / df.loc['mean'].mean()
        df.loc['var'] = df.loc['var'] / df.loc['var'].mean()

        flag1 = False
        flag2 = False
        for i in df.loc['mean']:
            if abs(i) > 1.1: 
                flag1 = True
                break
        for i in df.loc['var']:
             if abs(i) > 1.1: 
                flag2 = True
                break
        
        if (flag1 or flag2 == True):
            print("Процесс не стационарен: ")
            if(flag1==True): print('по матожиданию')
            #if (flag1 and flag2 == True): print(' и ')
            if(flag2==True): print('по дисперсии')
        else:
            print("Процесс стационарен")
        return df
    

    def Furie(data, dt, window=1):
        x_results = []
        y_results = []

        f_d = (1 / (2 * dt)) / (len(data['x']) / 2) #f граничн
        f = -(1 / (2 * dt))

        half = int(len(data['x']) / 2)
        zeros_count = half * (1 - window)

        y_arr = data['x'].copy()

        for i in range(half):
            if zeros_count > 0:
                y_arr[i] = 0
                y_arr[len(data['x']) - i - 1] = 0
                zeros_count -= 1
            else:
                break

        for j in range(-half, half):
            re = 0
            im = 0

            for i in range(len(y_arr)):
                re += y_arr[i] * math.cos(2 * math.pi * j * i / len(y_arr))
                im += y_arr[i] * math.sin(2 * math.pi * j * i / len(y_arr))
 
            re /= len(y_arr)
            im /= len(y_arr)

            x_results.append(f)
            f += f_d
            y_results.append(math.sqrt(re ** 2 + im ** 2)) #амплитудный спекртр Фурье
            
           
        return pd.DataFrame(list(zip(x_results[half + 1:], y_results[half + 1:])), columns=['F', 'A']) 
    #[x_results[half + 1:], y_results[half + 1:]]
    
    #from scipy.fft import fft, fftfreq
    def FastFurie(data : np.array, dt = 0.002):
        yf = fft(data)
        N = len(data)
        xf = fftfreq(N, dt)[:N//2]
        yf = np.abs(yf[0:N//2]) / N
        return pd.DataFrame(list(zip(xf, yf)), columns=['F', 'A']) 
    
    def Diff(vector, dt=1):
        N = len(vector)
        diff = np.zeros(N)

        for i in range(N-1):
            if vector[i]==0:
                diff[i]=0
                continue
            diff[i] = (vector[i+1]-vector[i])/dt
        return diff

    def convolution(x, h, trim=True):
        M = len(h)
        N = len(x)
        y = np.zeros(N+M)
        for k in range(N + M):
            tmp = 0;
            for j in range(M): 
                if (k - j >= 0 and k - j < N):
                    tmp += x[k - j]*h[j]

            y[k] = tmp;
        if trim:
            stop = N+M-M//2
            ret = y[M//2:stop-1]
            return ret
        return y

    
    #15, 0.002, 64

    def lpf(fc, dt=0.002, m=64, mirror=True):
        d = [0.35577019, 0.2436983, 0.07211497, 0.00630165]

        fact = 2 * fc * dt
        lpw = np.zeros(m + 1)
        lpw[0] = fact

        arg = fact*np.pi

        for i in range(1, m + 1):
            lpw[i] = math.sin(arg * i) / (math.pi * i)

        lpw[m] /= 2;
        sumg = lpw[0]

        for i in range(1, m + 1):
            summ = d[0]
            arg = np.pi * i / m
            for k in range(1, 4):
                summ += 2 * d[k] * math.cos(arg * k)
            lpw[i] *= summ
            sumg += 2 * lpw[i]

        for i in range(m + 1):
            lpw[i] /= sumg

        if not mirror:
            return lpw
        left = np.flip(lpw[1:])
        return np.concatenate([left, lpw])

    def hpf(fc, dt=0.002, m=64, mirror=True):
        """Фильтр высоких частот"""
        values = Analysis.lpf(fc, dt, m, mirror)
#         loper = len(values)  
#         for k in range(loper):
#             values[k] = 1-values[k] if k == m else -values[k]
        values = -values
        values[m] += 1.0
        return values

    def bpf(f1, f2, dt=0.002, m=64, mirror=True):
        """Полосовой фильтр"""
        values1 = Analysis.lpf(f1, dt, m, mirror)
        values2 = Analysis.lpf(f2, dt, m, mirror)

        loper = len(values1)
        values2 -= values1
        return values2;

    def bsw(f1, f2, dt=0.002, m=64, mirror=True):
        """Режекторный фильтр"""
        values1 = Analysis.lpf(f1, dt, m, mirror)
        values2 = Analysis.lpf(f2, dt, m, mirror)
        
#         loper = len(values1)
#         values = np.zeros(loper)
#         for k in range(loper):
#             if (k == m):
#                 values[k] = 1.0 + values1[k] - values2[k]
#             else:
#                 values[k] = values1[k] - values2[k]

        values = values1 - values2
        values[m] += 1.0
        return values
    