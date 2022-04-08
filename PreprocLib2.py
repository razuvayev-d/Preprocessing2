import numpy as np
import os

import scipy.fft
from PIL import Image
import struct
import matplotlib.pyplot as plt
from Task1 import *
from scipy import signal
import warnings

warnings.filterwarnings('ignore')


class MImage:
  
    def __init__(self, path: str = None, arr: np.array = None):
        if (path != None):
            self.img = Image.open(path).convert('L') 
            self.arr = np.array(self.img)#.astype(np.uint16)
        elif(arr != None):
            self.img = Image.fromarray(arr)
            self.arr = arr  
   

    def OpenJpg(self, path : str):
        self.img = Image.open(path).convert('L') 
        self.arr = np.array(self.arr)
    
    def OpenXcr(self, path: str, offset: int, size: tuple):
        file=open(path, "rb")
        data = file.read()
        cuts = data[offset:]
        floatarr = struct.unpack('>'+'H'*size[0]*size[1], cuts)
        self.arr = np.array(floatarr).reshape(size[0], size[1]).astype(np.uint16)
        self.img = Image.fromarray(self.arr)
        self.ToGray()
        
    def SaveJpg(self, path: str):
        self.img.save(path)

    def Shift(self, c: int):
        n = len(self.arr)
        m = len(self.arr[0])
        res = np.zeros((n, m)).astype(np.uint16) #чтобы было возможно 255+
        for i in range(0, n):
            for j in range(0, m):
                res[i, j] =  self.arr[i, j] + c
        self.arr = res
        self.img = Image.fromarray(self.arr)

    def Multi(self, c: float):
        n = len(self.arr)
        m = len(self.arr[0])
        for i in range(0, n):
            for j in range(0, m):
                self.arr[i, j] *= c
        self.img = Image.fromarray(self.arr)
        
    def Show(self):
        self.img = Image.fromarray(self.arr.astype(np.uint8))
        return self.img
    
    def ToGray(self):
        maximum = np.max(self.arr)
        minimum = np.min(self.arr)
        diff = maximum - minimum
        
        n = len(self.arr)
        m = len(self.arr[0])
        res = np.zeros((n, m))
        i = 0       
        for row in self.arr:
            res[i] = np.array([np.floor((y - minimum) * 255 / diff) for y in row])
            i+=1
            
        self.arr = res.astype(np.uint8)
        self.img = Image.fromarray(self.arr).convert('L') 
        
    def Resize(self, scale, resample_type = Image.BILINEAR ):
        """
        resample_type values: [Image.NEAREST, Image.BILINEAR]
        """
        new_height = int(len(self.arr) * scale)
        new_width = int(len(self.arr[0]) * scale)
        
        new_image = self.img.resize((new_width, new_height), resample_type)
        
        self.img = new_image
        self.arr = np.array(self.img)
        
    def Rotate(self, k=1):
        self.arr = np.rot90(self.arr, k)
        self.img = Image.fromarray(self.arr)
        
    def Negative(self):
        L = np.max(self.arr)
        res = self.arr.copy()
        i = 0
        for row in res:
            res[i] = np.array([L - r - 1 for r in row])
            i += 1
        self.arr = res
        self.img = Image.fromarray(self.arr)
        
    def Log_transformation(self, c):
        if (c <= 0): 
            raise "Transformation error"
        i = 0
        n = len(self.arr)
        m = len(self.arr[0])
        res = np.zeros((n, m))
        for row in self.arr:
            self.arr[i] = np.array([np.uint16(c*np.log10(r + 1)) for r in row])#.astype(np.uint16)
            i += 1
        self.img = Image.fromarray(self.arr)
        
    def Gamma_transformation(self, c, gamma):
        i = 0
        for row in self.arr:
            self.arr[i] = np.array([np.uint16(c*np.power(r, gamma)) for r in row])#.astype(np.uint16)
            i += 1
        self.img = Image.fromarray(self.arr)
        
    def Hist(self):
        hist = self.img.histogram()
        return hist
        #sns.lineplot(y = hist, x= np.arange(0, len(hist)))
    def Cdf(self):
        
        hist = self.img.histogram()
        cdf = np.cumsum(hist)
        #fig, axs = plt.subplots(1, 1, figsize=(10,10))
        #sns.lineplot(y = cdf, x= np.arange(0, len(cdf)))
        return cdf
    
    def equalize_image(self):
        new_arr = self.arr.copy()
        cdf = self.Cdf()
        cdf_min = cdf[cdf != 0].min()
        
        n = len(self.arr)
        m = len(self.arr[0])
        
        for x in range(n):
            for y in range(m):
                new_arr[x, y] = round(
                    (cdf[self.arr[x, y]] - cdf_min) * 255.0 / (n * m - 1)
                )

        self.arr = new_arr
        self.img = Image.fromarray(self.arr)
        #self.ToGray()
        #eq = ToGray(cdf)

    def PrintHistAndCdf(self):
        #plt.rcParams['figure.figsize'] = (20,7)
        fig, axs = plt.subplots(1, 2, figsize=(20,10))
        hist = self.Hist()
        cdf = self.Cdf()
        x = np.arange(0, len(hist))
        axs[0].plot(x, hist)
        axs[1].plot(x, cdf)
        # plt.subplot(1, 2, 1)
        # sns.lineplot(x=x, y=hist)
        # plt.subplot(1, 2, 2)
        # sns.lineplot(x=x, y= cdf)
    def Consistent(self):
        self.img = Image.fromarray(self.arr)
        
    def Copy(self):
        arr2 = self.arr.copy()
        new = MImage()
        new.arr = arr2
        new.img = Image.fromarray(arr2)
        return new
                
def PlotTwoImages(img1, img2, figsize=(20,10)):
   
    fig, axs = plt.subplots(1, 2, figsize=figsize)
   
    axs[0].imshow(img1.arr, cmap='gray', vmin=0, vmax=255.)
    axs[0].axis('off')
    axs[0].set_title('Оригинальное изображение')
    
    axs[1].imshow(img2.arr, cmap='gray', vmin=0, vmax=255.)
    axs[1].axis('off')
    axs[1].set_title('Обработанное изображение')
    
def Diff(vector, dt=1):
    N = len(vector)
    diff = np.zeros(N)

    for i in range(N-1):
        if vector[i]==0:
            diff[i]=0
            continue
        diff[i] = (vector[i+1]-vector[i])/dt
    return diff

def display(datas, marker=None, color='Blue', titles=None, labels=None, figsize=(10,10)):
        """Принимает список датафреймов (4) и выводит их"""
        sns.set_style('whitegrid', {'grid.color':'0.1'})
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        for i in range(len(datas)):
            plt.subplot(2, 2, i + 1)
            if (titles!=None): plt.title(titles[i])
            label=None
            if (labels!=None): label = labels[i]
            plt.plot(datas[i], marker=marker, color=color, label=label)
            
def Supressor(image: MImage, filt):
    res = np.zeros_like(image.arr)
    i = 0
    for row in image.arr:
        res[i] = signal.convolve(row, filt, mode='same')
        i+=1
    return res 


import  random
class Noise:
    @staticmethod
    def SaltAndPepper(image: MImage, saltProb= 0.01 , pepperProb = 0.01):
        arr = image.arr
        n = len(arr)
        m = len(arr[0])
        for i in range(n):
            for j in range(m):
                if random.random() <= saltProb:
                    arr[i][j] = 255
                elif random.random() <= pepperProb:
                    arr[i][j] = 0
        image.Consistent()

    @staticmethod
    def Random(image: MImage, probability = 0.01, scale = (0, 10)):
        arr = image.arr
        n = len(arr)
        m = len(arr[0])
        for i in range(n):
            for j in range(m):
                if random.random() <= probability:
                    arr[i][j] += random.uniform( scale[0],  scale[1])
        image.Consistent()

    def Mix(image: MImage, randomProb = 0.5, SPProb = 0.5, saltProb= 0.01 , pepperProb = 0.01, probability = 0.01, scale = (0, 10)):
        arr = image.arr
        n = len(arr)
        m = len(arr[0])
        for i in range(n):
            for j in range(m):
                if random.random() <= randomProb:
                    if random.random() <= probability:
                        arr[i][j] += random.uniform( scale[0],  scale[1])
                elif random.random() <= SPProb:
                    if random.random() <= saltProb:
                        arr[i][j] = 255
                    elif random.random() <= pepperProb:
                        arr[i][j] = 0
        image.Consistent()
        
    
class Filters:
    @staticmethod
    def Mean(image: MImage, mask_range = (3,3)):
        (m, n) = mask_range
        arr = image.arr
        M = len(arr)
        N = len(arr[0])
        result = np.copy(image.arr)
        for i in range(0, M):
            for j in range(0, N):
                result[i][j] = Filters.__GetMaskArr(arr, i, j, m, n).mean()
        return result
    
    @staticmethod
    def Median(image: MImage, mask_range = (3,3)):
        (m, n) = mask_range
        arr = image.arr
        M = len(arr)
        N = len(arr[0])
        result = np.copy(image.arr)
        for i in range(0, M):
            for j in range(0, N):
                result[i][j] = np.median(Filters.__GetMaskArr(arr, i, j, m, n))
        return result
    
    @staticmethod
    def __GetMaskArr(arr, x, y, m, n):
        """Возвращает маску в массиве arr с центром с координатами (x,y) и размером m*n"""
        startx = max(x - m//2, 0)
        stopx = min(x + m//2 + 1, len(arr))
        starty = max(y - n//2, 0)
        stopy = min(y + n//2 + 1, len(arr[0]))
        res = []      
        for i in range(startx, stopx):
            for j in range(starty, stopy):
                res.append(arr[i, j])
        return np.array(res)
    
class Correlation:
    def AutoCorrelation(data):       
        return [Correlation.ACF(data, i) for i in range(len(data))]
    
    def AutoCorrelation2(data):
        T = data.t;
        X = [Analysis.ACF2(data, i) for i in range(len(data))]
        return pd.DataFrame(list(zip(T, X)), columns=['t', 'x']) 
    
    def CrossCorrelation(data1, data2):
        
        X = [Correlation.VCF(data1, data2, i) for i in range(len(data1))]
        return X
    
    def ACF(trend, L):
        meanx = trend.mean()
        x = trend

        sum1 = 0
        for k in range(len(x) - L - 1):
            sum1 += (x[k] - meanx)*(x[k+L] - meanx)
        sum2 = 0
        for k in range(len(x) - 1):
            sum2 += (x[k] - meanx)*(x[k] - meanx)
        return sum1/sum2

    def VCF(trend1, trend2, L):
        x = trend1
        y = trend2
        if (len(x) != len(y)): raise Error()

        meanx = x.mean()
        meany = y.mean()
        sum1 = 0
        for k in range(len(x) - L - 1):
            sum1 += (x[k] - meanx)*(y[k + L] - meany)
        return sum1 / len(x)

    
class Fourier:
    @staticmethod
    def Direct(vector, Trim=True, Amp = True):  
        N = len(vector)
        length = N //2 if Trim else N
        data_new = np.zeros(length)
        ret = np.zeros(length)
        
        if Amp:
            for n in range(length):
                re = np.sum(vector * np.cos((2 * math.pi * n * np.arange(N)) / N))
                im = np.sum(vector * np.sin((2 * math.pi * n * np.arange(N)) / N))
                re /= N
                im /= N
                ret[n] = math.sqrt(re ** 2 + im ** 2)
            return ret
        else:
            for n in range(length):
                re = np.sum(vector * np.cos((2 * math.pi * n * np.arange(N)) / N))
                im = np.sum(vector * np.sin((2 * math.pi * n * np.arange(N)) / N))
                re /= N
                im /= N
                ret[n] = re + im
            return ret
    @staticmethod
    def Inverse(vector, Trim=True):
        N = len(vector)
        length = N //2 if Trim else N
        ret = np.zeros(length)

        for n in range(length):
            re = np.sum(vector * np.cos((2 * math.pi * n * np.arange(N)) / N))
            im = np.sum(vector * np.sin((2 * math.pi * n * np.arange(N)) / N))
            re /= N
            im /= N
            ret[n] = re + im
        return ret
    @staticmethod
    def F2d(matrix):
        res = np.zeros_like(matrix)

        n = len(matrix)
        for i in range(n):
            res[i] = Fourier.Direct(matrix[i], Trim=False, Amp=False)
        rot = res.transpose()
        n = len(rot)
        res2 = np.zeros_like(rot)
        for i in range(n):
            res2[i] = Fourier.Direct(rot[i], Trim=False)
        return res2
    @staticmethod
    def iF2d(matrix):
        res = np.zeros_like(matrix)

        n = len(matrix)
        for i in range(n):
            res[i] = Fourier.Inverse(matrix[i], Trim=False)
        rot = res.transpose()
        n = len(rot)
        res2 = np.zeros_like(rot)
        for i in range(n):
            res2[i] = Fourier.Inverse(rot[i], Trim=False)
        return res2
    
    @staticmethod
    def Comp(data):
        summ = np.zeros_like(data)
        n = len(data)
        m = len(data[0])
        for i in range(n):
            for j in range(m):
                summ[i][j] = data[i][j].imag + data[i][j].real
        return summ
    
    @staticmethod
    def Resize(fft2, scale)
        N = len(fft2)
        M = len(fft2[0])
        nN = round(N*scale)
        nM = round(M*scale)
        b = np.full((nN, nM), 0+0j )

        for i in range(0, N):
            for j in range(0, M):
                b[i][j] = fft2[i][j]
        return b