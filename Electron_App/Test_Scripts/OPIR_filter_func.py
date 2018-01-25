import gdal, osr
import numpy as np
import random
import math

from numpy import genfromtxt



def First_Order(array,array_past):
    #First Order IIR filter

    #Need way of reading this in automatically
    decay = .65
    x_dim = 400
    y_dim = 400
    array_temp = np.zeros((y_dim, x_dim))



    for x in range(0, x_dim, 1):

            for y in range(0, y_dim, 1):
                array_temp[y][x] = decay * array[y][x] - (1 - decay) * array_past[y][x]
    return array_temp



def Weighted_Moving_Avg(array_past):
    #Weight average over last 3 frames

    # Need way of reading this in automatically
    weight = 1
    x_dim = 400
    y_dim = 400
    array_temp=np.zeros((y_dim,x_dim))

    for i in range(1, n_samples):
        weight += pow(.5, i)


        for x in range(0, x_dim, 1):

            for y in range(0, y_dim, 1):

                for k in range(0, n_samples):
                    array_temp[y][x] += array_past[k][y][x]
                    # array[y][x]+=np.power(decay2,k)*array_past2[k][y][x]

                array_temp[y][x] = array_temp[y][x] / (weight * n_samples)
    return array_temp

##EXAMPLE READ IN
if __name__ == "__main__":
    # Need way of reading this in automatically
    x_dim = 400
    y_dim = 400
    number_snapshots = 900
    n_samples = 3 #5

    array_past=np.zeros((y_dim,x_dim)) #Need a history for filters. Some way of keeping track in function?
    array_past_avg = np.zeros((n_samples,y_dim,x_dim)) #need more history for avg func
    for j in range(0,number_snapshots):
        readRaster = "OPIR_test/datacube_slice_" + "%03d" % j + ".csv" #file name for .CSV
        array = genfromtxt(readRaster, delimiter=',') # set to numpy array
        #FILTER 1st Order
        outarray = First_Order(array,array_past) #First Order IIR filter
        array_past = outarray # set for next iteration of filter



        #SECONDARY FILTER IF STATEMENT?

        #for k in range(n_samples - 1, 0, -1): #Set for next iteration of filter
        #    array_past_avg[k][y][x] = array_past_avg[k - 1][y][x]  # shift past n_samples
        #array_past_avg[0][y][x] = array[y][x]  # set time 0

        #outarray = Weighted_Moving_Avg(array_past_avg) #weight avg filter


        #PLOTTING???

