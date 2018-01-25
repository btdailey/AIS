import gdal, osr
import numpy as np
import random
import math

from numpy import genfromtxt
import matplotlib.pyplot as plt

#Plotting algorithm
def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):

    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def main(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):
    reversed_arr = array[::-1] # reverse array so the tif looks like the array
    array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array) # convert array to raster




if __name__ == "__main__":
    #OK, Goal is to read in .csv files

    #Automatically read in some of these? (x_dim, y_dim, number??)
    decay = .65
    n_samples=3 #5
    number_snapshots = 90 #900
    x_dim = 400 #400
    y_dim = 400 #400
    array_past=np.zeros((y_dim,x_dim))
    array_past2 = np.zeros((n_samples,y_dim,x_dim))
    rasterOrigin = (-123.25745, 45.43013)
    pixelWidth = 100
    pixelHeight = 100
    avg_val=[]
    avg_val10 = np.zeros((y_dim,x_dim))
    value_array=[]
    diff_array=[]
    weight = 1
    weight_array=[]
    weight_array.append(weight)
    max_val=0
    min_val=300
    for i in range(1, n_samples):
        weight += pow(.5, i)
        weight_array.append(pow(.5,i))


    print weight
    print weight_array
    counts=0
    num_cut=0
    cut_val=0
    array_counts=[0 for i in range(256)]
    for j in range(0,number_snapshots):
        readRaster = "OPIR_test/Data_sets/Set8/datacube_slice_" + "%03d" % j + ".csv"
        array = genfromtxt(readRaster, delimiter=',')
        if(j==0): #read in zeroth array
            for m in range(0,x_dim,1):
                for mm in range(0,y_dim,1):
                    value_array.append(array[mm][m])
                    array_counts[int(array[mm][m])]+=1
                    if(array[mm][m]<255):
                        counts+=1
                    if(array[mm][m] >max_val):
                        max_val=array[mm][m]
                    if (array[mm][m] < min_val):
                        min_val = array[mm][m]
            for m in range(0,255):
                num_cut+=array_counts[m]
                if(num_cut > .7*counts):
                    cut_val=m
                    cut_val = int(100 * math.ceil(cut_val / 100.0))
                    spread = 255-cut_val
                    spread = int(math.floor(255/spread))

                    break

        #print "counts, num_cut, m",counts,num_cut,cut_val

        #print 'max,min',max_val,min_val
        if(j<10):
            avg_val10+=array
        if(j==10):
            avg_val10 = avg_val10/10

        if (j < 3):
            newRasterfn = "OPIR_test/WAfiltered_OPIR_" + "%03d" % j + ".jpeg"
            main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)
            #print 'array',array
            for k in range(n_samples - 1, 0, -1):
                array_past2[k] = array_past2[k-1]
            array_past2[0] = array
            #print 'array_last', array_past2[0]
            continue

        else:
            # FILTER!!!!!
            #for x in range(0, x_dim, 1):

            #    for y in range(0, y_dim, 1):
            #print 'FILTER!!!!!'
            #print 'array_last',array_past2[2]
            avg_val = np.zeros((y_dim, x_dim))
            for k in range(n_samples - 1, 0, -1):

                array_past2[k] = array_past2[k - 1]  # shift past n_samples
                avg_val+=array_past2[k]
            array_past2[0] = array  # set time 0
            avg_val+=array_past2[0]
            avg_val= avg_val / n_samples

            #print j, array_past2[0, 100, 100], array_past2[1, 100, 100], array_past2[2, 100, 100], avg_val[100, 100]



            array = np.zeros((y_dim,x_dim))
            for k in range(0, n_samples):
                array += array_past2[k]*weight_array[k]
                # array[y][x]+=np.power(decay2,k)*array_past2[k][y][x]
            #print 'array_avg',array
            array = array/(weight)# * n_samples)

            for m in range(0, x_dim, 1):  # loop through read in array
                for mm in range(0, y_dim, 1):  # loop through read in array

                    if (array[mm][m] - cut_val < 0.):  # Pixel value is less than cut off. Would give negative number, so set to zero.
                        array[mm][m] = 0.
                    else:
                        array[mm][m] = spread * (array[mm][m] - cut_val)  # How far above the cut off is the value? Use the spread const
                                                                        # to make sure there is large separation between
                                                                        # just above cut off and much higher than cutoff


            #print avg_val[100,100],avg_val10[100,100],array[100,100]
            diff_array = 10*(avg_val - array)
            #print j, array_past2[0, 100, 100], array_past2[1, 100, 100], array_past2[2, 100, 100], array[100,100]
            #array_past2[0] = array
            #print 'array_divided',array
            newRasterfn = "OPIR_test/WAfiltered_OPIR_" + "%03d" % j + ".jpeg"
            main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)

            newRasterfn = "OPIR_test/Avgfiltered_OPIR_" + "%03d" % j + ".jpeg"
            main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, avg_val)

            newRasterfn = "OPIR_test/Difffiltered_OPIR_" + "%03d" % j + ".jpeg"
            main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, diff_array)

            newRasterfn = "OPIR_test/Avg10filtered_OPIR_" + "%03d" % j + ".jpeg"
            main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, avg_val10)

    plt.hist(value_array, bins=200)
    plt.savefig("OPIR_test/value_array.png")
