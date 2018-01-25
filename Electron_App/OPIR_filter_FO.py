import gdal, osr
import numpy as np
import random
import math
import scipy as sp
from scipy import signal
from numpy import genfromtxt

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

    decay = .2
    n_samples=3 #5
    number_snapshots = 900
    x_dim = 400
    y_dim = 400
    array_past=np.zeros((y_dim,x_dim))
    array_past2 = np.zeros((n_samples,y_dim,x_dim))
    rasterOrigin = (-123.25745, 45.43013)
    pixelWidth = 100
    pixelHeight = 100
    for j in range(0,number_snapshots):
        readRaster = "OPIR_test/Data_sets/Set7/datacube_slice_" + "%03d" % j + ".csv"
        array = genfromtxt(readRaster, delimiter=',')

        #newRasterfn = "OPIR_test/tester1_OPIR_" + "%03d" % j + ".jpeg"
        #main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)
        if(j<1):
            array_past=array
            continue

        array = decay * array - (1 - decay) * array_past
        array_past = array

        newRasterfn = "OPIR_test/FOfiltered_OPIR_" + "%03d" % j + ".jpeg"
        main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)