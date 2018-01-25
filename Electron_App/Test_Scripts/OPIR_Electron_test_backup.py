import gdal, osr, os, ogr
import numpy as np
import random
random.seed(10684)# SetSeed for reproducibility

global xmin
global xmax
global ymin
global ymax


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

def newminmax(binnumx,binnumy,signal_size):
    global xmin
    global xmax
    global ymin
    global ymax


    xmin = binnumx - signal_size
    xmax = binnumx + signal_size

    ymin = binnumy - signal_size
    ymax = binnumy + signal_size

if __name__ == "__main__":
    random_movement_flag=1
    n = random.randint(1, 5)  # number of objects to appear in map
    print "Number of signals is ",n
    #startionary_flag =0: object is stationary, 1: moving
    stationary_flag=[]
    for k in range(0, n, 1):
        if random.random() <.75:
            stationary_flag.append(0)
        else:
            stationary_flag.append(1)
    
    
    rasterOrigin = (-123.25745,45.43013)
    #rasterOrigin = (0, 0)
    pixelWidth = 10
    pixelHeight = 10
    newRasterfn = 'OPIR_test/tester_OPIR.tif'

    array = np.zeros((400, 400))

    print "starting program"

    binnumx=[]
    binnumy=[]
    signal_size=[]
    brightness=[]
    for k in range(0,n,1):
        binnumx.append(random.randint(0, 400))
        binnumy.append(random.randint(0, 400))

        signal_size.append(random.randint(1, 9))
        brightness.append(random.uniform(0.5, 1.0))
        print binnumx[k],binnumy[k],signal_size[k],brightness[k]

    #Need to update xmin,xmax,ymin,ymax for multiple sources
    # need to update velocity for multiple sources
    velocityx=[]
    velocityy=[]
    for k in range(n):
        velocityx.append(0)
        velocityy.append(0)

    #newminmax(binnumx[k], binnumy[k],signal_size[k])
    for k in range(0, n, 1):
        if stationary_flag==0:
            velocityx[k] = 0
            velocityy[k] = 0
        else:
            velocityx[k] = random.randint(-1,1)
            velocityy[k] = random.randint(-1,1)

    print("placement and strength is ", binnumx, binnumy, signal_size, brightness)
    ##How to do loop for multiple sources?
    ##Loop over 30 secs at 30Hz
    for j in range(0,900,1):
        array = np.zeros((400, 400))

        for k in range(0,n,1):
            if random_movement_flag==1 and stationary_flag[k]==0:
                velocityx[k] = random.randint(-1, 1)
                velocityy[k] = random.randint(-1, 1)
                #print "velocity for ",k," is ",velocityx[k],velocityy[k]

            if(j>0):
                binnumx[k]+=velocityx[k]
                binnumy[k]+=velocityy[k]
            #print "k, x,y are ",k,binnumx[k],binnumy[k]
            newminmax(binnumx[k],binnumy[k],signal_size[k])
            #print binnumx, binnumy
            for x in range(0,400,1):

                for y in range(0,400,1):
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        #print x,y
                        array[y][x]= array[y][x] + 256*brightness[k]

        newRasterfn = "OPIR_test/tester_OPIR_"+str(j)+".jpeg"
        main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)