import gdal, osr, os, ogr
import numpy as np
import random
import math
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


def distance_between(x1,y1,x2,y2):
    r = math.sqrt(pow(x2-x1,2) + pow(y2-y1,2))
    return r #returns same units as input

def distance_closest_pixel(x1,y1,x2,y2,signal):
    r=5000000

    for i in range(x2-signal, x2+signal, 1):
        for j in range(y2-signal,y2+signal,1):
            if distance_between(x2,y2,i,j) <signal:
                r_temp = distance_between(x1,y1,i,j)
                if r_temp < r:
                    r = r_temp

    return r


def Rayleigh_dis_invert(y,sigma):
    x = 1-y
    x = math.log(1-y) #nat log
    x = x*2*pow(sigma,2)
    x = math.sqrt(-x)
    return x

def thermal_noise(mean):
    sigma = mean*math.sqrt(2/math.pi) #Calculate sigma for dist
    rndm = random.random()
    voltage=Rayleigh_dis_invert(rndm,sigma)
    return voltage





if __name__ == "__main__":
    random_movement_flag=0

    height = 200 #Km
    pixelSize= 2 #m



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
    pixelMaxSize=1000
    pixelmaxVoltage=30
    voltage2pixel = 256/30
    newRasterfn = 'OPIR_test/tester_OPIR.tif'

    array = np.zeros((400, 400))

    print "starting program"

    binnumx=[]
    binnumy=[]
    signal_size=[]
    brightness=[]
    signal_strength=[]
    for k in range(0,n,1):
        binnumx.append(random.randint(0, 400))
        binnumy.append(random.randint(0, 400))

        signal_size.append(random.randint(1, 9)/pixelSize)
        brightness.append(random.uniform(0.5, 1.0))
        signal_strength.append(random.uniform(5.0,30))

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
        for k in range(0, n, 1):
            if random_movement_flag == 1 and stationary_flag[k] == 0:
                velocityx[k] = random.randint(-1, 1)
                velocityy[k] = random.randint(-1, 1)
                # print "velocity for ",k," is ",velocityx[k],velocityy[k]

            if (j > 0):
                binnumx[k] += velocityx[k]
                binnumy[k] += velocityy[k]OPIR_Electron_test.py

        for x in range(0,400,1):

            for y in range(0,400,1):
                voltage = thermal_noise(10)
                array[y][x]=voltage*voltage2pixel
                #if xmin <= x <= xmax and ymin <= y <= ymax:
                #print x,y
                # array[y][x]= array[y][x] + 256*brightness[k]
                for k in range(0,n,):
                    # print "k, x,y are ",k,binnumx[k],binnumy[k]
                    newminmax(binnumx[k], binnumy[k], signal_size[k])
                    # print binnumx, binnumy
                    distance = distance_between(binnumx[k],binnumy[k],x,y)

                    #print distance, signal_size[k]
                    if distance < 1:
                        array[y][x]=array[y][x]+signal_strength[k]*voltage2pixel
                    elif distance < signal_size[k]+1:
                        array[y][x]=array[y][x]+signal_strength[k]*voltage2pixel
                    elif distance < signal_size[k]*2:
                        distance = distance_closest_pixel(x,y,binnumx[k],binnumy[k],signal_size[k])
                        #print distance, brightness[k]
                        distance *= pixelSize
                        #print distance,x,y,binnumx[k],binnumy[k],signal_size[k]

                        array[y][x] = array[y][x] + pixelMaxSize * signal_strength[k]*voltage2pixel/(distance*distance)



        newRasterfn = "OPIR_test/tester_OPIR_"+str(j)+".jpeg"
        main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)