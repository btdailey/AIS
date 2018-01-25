import gdal, osr
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


def distance_between(x1,y1,x2,y2): #distance between two pixels
    r = math.sqrt(pow(x2-x1,2) + pow(y2-y1,2))
    return r #returns same units as input

def distance_closest_pixel(x1,y1,x2,y2,signal): #distance to closest pixel with signal
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
    random_movement_flag=0 #Signals move randomly?
    x_dim = 200 #temporary
    y_dim = 200 #temporary
    n_samples=3 #temp

    height = 200 #Km
    pixelSize= 1 #m

    pixelWidth = 100
    pixelHeight = 100
    pixelMaxSize = 1000
    pixelmaxVoltage = 30
    voltage2pixel = 255 / pixelmaxVoltage
    number_snapshots = 300 #temp

    x_dim = int(input("Size of X-dimension(pixels): "))
    y_dim = int(input("Size of Y-dimension(pixels): "))
    freq = int(input("Frequency of capture (Hz): "))
    time = int(input("Length of time record (seconds): "))
    seed = int(input("Set Seed: "))

    number_snapshots = freq*time
    data_cube = np.ndarray([y_dim, x_dim, number_snapshots])
    array_past2 = np.ndarray([n_samples, y_dim, x_dim])
    random.seed(seed)

    text_file = open("OPIR_test/Data_header.txt","w")
    text_file.write("%d \n" % x_dim)
    text_file.write("%d \n" % y_dim)
    text_file.write("%d \n" % freq)
    text_file.write("%d \n" %time)
    text_file.write("%d \n" %seed)

    noise = int(input("Noise = 1, No-Noise = 2: "))
    coarse=0

    if noise ==1:
        coarse = int(input("Coarse Noise (NOT IMPLEMENTED CORRECTLY!) = 1, Fine = 2: "))




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

    newRasterfn = 'OPIR_test/tester_OPIR.tif'

    array = np.zeros((y_dim, x_dim))
    array_past = np.zeros((y_dim,x_dim))
    array_past2 = np.zeros((n_samples,y_dim,x_dim))
    print "starting program"
    #initialize
    binnumx=[]
    binnumy=[]
    signal_size=[]
    brightness=[]
    signal_time=[]
    signal_strength=[]
    signal_variation=[]
    voltagearray=[]
    voltagearray2=[]
    voltagearray3=[]
    voltagearray4=[]
    max_size = min(x_dim, y_dim) / 25


    for k in range(0,n,1):
        binnumx.append(random.randint(0, x_dim))
        binnumy.append(random.randint(0, y_dim))

        signal_size.append(random.randint(1, max_size)/pixelSize)
        brightness.append(random.uniform(0.5, 1.0))
        signal_strength.append(random.uniform(5.0,30))
        signal_time.append(random.uniform(0,number_snapshots))
        if random.random()<.2:
            signal_variation.append(0)
        else:
            signal_variation.append(random.uniform(0,2))

        #signal_variation[k]=1 #Force Signal variation

        print 'Signal info:', binnumx[k],binnumy[k],signal_size[k],signal_strength[k],signal_time[k]


    #Need to update xmin,xmax,ymin,ymax for multiple sources
    # need to update velocity for multiple sources
    velocityx=[]
    velocityy=[]


    for k in range(n):
        velocityx.append(0)
        velocityy.append(0)

    for k in range(0,n, 1):
        print signal_strength[k],signal_variation[k]

        if stationary_flag==0:
            velocityx[k] = 0
            velocityy[k] = 0
        else:
            velocityx[k] = random.randint(-1,1)
            velocityy[k] = random.randint(-1,1)


    print("placement and strength is ", binnumx, binnumy, signal_size, brightness)
    ##How to do loop for multiple sources?
    ##Loop over snapshots at freq and time
    for j in range(0,number_snapshots,1):
        array = np.full((y_dim, x_dim),-1)

        for k in range(0, n, 1):
            if random_movement_flag == 1 and stationary_flag[k] == 0: #if random movement, set random number for velocity in x and y
                velocityx[k] = random.randint(-1, 1)
                velocityy[k] = random.randint(-1, 1)


            if (j > signal_time[k]): # is signal has appeared, start moving it
                binnumx[k] += velocityx[k]
                binnumy[k] += velocityy[k]

                if(k == n-1):
                    if(binnumx[k] > x_dim): #goes off screen
                        binnumx[k]=0
                    if(binnumy[k]> y_dim): #goes off screen
                        binnumy[k]=0

            if j > signal_time[k]: #signal appeared
                if signal_variation[k] != 0:
                    signal_strength[k] = signal_strength[k] * signal_variation[k] #vary strength

                if signal_strength[k] > 1000: #set max strength
                    signal_strength[k] = 1000



        for x in range(0,x_dim,1):

            for y in range(0,y_dim,1):

                voltage = thermal_noise(30) #Choose strength of noise

                if noise == 2: #No Noise
                    voltage=0

                #Choose small radius? choose random number of pixels?
                if(noise==1 and coarse==1):
                    signal = random.randint(2,4) #choose number of pixels at same value (COARSE NOISE NOT WORKING)
                else:
                    signal = 1



                if noise ==1 and coarse==1:
                    newminmax(x, y, signal)
                    for x1 in range(xmin,xmax,1):
                        for y1 in range (ymin,ymax,1):
                            if y_dim > y1 > 0 and x_dim > x1 > 0:
                                rewrite = random.random()
                                if array[y1][x1]==-1 or rewrite>.95:
                                    array[y1][x1]=voltage*voltage2pixel
                elif noise ==1:
                    array[y][x]=voltage*voltage2pixel

                for k in range(0,n,1):

                    if j > signal_time[k]:
                        distance = distance_between(binnumx[k],binnumy[k],x,y)

                        if distance < 1:
                            array[y][x]=array[y][x]+signal_strength[k]*voltage2pixel

                        elif distance < signal_size[k]+1:
                            array[y][x]=array[y][x]+signal_strength[k]*voltage2pixel

                        elif distance < signal_size[k]*2:
                            distance = distance_closest_pixel(x,y,binnumx[k],binnumy[k],signal_size[k])
                            distance *= pixelSize

                            array[y][x] = array[y][x] + signal_strength[k]*voltage2pixel/(distance*distance)

                if array[y][x]>255: #Pixel Overload? Set to max pixel size!
                    array[y][x]=255

                if(array[y][x] ==-1): #Didnt fill pixel? Set to zero!
                    array[y][x]=0







        namer = "OPIR_test/datacube_slice_%03d" % j + ".csv"
        np.savetxt(namer, array, delimiter=",", fmt="%03.02f")


        newRasterfn = "OPIR_test/tester_OPIR_" + "%03d"%j + ".jpeg"
        main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)
