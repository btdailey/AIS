import gdal, osr
import numpy as np
import random
import math
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#random.seed(10684)# SetSeed for reproducibility
#random.seed(10686)# SetSeed for reproducibility

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

def edge_detection(array,x,y):
    gx = -1*array[y-1][x-1]+ -2*array[y][x-1]+ -1*array[y+1][x-1] + array[y-1][x+1]+2*array[y][x+1]+array[y+1][x+1]
    gy = -1*array[y-1][x-1]+ -2*array[y-1][x]+ -1*array[y-1][x+1] + array[y+1][x-1]+2*array[y+1][x]+array[y+1][x+1]
    magnitude = math.sqrt(gx*gx + gy*gy)
    #print "x,y, magnitude",x,y,magnitude
    return magnitude

def FOIIR(signal_mag,background_mag,coefficient):
    magnitude = background_mag - coefficient*background_mag-signal_mag
    return magnitude




if __name__ == "__main__":
    random_movement_flag=0
    x_dim = 200
    y_dim = 200
    n_samples=3 #5

    height = 200 #Km
    pixelSize= 1 #m
    decay = .65
    decay2=.8
    foiir_coefficient=0.020 ##From Jon
    pixelWidth = 100
    pixelHeight = 100
    pixelMaxSize = 1000
    pixelmaxVoltage = 30
    voltage2pixel = 255 / 30
    number_snapshots = 300

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
    filter=0
    if noise ==1:
        coarse = int(input("Coarse Noise= 1, Fine = 2: "))
        filter = int(input("No Filter=0, First Order=1, Edge Detection=2, MedFilter=3, Wiener=4, Average=5, Moving Average=6, FOIIR=7,"
                           "FirstOrder+Edge=8, FirstOrder+MovingAverage=9: "))

    weight=1
    if(filter==9):
        for i in range(1,n_samples):
            weight+=pow(.5,i)

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

        signal_variation[k]=1
        #stationary_flag[k]=1
        #print signal_time[k]
        print binnumx[k],binnumy[k],signal_size[k],signal_strength[k],signal_time[k]

    print "PUTTING IN SIZABLE SIGNAL FOR EMILY"

    binnumx.append(0)
    binnumy.append(0)
    signal_size.append(max_size)
    signal_time.append(0)
    signal_strength.append(100)
    signal_variation.append(0)

    binnumx.append(3*x_dim/4)
    binnumy.append(100)
    signal_size.append(max_size/4)
    signal_time.append(0)
    signal_strength.append(10)
    signal_variation.append(0)
    #Need to update xmin,xmax,ymin,ymax for multiple sources
    # need to update velocity for multiple sources
    velocityx=[]
    velocityy=[]


    for k in range(n):
        velocityx.append(0)
        velocityy.append(0)


    #newminmax(binnumx[k], binnumy[k],signal_size[k])
    print "Forcing change on signals!!! "
    for k in range(0,n, 1):
        print signal_strength[k],signal_variation[k]
        #signal_strength[k]=5
        #signal_time[k]=0
        if stationary_flag==0:
            velocityx[k] = 0
            velocityy[k] = 0
        else:
            velocityx[k] = random.randint(-1,1)
            velocityy[k] = random.randint(-1,1)

    velocityx.append(1)
    velocityy.append(1)

    velocityx.append(0)
    velocityy.append(0)

    n = n+2

    print("placement and strength is ", binnumx, binnumy, signal_size, brightness)
    ##How to do loop for multiple sources?
    ##Loop over 30 secs at 30Hz
    for j in range(0,number_snapshots,1):
        array = np.full((y_dim, x_dim),-1)

        for k in range(0, n, 1):
            if random_movement_flag == 1 and stationary_flag[k] == 0:
                velocityx[k] = random.randint(-1, 1)
                velocityy[k] = random.randint(-1, 1)
                # print "velocity for ",k," is ",velocityx[k],velocityy[k]

            if (j > signal_time[k]):
                binnumx[k] += velocityx[k]
                binnumy[k] += velocityy[k]

                if(k == n-1):
                    if(binnumx[k] > x_dim):
                        binnumx[k]=0
                    if(binnumy[k]> y_dim):
                        binnumy[k]=0

            if j > signal_time[k]:
                if signal_variation[k] != 0:
                    signal_strength[k] = signal_strength[k] * signal_variation[k]

                if signal_strength[k] > 1000:
                    signal_strength[k] = 1000
                #print signal_strength[k]


        for x in range(0,x_dim,1):

            for y in range(0,y_dim,1):
                #if(array[y][x]!=-1):
                #    continue
                voltage = thermal_noise(30)
                #voltage = thermal_noise(10) + 25
                #voltage = thermal_noise(20)
                #voltagearray.append(voltage)
                #voltagearray2.append(thermal_noise(15)+20)
                #voltagearray3.append(thermal_noise(25) + 10)
                #voltage = thermal_noise(5)+30

                #Testing new thermal. Change 'units' to radiance "W/steradian/m^2"?
                #voltagearray.append(thermal_noise(600))
                #voltagearray2.append(thermal_noise(100)+500)
                #voltagearray3.append(thermal_noise(200)+400)
                #voltagearray4.append(thermal_noise(500))


                if noise == 2:
                    voltage=0

                #Choose small radius? choose random number of pixels?
                if(noise==1 and coarse==1):
                    signal = random.randint(2,4) #choose number of pixels at same value
                else:
                    signal = 1


                #print "noise,coarse :",noise,coarse
                if noise ==1 and coarse==1:
                    #print "here"
                    newminmax(x, y, signal)
                    for x1 in range(xmin,xmax,1):
                        for y1 in range (ymin,ymax,1):
                            if y_dim > y1 > 0 and x_dim > x1 > 0:
                                rewrite = random.random()
                                if array[y1][x1]==-1 or rewrite>.95:

                                    #print x1,y1
                                    array[y1][x1]=voltage*voltage2pixel
                                    #print "filling with noise",array[y1][x1]
                                    #array[y1][x1] = 0
                                    #print "now",array[y1][x1]
                elif noise ==1:
                    array[y][x]=voltage*voltage2pixel
                #print array[y][x]
                #print "x,y is ",x,y
                for k in range(0,n,1):

                    if j > signal_time[k]:

                        #print "here j is ",j
                        distance = distance_between(binnumx[k],binnumy[k],x,y)
                        #print "distance, x,y ",distance,x,y

                        if distance < 1:
                            array[y][x]=array[y][x]+signal_strength[k]*voltage2pixel
                            #print "stong j doing signal ", j, x, y, binnumx[k], binnumy[k], array[y][x],signal_strength[k],distance

                        elif distance < signal_size[k]+1:
                            array[y][x]=array[y][x]+signal_strength[k]*voltage2pixel
                            #print "stong j doing signal ", j, x, y,binnumx[k],binnumy[k],array[y][x], signal_strength[k],distance
                            #array[y][x] = signal_strength[k] * voltage2pixel
                            #print signal_strength[k]


                        elif distance < signal_size[k]*2:
                            distance = distance_closest_pixel(x,y,binnumx[k],binnumy[k],signal_size[k])
                            distance *= pixelSize
                            #print "weak j doing signal ",j,x,y,binnumx[k],binnumy[k],array[y][x],signal_strength[k],distance
                            array[y][x] = array[y][x] + signal_strength[k]*voltage2pixel/(distance*distance)
                            #array[y][x] = signal_strength[k] * voltage2pixel / (distance * distance)
                if array[y][x]>255:
                    array[y][x]=255

                if(array[y][x] ==-1):
                    array[y][x]=0


                ##Moving mean filter????
                #array[y][x] = .9*array_past[y][x]+.1*array[y][x]
                if(filter ==1):
                    #print "x,y array,past",x,y,array[y][x],array_past[y][x]
                    #array[y][x] =array_past[y][x]+(1-decay)*(array_past[y][x]-array[y][x])
                    array[y][x]=decay*array[y][x]-(1-decay)*array_past[y][x]
                    #array[y][x]=(1-decay)*array_past[y][x]-decay*array[y][x]
                    #array[y][x] = float*(.5 * array_past[y][x] + .5 * array[y][x])
                    array_past[y][x]=array[y][x]


                if(filter==2 or filter==8):
                    array_past[y][x]=array[y][x]
                if(filter==5):
                    array_past[y][x] += array[y][x]
                    array[y][x] = array_past[y][x]/(j+1)
                if (filter == 6):
                    for k in range(n_samples-1,0,-1):
                        array_past2[k][y][x]=array_past2[k-1][y][x] # shift past n_samples
                    array_past2[0][y][x]=array[y][x] #set time 0

                    array[y][x]=0
                    for k in range(0,n_samples):
                        array[y][x]+=array_past2[k][y][x]

                    array[y][x]=array[y][x]/n_samples
                if(filter==7):

                    mag_temp = FOIIR(array[y][x],array_past[y][x],foiir_coefficient)
                    array[y][x] += mag_temp

                    print mag_temp
                    array_past[y][x] = array[y][x]
                if(filter==9):
                    #array[y][x] = decay2 * array[y][x] - (1 - decay2) * array_past2[0][y][x]
                    #array_past[y][x] = array[y][x]

                    for k in range(n_samples-1,0,-1):
                        array_past2[k][y][x]=array_past2[k-1][y][x] # shift past n_samples
                    array_past2[0][y][x]=array[y][x] #set time 0

                    array[y][x]=0
                    for k in range(0,n_samples):
                        array[y][x] += array_past2[k][y][x]
                        #array[y][x]+=np.power(decay2,k)*array_past2[k][y][x]

                    array[y][x]=array[y][x]/(weight*n_samples)












                data_cube[y][x][j]=array[y][x]
        #print "j is ",j,"Array is ",array
        #if(filter==1):
            #min_temp = array.min()
            #print min_temp
            #for x in range(0,x_dim):
                #for y in range(0,y_dim):
                    #array[y][x]=array[y][x]-min_temp


        if(filter==2):
            for x in range(1, x_dim -1, 1):

                for y in range(1, y_dim -1, 1):
                    array[y][x]= edge_detection(array_past,x,y)
        if(filter==8):
            for x in range(1, x_dim -1, 1):

                for y in range(1, y_dim -1, 1):

                    #array[y][x]
                    temp1  = decay2 * array[y][x] - (1 - decay2) * array_past[y][x]
                    #array_past[y][x] = array[y][x]
                    #array[y][x]\
                    temp2   = edge_detection(array_past,x,y)
                    array[y][x]=max(temp1,temp2)
                    array_past[y][x]=array[y][x]



                    #print "1",array[y][x]

                    #print "2",array[y][x]


        if (filter == 3):
            print "first: ", array[y][x]
            array_new = sp.signal.medfilt(array)
            print "now: ", array[y][x]
        if (filter == 4):
            array = sp.signal.wiener(array)

        namer = "OPIR_test/datacube_slice_%03d" % j + ".csv"
        np.savetxt(namer, array, delimiter=",", fmt="%03.02f")

        #namer = "%03j"%j
        #print namer

        #newRasterfn = "OPIR_test/tester_OPIR_"+str(j)+".jpeg"
        newRasterfn = "OPIR_test/tester_OPIR_" + "%03d"%j + ".jpeg"
        main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)


    #plt.hist(voltagearray,bins=200)
    #plt.hist(voltagearray4, bins=200)

    #plt.hist(voltagearray2, bins=200)
    #plt.hist(voltagearray3, bins=200)

    #plt.savefig("OPIR_test/voltage.png")

    #np.save('OPIR_test/data_cube.npy',data_cube)
    #