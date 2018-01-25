import cv2
import numpy as np
from matplotlib import pyplot as plt
from owslib.wms import WebMapService


wms = WebMapService('http://onearth.jpl.nasa.gov/wms.cgi?',version='1.1.1')

#'http://wms.jpl.nasa.gov/wms.cgi', version='1.1.1')
wms.identification.type
wms.getmap()

# img_wms=wms.getmap( layers=['global_mosaic'],
#                   styles=['visual_bright'],
#                   srs='EPSG:4326',
#                   bbox=(-112, 36, -106, 41),
#                   size=(300, 250),
#                   format='image/jpeg',
#                   transparent=True
#                   )
# out = open('jpl_mosaic_visb.jpg', 'wb')
# out.write(img_wms.read())
# out.close()

img = cv2.imread('test.jpeg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('test_2_distorted.png')
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


surf = cv2.xfeatures2d.SURF_create()
(kps, descs) = surf.detectAndCompute(gray, None)
(kps2, descs2) = surf.detectAndCompute(gray2, None)
#kp=sift.detect(gray,None)
    #.detect(gray,None)

cv2.drawKeypoints(gray,kps,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('./SURF_test/surf_keypoints.jpeg',img)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann=cv2.FlannBasedMatcher(index_params,search_params)

#matches = flann.knnMatch(des1,des2,k=2)
matches = flann.knnMatch(descs,descs2,k=2)
print len(matches)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]
good =[]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        good.append(m)


print len(good)

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img,kps,img2,kps2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()
