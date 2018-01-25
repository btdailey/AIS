from keras.models import load_model

model = load_model('explosion_trained_2.h5py')

filenamer = raw_input("Path to image: ")
#filenamer = '/Users/briandailey/PycharmProjects/Explosions/landmine.jpg'
#filenamer = '/Users/briandailey/PycharmProjects/Explosions/nuclear.jpg'
#filenamer = '/Users/briandailey/PycharmProjects/Explosions/images_explosion/train/Explosions/0.jpg'
#filenamer ='/Users/briandailey/PycharmProjects/Explosions/images_explosion/test_set/Explosions/348.jpg'
#filenamer = '/Users/briandailey/PycharmProjects/Explosions/dog.jpg'
#filenamer = '/Users/briandailey/PycharmProjects/Explosions/park.jpg'
#filenamer = '/Users/briandailey/PycharmProjects/Explosions/fire.jpg'
#filenamer = '/Users/briandailey/PycharmProjects/Explosions/smoke.jpg'
#filenamer = '/Users/briandailey/PycharmProjects/Explosions/waterfall.jpg'
#print(filenamer)
# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(filenamer, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
#print result[0]
#training_set.class_indices
if result == 1:
    prediction = 'Non-Explosion'
else:
    prediction = 'Explosion'
print prediction