from PIL import Image
import os

dataPath = './training_hr_images/'
path = ['./train_H/', './train_L/', './valid_H/', './valid_L/']

for d in path:
	if not os.path.exists(d):
		os.makedirs(d)
	
img_list = os.listdir(dataPath)

for i in img_list:
	im = Image.open(dataPath + i)
	x, y = im.size
	newsize_L = (x//3, y//3)
	newsize_H = ((x//3)*3, (y//3)*3)
	im1 = im.resize(newsize_L)
	im3 = im.resize(newsize_H)

	if i[0] in '89':  # validation
		im3.save(path[2] + i)
		im1.save(path[3] + i)
	else:
		im3.save(path[0] + i)
		im1.save(path[1] + i)
	print(i)
