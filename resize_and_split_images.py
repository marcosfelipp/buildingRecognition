import os
from PIL import Image
import math
import string

# insert here the path of imagens that you want resize
PATH = './images_resized_4000/label/'

# Change this path to new path:
PATH_TO_SAVE = './images_austin/label/'

#Change this value for change resolution: eg.: basewidth = 2000 = 2000x2000px   
basewidth = 4000

def count_pixels():
	caminhos = [os.path.join(PATH, nome) for nome in os.listdir(PATH)]
	arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
	pngs = [arq for arq in arquivos if arq.lower().endswith(".png")]
	
	black = 0
	white = 0	
	
	for path_img in pngs:
		image = Image.open(path_img)
		for pixel in image.getdata():
			if pixel == 0:
				black+= 1
			else:
				white+= 1		
		
	print("% of black: " + str(black/(black + white)) + " % of white: " + str(white/(black + white)) )
		

def resize_images():
	caminhos = [os.path.join(PATH, nome) for nome in os.listdir(PATH)]
	arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
	tifs = [arq for arq in arquivos if arq.lower().endswith(".tif")]

	for path_img in tifs:  
		image = Image.open(path_img)
		path_img = path_img.split('/')
		wpercent = (basewidth / float(image.size[0]))
		hsize = int((float(image.size[1]) * float(wpercent)))
		new_image = image.resize((basewidth, hsize), Image.ANTIALIAS)
		path_img[-1] = path_img[-1].replace('.tif', '.png')
		print("Image " + path_img[-1] + " Resized")
		height_slice(new_image, path_img[-1], 625)
		print("Image " + path_img[-1] + " Sliced")


def height_slice(img, img_name, slice_size):
    width, height = img.size
    upper = 0
    left = 0
    slices = int(math.ceil(height/slice_size))
    count = 1

    for slice in range(slices):
        if count == slices:
            lower = height
        else:
            lower = int(count * slice_size)  

        bbox = (left, upper, width, lower)
        working_slice = img.crop(bbox)
        upper += slice_size
	
        width_slice(working_slice, img_name, slice_size, count)
        count +=1

def width_slice(img, img_name, slice_size, count_h):
    width, height = img.size
    upper = 0
    left = 0
    slices = int(math.ceil(width/slice_size))
    count = 1

    for slice in range(slices):
        if count == slices:
            lower = width
        else:
            lower = int(count * slice_size)  

        bbox = (left, upper, lower, height)
        working_slice = img.crop(bbox)
        left += slice_size
	
        working_slice.save(PATH_TO_SAVE + str(count_h) + str(count) + img_name)
        count +=1

def select_city():
	caminhos = [os.path.join(PATH, nome) for nome in os.listdir(PATH)]
	arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
	pngs = [arq for arq in arquivos if arq.lower().endswith(".png")]
	for path_img in pngs:
		if string.find(path_img, "austin") != -1:
			image = Image.open(path_img)
			path_save = path_img.split('/')
			image.save(PATH_TO_SAVE + path_save[-1])

if __name__ == '__main__':
	#resize_images()
	count_pixels()
	#select_city()
