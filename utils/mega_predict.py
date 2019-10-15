import cv2
import numpy as np
def break_images(l_image, cell_height, cell_width):
    images = []
    height, width, channels = l_image.shape
    for x in range(0,height, cell_height):
        for y in range(0,width, cell_width):
            sub_img = l_image[x:x+cell_height,y:y+cell_width,:]
            s_h, s_w, t_ = sub_img.shape
            sub_img = np.pad(sub_img,((0,cell_height-s_h),(0,cell_width-s_w),(0,0)),mode='constant')
            images.append(sub_img)
    images = np.array(images)
    return images

def assemble_images(images, image_height,image_width):
    img = np.zeros([image_height,image_width,images.shape[-1]])
    tile_height = images.shape[1]
    tile_width = images.shape[2]
    i=0
    for y in range(0,image_height,tile_height):
        for x in range(0,image_width,tile_width):
            cp_h_size = image_height-y if y+tile_height > image_height else tile_height
            cp_w_size = image_width-x if x+tile_width > image_width else tile_width
            img[y:y+cp_h_size,x:x+cp_w_size,:] = images[i,:cp_h_size,:cp_w_size,:]
            i+=1
    return img[:image_height,:image_width,:]
