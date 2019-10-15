import os
import random
import shutil
import cv2

def gray_to_rgb(input,output):
    c = cv2.imread(input,0)
    rgb = cv2.merge((c,c,c))
    cv2.imwrite(output,rgb)

def valid_pair(image, mask):
    image = cv2.imread(image,1)
    mask = cv2.imread(mask,0)
    img_h, img_w, img_c = image.shape
    mask_h, mask_w = mask.shape
    if img_h == mask_h and img_w==mask_w:
        return True
    else:
        return False


if __name__ == "__main__":
    p = '/hdd/building footprints/AIRS/training/trainval'
    paths = [os.path.join(p,'train'),os.path.join(p,'val')]
    data = {}
    for path in paths:
        img_path = os.path.join(path,'image')
        for f in os.listdir(img_path):
            if '.tif' in f:
                filename, file_ext = os.path.splitext(f)
                data[filename] = {'img':os.path.join(img_path,f),'filename':filename}
        l_path = os.path.join(path,'label')
        for f in os.listdir(l_path):
            if '.tif' in f:
                filename, file_ext = os.path.splitext(f)
                if '_vis' in filename:
                    filename_1 = filename.split('_vis')[0]
                    data[filename_1]['anno'] = os.path.join(l_path,f)
    values = list(data.values())
    random.shuffle(values)
    test_size  = 20
    val_size = 80
    train_size = len(values)-test_size-val_size
    os.mkdir('/hdd/building footprints/airs_data')
    image_dir = '/hdd/building footprints/airs_data/val'
    label_dir = '/hdd/building footprints/airs_data/val_labels'
    os.mkdir(image_dir)
    os.mkdir(label_dir)
    for i in range(val_size):
        d = values[i]
        if valid_pair(d['img'],d['anno']):
            shutil.copy(d['img'],os.path.join(image_dir,d['filename']+'.tif'))
            gray_to_rgb(d['anno'],os.path.join(label_dir,d['filename']+'_L.tif'))
    image_dir = '/hdd/building footprints/airs_data/test'
    label_dir = '/hdd/building footprints/airs_data/test_labels'
    os.mkdir(image_dir)
    os.mkdir(label_dir)
    for i in range(test_size):
        d = values[val_size+i]
        if valid_pair(d['img'],d['anno']):
            shutil.copy(d['img'], os.path.join(image_dir, d['filename'] + '.tif'))
            gray_to_rgb(d['anno'], os.path.join(label_dir, d['filename'] + '_L.tif'))
    image_dir = '/hdd/building footprints/airs_data/train'
    label_dir = '/hdd/building footprints/airs_data/train_labels'
    os.mkdir(image_dir)
    os.mkdir(label_dir)
    for i in range(train_size):
        d = values[test_size+val_size+i]
        if valid_pair(d['img'],d['anno']):
            shutil.copy(d['img'], os.path.join(image_dir, d['filename'] + '.tif'))
            gray_to_rgb(d['anno'], os.path.join(label_dir, d['filename'] + '_L.tif'))