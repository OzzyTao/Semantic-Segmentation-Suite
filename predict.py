import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from utils.mega_predict import break_images, assemble_images
from utils import utils, helpers
from builders import model_builder
from osgeo import gdal
from osgeo.gdalconst import *

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=True, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped image ')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped image')
parser.add_argument('--input_height',type=int,default=512, help='Height of input image to network')
parser.add_argument('--input_width',type=int,default=512,help='Width of input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Input Height -->",args.input_height)
print("Input Width -->", args.input_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.input_width,
                                        crop_height=args.input_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_path))


print("Testing image " + args.image)

loaded_image = utils.load_image(args.image)
images = break_images(loaded_image,args.crop_height,args.crop_width)
input_images = images/255.0
output_images = []

st = time.time()
for i in range(input_images.shape[0]):
    input_image = input_images[i]
    input_image = cv2.resize(input_image,(args.input_width, args.input_height))
    output_image = sess.run(network, feed_dict={net_input:np.expand_dims(input_image,axis=0)})
    output_image = cv2.resize(output_image[0],(args.crop_width,args.crop_height))
    output_images.append(output_image)
output_images = np.array(output_images)
run_time = time.time()-st

output_image = assemble_images(output_images,loaded_image.shape[0],loaded_image.shape[1])
#output_image = np.array(output_image[0,:,:,:])
output_image = helpers.reverse_one_hot(output_image)

rows = output_image.shape[0]
cols = output_image.shape[1]

raster = gdal.Open(args.image)
geoTrans = raster.GetGeoTransform()
projection = raster.GetProjectionRef()
driver = raster.GetDriver()

out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
file_name = utils.filepath_to_name(args.image)
dir_name = os.path.dirname(args.image)
outDs = driver.Create(os.path.join(dir_name,"%s_pred.png"%(file_name)),cols,rows,1,GDT_UInt16)
for i in [1,2,3]:
    band = outDs.GetRasterBand(i)
    band.WriteArray(out_vis_image[:,:,i],0,0)
    band.FlushCache()
# cv2.imwrite(os.path.join(dir_name,"%s_pred.png"%(file_name)),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

# out_vis_image = 0.5 * loaded_image + 0.5 * out_vis_image
out_vis_image = cv2.addWeighted(loaded_image,0.8,out_vis_image,0.2,0)
# cv2.imwrite(os.path.join(dir_name,"%s_pred_vis.png"%(file_name)),np.uint8(out_vis_image))
outDs = driver.Create(os.path.join(dir_name,"%s_pred_vis.png"%(file_name)),cols,rows,1,GDT_UInt16)
for i in [1,2,3]:
    band = outDs.GetRasterBand(i)
    band.WriteArray(out_vis_image[:,:,i],0,0)
    band.FlushCache()

print("")
print("Finished!")
print("Wrote image " + "%s_pred.png"%(file_name))
print("Wrote image " + "%s_pred_vis.png"%(file_name))
