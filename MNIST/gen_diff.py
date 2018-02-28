'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils1 import *

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

# MINIST: python gen_diff.py blackout 1 0.1 10 20 50 0
#args.transformation = 'blackout'
#args.weight_diff = 1
#args.weight_nc = 0.1
#args.step = 10
#args.seeds = 20
#args.grad_iterations = 50
#args.threshold = 0

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
# #####
# load data from "MNIST database of handwritten digits", which is a:
# dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
# #####
(_, _), (x_test, _) = mnist.load_data()

# #####
# print out the x_test matrix array values, only [0], [1], [2] are available
# print (x_test.shape[0], x_test.shape[1], x_test.shape[2])
# 10000 28 28
# #####

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# #####
# print out the x_test matrix array values after the reshape, [0], [1], [2], [3] are available
# print (x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])
# 10000 28 28 1
# #####


# #####
# change values of the x_test matrix into "float32" type
# #####
x_test = x_test.astype('float32')

# #####
# normalize, devide all x_test matrix values by 255
# #####
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

# ==============================================================================================
# start gen inputs
print ('the overall seeds is: ' + str(args.seeds))
for _ in range(args.seeds):
    # choose a random image from 10000 images
    gen_img = np.expand_dims(random.choice(x_test), axis=0)
    print ("gen_img.shape: " + str(gen_img.shape))
    # initialize the origial image
    orig_img = gen_img.copy()
    # first check if input already induces differences
    # the return value of model1.predict(gen_img) is an array
    # np.argmax: return the "index" of the biggst value of axis labelled ax
    print("model1.predict(gen_img)[0]: " + str(model1.predict(gen_img)[0]))
    label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(
        model3.predict(gen_img)[0])

    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + '**************************input already causes different outputs: {}, {}, {}'.format(label1, label2,
                                                                                            label3) + bcolors.ENDC)
        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        update_coverage(gen_img, model3, model_layer_dict3, args.threshold)
        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                 neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                 neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
        averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                       neuron_covered(model_layer_dict3)[0]) / float(
            neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
            neuron_covered(model_layer_dict3)[1])
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        imsave('./generated_inputs/' + 'already_differ_' + str(label1) + '_' + str(
            label2) + '_' + str(label3) + '.png', gen_img_deprocessed)
        continue

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    print("################### layer_name1, index1: " + str(layer_name1) + ", " + str(index1))
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx model1.get_layer('before_softmax').output[..., orig_label]: " + str(K.mean(model1.get_layer('before_softmax').output[..., orig_label])))
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        print("loss1:" + str(loss1))
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        print("loss2:" + str(loss2))
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
        print("loss3:" + str(loss3))
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    print("loss1_neuron:" + str(loss1_neuron))
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    print("loss2_neuron:" + str(loss2_neuron))
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    print("loss2_neuron:" + str(loss2_neuron))
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)
    print("layer_output: " + str(layer_output))

    # for adversarial image generation
    final_loss = K.mean(layer_output)
    print("final_loss: " + str(final_loss))

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])
    print("grads: " + str(grads))

    # this function returns the loss and grads given the input picture
    # K.function will initiate a Keras function
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

    # we run gradient ascent for 20 steps
    print ('the overall grad iterations is: ' + str(args.grad_iterations))
    for iters in range(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate(
            [gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        predictions1 = np.argmax(model1.predict(gen_img)[0])
        predictions2 = np.argmax(model2.predict(gen_img)[0])
        predictions3 = np.argmax(model3.predict(gen_img)[0])

        if not predictions1 == predictions2 == predictions3:
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            # save the result to disk
            imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '.png',
                   gen_img_deprocessed)
            imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '_orig.png',
                   orig_img_deprocessed)
            break
