import os
import shutil
import numpy as np
from numpy import linalg as LA
import seaborn as sns
from PIL import ImageFile, Image
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
import matplotlib.pyplot as plt
import matplotlib
import skimage.measure
import random
import cv2
matplotlib.use('Agg')

# from train_places import AverageMeter, accuracy

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# {'Alosa chrysochloris': 0, 'Carassius auratus': 1, 'Cyprinus carpio': 2, 'Esox americanus': 3, 'Gambusia affinis': 4, 'Lepisosteus osseus': 5, 
# 'Lepisosteus platostomus': 6, 'Lepomis auritus': 7, 'Lepomis cyanellus': 8, 'Lepomis gibbosus': 9, 'Lepomis gulosus': 10, 'Lepomis humilis': 11,
#  'Lepomis macrochirus': 12, 'Lepomis megalotis': 13, 'Lepomis microlophus': 14, 'Morone chrysops': 15, 'Morone mississippiensis': 16, 
#  'Notropis atherinoides': 17, 'Notropis blennius': 18, 'Notropis boops': 19, 'Notropis buccatus': 20, 'Notropis buchanani': 21, 'Notropis dorsalis': 22, 
#  'Notropis hudsonius': 23, 'Notropis leuciodus': 24, 'Notropis nubilus': 25, 'Notropis percobromus': 26, 'Notropis stramineus': 27, 'Notropis telescopus': 28,
#   'Notropis texanus': 29, 'Notropis volucellus': 30, 'Notropis wickliffi': 31, 'Noturus exilis': 32, 'Noturus flavus': 33, 'Noturus gyrinus': 34, 
#   'Noturus miurus': 35, 'Noturus nocturnus': 36, 'Phenacobius mirabilis': 37}

from tqdm import tqdm

'''
    This function finds the top 50 images that gets the greatest activations with respect to the concepts.
    Concept activation values are obtained based on iternorm_rotation module outputs.
    Since concept corresponds to channels in the output, we look for the top50 images whose kth channel activations
    are high.
'''
# def plot_concept_top50(args, val_loader, model, whitened_layers, print_other = False, activation_mode = 'pool_max'):
def plot_concept_top50(val_loader, model, whitened_layers, print_other = False, activation_mode = 'pool_max_s1'):
    # switch to evaluate mode
    model.eval()
    concepts = 'Alosa chrysochloris,Carassius auratus,Cyprinus carpio,Esox americanus,Gambusia affinis,Lepisosteus osseus'
    # concepts = 'Notropis percobromus,Notropis blennius,Morone mississippiensis,Notropis buchanani,Esox americanus,Cyprinus carpio'
    # concepts = 'test_label'
    from shutil import copyfile
    # dst = '/home/mridul/taming-transformers/plot/' + '_'.join(concepts.split(',')) + '/'
    dst = '/home/mridul/taming-transformers/plot/' + 'concepts_train_pool_max_s1' + '/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    layer_list = whitened_layers.split(',')
    folder = dst + '_'.join(layer_list) + '_rot_cw/'
    print(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    # model = model.model

    outputs= []
    def hook(module, input, output):
        from MODELS.iterative_normalization import iterative_normalization_py
        X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                 module.eps, module.momentum, module.training)
        size_X = X_hat.size()
        size_R = module.running_rot.size()
        X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

        X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
        X_hat = X_hat.view(*size_X)

        outputs.append(X_hat.cpu().numpy())
    
    model.encoder.norm_out.register_forward_hook(hook)
    # concepts = ['Notropis percodbromus', 'Notropis blennius', 'Morone mississippiensis', 'Notropis buchanani', 'Esox americanus', 'Cyprinus carpio']

    # modify from here
    begin = 0
    end = len(concepts.split(','))
    if print_other:
        # begin = len(args.concepts.split(','))
        # end = begin+30
        begin = print_other
        end = begin + 1
    concepts = concepts.split(',')

    
    with torch.no_grad():
        for k in range(begin, end):
            print(k)
            if k < len(concepts):
                output_path = os.path.join(folder, concepts[k])
            else:
                output_path = os.path.join(folder, 'other_dimension_'+str(k))
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            paths = []
            vals = None
            # for i, (input, _, path) in enumerate(val_loader):
            for i, batch in tqdm(enumerate(val_loader)):
                paths += list(batch['file_path_'])
                input_var = torch.autograd.Variable(batch['image']).cuda()
                # input_var = torch.autograd.Variable(batch['image'])
                X_var = input_var.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                X_var = X_var.float()
                outputs = []
                model.cuda()
                model(X_var)
                val = []
                for output in outputs:
                    if activation_mode == 'mean':
                        val = np.concatenate((val,output.mean((2,3))[:,k]))
                    elif activation_mode == 'max':
                        val = np.concatenate((val,output.max((2,3))[:,k]))
                    elif activation_mode == 'pos_mean':
                        pos_bool = (output > 0).astype('int32')
                        act = (output * pos_bool).sum((2,3))/(pos_bool.sum((2,3))+0.0001)
                        val = np.concatenate((val,act[:,k]))
                    elif activation_mode == 'pool_max':
                        kernel_size = 3
                        r = output.shape[3] % kernel_size
                        if r == 0:
                            val = np.concatenate((val,skimage.measure.block_reduce(output[:,:,:,:],(1,1,kernel_size,kernel_size),np.max).mean((2,3))[:,k]))
                        else:
                            val = np.concatenate((val,skimage.measure.block_reduce(output[:,:,:-r,:-r],(1,1,kernel_size,kernel_size),np.max).mean((2,3))[:,k]))
                    elif activation_mode == 'pool_max_s1':
                        X_test = torch.Tensor(output)
                        maxpool_value, maxpool_indices = nn.functional.max_pool2d(X_test, kernel_size=3, stride=1, return_indices=True)
                        X_test_unpool = nn.functional.max_unpool2d(maxpool_value,maxpool_indices, kernel_size=3, stride =1)
                        maxpool_bool = X_test == X_test_unpool
                        act = (X_test_unpool.sum((2,3)) / maxpool_bool.sum((2,3)).float()).numpy()
                        val = np.concatenate((val,act[:,k]))
                val = val.reshape((len(outputs),-1))
                if i == 0:
                    vals = val
                else:
                    vals = np.concatenate((vals,val),1)
                    # if val.shape[1]!= 5:
                    #     zeros = np.zeros((1,val.shape[0]))
                    #     val = np.concatenate((val,zeros.T), axis=1)

                    # vals = np.concatenate((vals,val),0)
            # print('here')
            # breakpoint()
            for i, layer in enumerate(layer_list):
                arr = list(zip(list(vals[i,:]),list(paths)))
                arr.sort(key = lambda t: t[0], reverse = True)
                # arr.sort(key = lambda t: t[0], reverse = False)
                # with open('76dim.txt', 'w') as f:
                #     for item in arr:
                #         f.write(item[1]+'\n')

                for j in range(10):
                    src = arr[j][1]
                    copyfile(src, output_path+'/'+'layer'+'_'+layer+'_'+str(j+1)+'.jpg')  
                    # copyfile(src, output_path+'/'+'layer'+layer+'_'+str(j+1)+'_reversed.jpg')  

    return 0
    