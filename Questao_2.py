from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import time, threading

def HU_FE(image):
        moments = cv2.moments(image.astype(np.float64))
        return np.asarray( cv2.HuMoments(moments).flatten())

def LBP_FE(image):
        lbp_image = local_binary_pattern(image, 256, 1, "uniform")
        hist, ret = np.histogram(lbp_image.ravel(), bins=256)
        return hist

def GLCM_FE(image):
        glcm = greycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
        xs = []
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        xs.append(greycoprops(glcm, 'correlation')[0, 0])
        xs.append(greycoprops(glcm, 'homogeneity')[0, 0])
        xs.append(greycoprops(glcm, 'ASM')[0, 0])
        xs.append(greycoprops(glcm, 'energy')[0, 0])
        xs.append(greycoprops(glcm, 'correlation')[0, 0])
        return np.asarray(xs);

def normalizaImagem(v):
    v = (v - v.min()) / (v.max() - v.min())
    resultado = (v * 255).astype(np.uint8)
    return resultado


dim = (300,300)
path = "./number1"
mom_Hu = []
GLCM = []
hist_LBP = []
hist_GLCM = []
hist_HU = []
LBP = []
eps = 1e-7

for r, d, f in os.walk(path):
    
    for filename in f:
        imagem = cv2.imread(os.path.join(path, filename), 0)
        imagem = cv2.resize(imagem, dim, interpolation=cv2.INTER_AREA)

        hist_HU = HU_FE(imagem)
        hist = hist_HU.astype('float')
        imagem_HU = [item for item in list(hist)]
        mom_Hu.append(imagem_HU)

        hist_GLCM = GLCM_FE(imagem)
        GLCM.append(hist_GLCM)

        hist_LBP =LBP_FE(imagem)
        hist2 = hist_LBP.astype('float')
        hist2 /= (hist2.sum() + eps)

        imagem_lbp = [item for item in list(hist2)]

        LBP.append(imagem_lbp)

    print('Salvando arquivo LBP')
    with open ('LBP' + '.csv', 'w') as outfileLBP:
        writer = csv.writer(outfileLBP)
        writer.writerows(GLCM)

    print('Salvando arquivo GLCM')
    with open('GLCM' + '.csv', 'w') as outfileGLCM:
        writer = csv.writer(outfileGLCM)
        writer.writerows(GLCM)

    print('Salvando arquivo HU')
    with open('HU' + '.csv', 'w') as outfileHU:
        writer = csv.writer(outfileHU)
        writer.writerows(mom_Hu)