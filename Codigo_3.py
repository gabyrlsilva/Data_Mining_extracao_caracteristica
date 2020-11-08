import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

dim = (300,300)
ChainCode = []
TamanhoSinal = []
tamanhoImagem = 8
contar = 0
imagemConta = 1
path = "./number1"

def recuperar(imagem, prox_ponto, valor):
    if valor==0:
        imagem[prox_ponto[0], prox_ponto[1]] = 255
        return (prox_ponto[0]-1, prox_ponto[1])

    elif valor==1:
        imagem[prox_ponto[0], prox_ponto[1]] = 255
        return (prox_ponto[0], prox_ponto[1]+1)

    elif valor==2:
        imagem[prox_ponto[0], prox_ponto[1]] = 255
        return (prox_ponto[0]+1, prox_ponto[1])

    elif valor==3:
        imagem[prox_ponto[0], prox_ponto[1]] = 255
        return (prox_ponto[0], prox_ponto[1]-1)

    else:
        return prox_ponto

def verifique(imagem, ponto, conectividade):
    global contar
    if conectividade == 4:
        if imagem[ponto[0]-1, ponto[1]] == 255:
            imagem[ponto[0]-1,ponto[1]] = 0
            ChainCode.append(0)
            TamanhoSinal.append(contar)
            contar += 1
            return (ponto[0]-1, ponto[1])
        elif imagem[ponto[0], ponto[1]+1] == 255:
                imagem[ponto[0],ponto[1]+1] = 0
                ChainCode.append(1)
                TamanhoSinal.append(contar)
                contar += 1
                return (ponto[0], ponto[1]+1)
        elif imagem[ponto[0]+1, ponto[1]] == 255:
                imagem[ponto[0]+1,ponto[1]] = 0
                ChainCode.append(2)
                TamanhoSinal.append(contar)
                contar += 1
                return (ponto[0]+1, ponto[1])
        elif imagem[ponto[0], ponto[1]-1] == 255:
                imagem[ponto[0],ponto[1]-1] = 0
                ChainCode.append(3)
                TamanhoSinal.append(contar)
                contar += 1
                return (ponto[0], ponto[1]-1)
        else:
            print('none')
    else:
        return ponto

def normalizaImagem(v):
    v = (v - v.min()) / (v.max() - v.min())
    resultado = (v * 255).astype(np.uint8)
    return resultado

for r, d, f in os.walk(path):
    tamanhoImagem = len(f)
    for filename in f:
        imagem = cv2.imread(os.path.join(path, filename))
        imagem = cv2.resize(imagem, dim, interpolation=cv2.INTER_AREA)
        imagemBin = 255 - imagem[:,:,0]
        
        novaImagem = np.zeros(np.shape(imagemBin))
        kernel = np.ones((3,3), np.uint8)
        novaImagem = normalizaImagem((imagemBin>100)*1)

        imCopy = np.copy(novaImagem)
        imPlot = np.zeros(np.shape(imagem))
        imPlot[:,:,0] = imPlot[:,:,1] = imPlot[:,:,2] = imCopy

        novaImagem = cv2.dilate(novaImagem, kernel, iterations=1) - novaImagem
        novaImagem = cv2.resize(novaImagem, dim, interpolation=cv2.INTER_AREA)

        #cv2.imshow('Imagem', novaImagem)
        #cv2.waitKey(0)

        max_xy = np.where(novaImagem == 255)
        #print (np.shape(novaImagem))

        novaImagemRGB = np.zeros(np.shape(imagem))
        novaImagemRGB[:,:,0] = novaImagemRGB[:,:,1] = novaImagemRGB[:,:,2] = novaImagem

        cv2.circle(novaImagemRGB, (max_xy[1][0], max_xy[0][0]), int(3), (0,0,255), 2)
        iniciarPonto = (max_xy[0][0], max_xy[1][0])
        ponto = verifique(novaImagem, iniciarPonto, 4)

        while(ponto!=iniciarPonto):
            cv2.circle(imPlot, (ponto[1], ponto[0]), int(3), (0,0,255), 6)
            cv2.imshow('Imagem', imPlot)
            cv2.waitKey(1)

            cv2.circle(imPlot, (ponto[1], ponto[0]), int(3), (0,255,255), 4)
            ponto = verifique(novaImagem, ponto, 4)
        

        res = dict(Counter( ChainCode ))
        imagem_retornada = np.zeros(np.shape(imPlot))
        ponto_in = (1, res[1])
        prox_ponto = ponto_in
        
        #imagemRetornada = np.zeros((int(res[0]*1.2), int((res[1]+res[3])*1.2) ))

        for valor in ChainCode:
            prox_ponto = recuperar(imagem_retornada, prox_ponto, valor)
            cv2.imshow('Imagem Retornada', imagem_retornada)
            cv2.waitKey(1)

        print(ChainCode)
        plt.subplot(tamanhoImagem, 1, imagemConta)
        plt.plot(ChainCode)
        imagemConta += 1
        ChainCode=[]
    plt.show()