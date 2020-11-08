import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from random import randint

dim = (300,300)
ChainCode = []
conj_ChainCode = []
TamanhoSinal = []
contar = 0
imagemConta = 1
path = "./number1"

def most_frequent(list):
    occurence_count = Counter(list)
    rand = 0
    if occurence_count.most_common(1)[0][1]<=2:
        rand = randint(0,1)

    return occurence_count.most_common(1)[0][rand]

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

def normalizaSinal(maior_sinal, menor_tam):
    maior_tam = len(maior_sinal)
    r = float(maior_tam/menor_tam)
    sinal_reduzido = []
    sinal_reduzido.append(maior_sinal[0])
    val = 0
    contar=1

    while (len(sinal_reduzido)<menor_tam):
        flag = int(r)
        val = val +int(r)
        resto = r - flag
        sinal_reduzido.append(maior_sinal[val])
    
    print(sinal_reduzido)
    return sinal_reduzido

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

        max_xy = np.where(novaImagem == 255)
        novaImagemRGB = np.zeros(np.shape(imagem))
        novaImagemRGB[:,:,0] = novaImagemRGB[:,:,1] = novaImagemRGB[:,:,2] = novaImagem

        cv2.circle(novaImagemRGB, (max_xy[1][0], max_xy[0][0]), int(3), (0,0,255), 2)
        iniciarPonto = (max_xy[0][0], max_xy[1][0])
        ponto = verifique(novaImagem, iniciarPonto, 4)

        while(ponto!=iniciarPonto):
            ponto = verifique(novaImagem, ponto, 4)

        conj_ChainCode.append(ChainCode)        
        print(ChainCode)
        plt.subplot(tamanhoImagem, 1, imagemConta)
        plt.plot(ChainCode)
        imagemConta += 1
        ChainCode = []

    menor = 99999
    for num in conj_ChainCode:
        tamNum = len(num)
        if tamNum < menor:
            menor = tamNum
    sinal_reduzido = []
    novo_sinal = []
    for num in conj_ChainCode:
        val = len(num) - menor
        if val != 0:
            novo_sinal = normalizaSinal(num, menor)
            sinal_reduzido.append(novo_sinal)

    imagem_recuperada = np.zeros((dim))

    pronto_in = (0,50)
    prox_ponto = pronto_in

    sinal_reduzido = np.array(sinal_reduzido)
    print('---------------- Final ---------------')
    print(np.shape(sinal_reduzido))
    b = 0
    sinal_final = []

    for ch in range(menor):
        listch = sinal_reduzido[:,ch]

        npArray = np.array([b for elemento in listch])
        val = most_frequent(sinal_reduzido[:,ch])

        prox_ponto = recuperar(imagem_recuperada, prox_ponto, val)
        cv2.imshow('Imagem Recuperada', imagem_recuperada)
        cv2.waitKey(1)

    cv2.waitKey(0)    