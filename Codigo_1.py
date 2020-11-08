import cv2
import numpy as np

dim = (300,300)

def verifique(imagem, ponto, conectividade):
    
    if conectividade == 4:
        print (ponto)
        if imagem[ponto[0]-1, ponto[1]] == 255:
            imagem[ponto[0]-1,ponto[1]] = 0
            print('0')
            return (ponto[0]-1, ponto[1])
        elif imagem[ponto[0], ponto[1]+1] == 255:
                imagem[ponto[0],ponto[1]+1] = 0
                print('1')
                return (ponto[0], ponto[1]+1)
        elif imagem[ponto[0]+1, ponto[1]] == 255:
                imagem[ponto[0]+1,ponto[1]] = 0
                print('2')
                return (ponto[0]+1, ponto[1])
        elif imagem[ponto[0], ponto[1]-1] == 255:
                imagem[ponto[0],ponto[1]-1] = 0
                print('3')
                return (ponto[0], ponto[1]-1)
        else:
            print('none')
    else:
        return ponto

def normalizaImagem(v):
    v = (v - v.min()) / (v.max() - v.min())
    resultado = (v * 255).astype(np.uint8)
    return resultado

imagem = cv2.imread('1_5.jpg')
imagemBin = 255 - imagem[:,:,0]

novaImagem = np.zeros(np.shape(imagemBin))
kernel = np.ones((3,3), np.uint8)
novaImagem = normalizaImagem((imagemBin>100)*1)

imCopy = np.copy(novaImagem)
imPlot = np.zeros(np.shape(imagem))
imPlot[:,:,0] = imPlot[:,:,1] = imPlot[:,:,2] = imCopy

novaImagem = cv2.dilate(novaImagem, kernel, iterations=1) - novaImagem
cv2.imshow('Imagem', imPlot)
cv2.waitKey(0)

max_xy = np.where(novaImagem == 255)
print (max_xy[0][0], max_xy[1][0])

novaImagemRGB = np.zeros(np.shape(imagem))
novaImagemRGB[:,:,0] = novaImagemRGB[:,:,1] = novaImagemRGB[:,:,2] = novaImagem

cv2.circle(novaImagemRGB, (max_xy[1][0], max_xy[0][0]), int(3), (0,0,255), 2)
iniciarPonto = (max_xy[0][0], max_xy[1][0])
ponto = verifique(novaImagem, iniciarPonto, 4)

while(ponto!=iniciarPonto):
    cv2.circle(novaImagemRGB, (ponto[1], ponto[0]), int(3), (0,0,255), 4)
    cv2.imshow('Imagem', imPlot)
    cv2.waitKey(1)

    cv2.circle(imPlot, (ponto[1], ponto[0]), int(3), (0,255,255), 6)
    ponto = verifique(novaImagem, ponto, 4)

cv2.imshow('Imagem', imPlot)
cv2.waitKey(0)
