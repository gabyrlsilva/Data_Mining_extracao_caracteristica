import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np

def LBP_FE(image):
    lbp_image = local_binary_pattern(image, 256, 1, "uniform")
    hist, ret = np.histogram(lbp_image.ravel(), bins=256)
    return hist

eps = 1e-7
captura = cv2.VideoCapture('video.mp4')
conta = 0

while (1):
    ret, frame = captura.read()
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(conta%12==0):
        frame3 = frame2.astype('uint8')
        hist_LBP = LBP_FE(frame3)
        hist2 = hist_LBP.astype('float')
        hist2 /= (hist2.sum() + eps)

        image_lbp = [item for item in list(hist2)]
        plt.plot(image_lbp)
        plt.show()
        cv2.imshow("Video", frame2)
        k = cv2.waitKey(30) & 0xff
        conta += 1
        if k == 27:
            break
    else:
        cv2.imshow("Video", frame2)
        conta = conta + 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    plt.show(captura)
captura.release()
cv2.destroyAllWindows()