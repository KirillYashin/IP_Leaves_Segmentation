import cv2 as cv
import numpy as np


def remove_shadow(image_num: int, image):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()

    lower = np.array(de_shadow[str(image_num)][0])
    upper = np.array(de_shadow[str(image_num)][1])

    _mask = cv.inRange(hsv_image, lower, upper)

    cnts = cv.findContours(_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    for c in cnts:
        cv.drawContours(blank_mask, [c], -1, (255, 255, 255), -1)
        break
    image = cv.bitwise_and(original, blank_mask)

    return image


def damaged_leaves(image_num, mode: int):
    image = cv.imread(f'data/{str(image_num)}.jpg')

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    image = remove_shadow(image_num, image)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

    if mode == 0:
        image_denoised = cv.erode(image, kernel)
    elif mode == 1:
        image_denoised = cv.fastNlMeansDenoisingColored(image, None, h=3, templateWindowSize=3, searchWindowSize=11)
    elif mode == 2:
        image_denoised = cv.bilateralFilter(image, 5, 15, 15)
    elif mode == 3:
        image_denoised = cv.medianBlur(image, 7)
    elif mode == 4:
        image_denoised = cv.GaussianBlur(image, (1, 1), 0)

    markers = np.zeros((image.shape[0], image.shape[1]), dtype='int32')
    markers[90:140, 90:140] = 255
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[0:20, 236:255] = 1
    markers[236:255, 236:255] = 1

    leaves_area_BGR = cv.watershed(image_denoised, markers)
    healthy_part = cv.inRange(hsv_image, (36, 25, 25), (86, 255, 255))
    ill_part = leaves_area_BGR - healthy_part

    cv.imshow('Img', image)
    k = cv.waitKey(0)
    if k == ord('q'):
        cv.destroyAllWindows()

    mask = np.zeros_like(image, np.uint8)
    mask[leaves_area_BGR > 1] = (255, 0, 255)
    mask[ill_part > 1] = (0, 0, 255)

    return mask


de_shadow = {"1": [[0, 28, 36], [130, 255, 255]],
             "2": [[0, 28, 36], [130, 255, 255]],
             "3": [[0, 28, 36], [130, 255, 255]],
             "4": [[0, 0, 28], [112, 255, 255]],
             "5": [[0, 28, 36], [130, 255, 255]],
             "6": [[0, 13, 0], [103, 255, 255]],
             "7": [[0, 28, 36], [130, 255, 255]],
             "8": [[0, 28, 36], [130, 255, 255]],
             "9": [[0, 28, 36], [130, 255, 255]],
             "10": [[0, 28, 36], [130, 255, 255]],
             "11": [[0, 28, 36], [130, 255, 255]],
             "12": [[0, 28, 36], [130, 255, 255]]}


for i in range(1, 13):
    img = cv.imread(f'data/{str(i)}.jpg')
    cv.imwrite(f'{str(i)}_1.jpg', img)
    img = damaged_leaves(i, 2)
    cv.imwrite(f'{str(i)}.jpg', img)
