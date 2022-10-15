import cv2
import numpy as np
from scipy import ndimage
from numpy.linalg.linalg import LinAlgError
import timeit
from statistics import variance as var


def evalAlignment(aligned1, im2):
  '''
  Computes the error of the aligned image (aligned1) and im2, as the
  average of the average minimum distance of a point in aligned1 to a point in im2
  and the average minimum distance of a point in im2 to aligned1.
  '''
  d2 = ndimage.distance_transform_edt(1-im2) #distance transform
  err1 = np.mean(np.mean(d2[aligned1 > 0]))
  d1 = ndimage.distance_transform_edt(1-aligned1)
  err2 = np.mean(np.mean(d2[im2 > 0]))
  err = (err1+err2)/2
  return err

def displayAlignment(im1, im2, aligned1, thick=False):
  im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
  '''
  Displays the alignment of im1 to im2
     im1: first input image to alignment algorithm (im1(y, x)=1 if (y, x) 
      is an original point in the first image)
     im2: second input image to alignment algorithm
     aligned1: new1(y, x) = 1 iff (y, x) is a rounded transformed point from the first time 
     thick: true if a line should be thickened for display
  ''' 
  if thick:
    # for thick lines (looks better for final display)
    dispim = np.concatenate((cv2.dilate(im1.astype('uint8'), np.ones((3,3), np.uint8), iterations=1), \
                             cv2.dilate(aligned1.astype('uint8'), np.ones((3,3), np.uint8), iterations=1), \
                             cv2.dilate(im2.astype('uint8'), np.ones((3,3), np.uint8), iterations=1)), axis=-1)
  else:
    # for thin lines (faster)
    dispim = np.concatenate((im1, aligned1, im2), axis = -1)
  return dispim

def createImagepairs(list_images, imgpath):
  images = []
  for i in range(len(list_images)):
    img1 = cv2.imread(f'{imgpath}{list_images[i]}_1.png')
    img2= cv2.imread(f'{imgpath}{list_images[i]}_2.png')
    temp = [img1, img2]
    images.append(temp)
  return images

def getAB(coor_image1, coor_image2):
    A = []
    B = []

    for i in range(len(coor_image1)):
        A.append(np.array([coor_image1[i, 0], coor_image1[i, 1], 0, 0, 1, 0]))
        A.append(np.array([0, 0, coor_image1[i, 0], coor_image1[i, 1], 0, 1]))
        B.append([coor_image2[i, 0]])
        B.append([coor_image2[i, 1]])

    A = np.asarray(A)
    B = np.asarray(B)

    return A, B


def initT(im1, im2):
    im1 = gray(im1)
    im2 = gray(im2)
    x1, y1 = np.nonzero(im1)[0], np.nonzero(im1)[1]
    x2, y2 = np.nonzero(im2)[0], np.nonzero(im2)[1]
    T_init = np.array([[np.sqrt(var(x2 - np.mean(x2)) / var(x1 - np.mean(x1))), 0, (np.mean(x2) - np.mean(x1))],
                       [0, np.sqrt(var(y2 - np.mean(y2)) / var(y1 - np.mean(y1))), (np.mean(y2) - np.mean(y1))],
                       [0, 0, 1]])
    return x1, y1, x2, y2, T_init


def alignimages(im1, im2):
    coor_image1 = []
    coor_image2 = []
    x1, y1, x2, y2, T_init = initT(im1, im2)
    X1 = x1.copy()
    Y1 = y1.copy()

    for i in range(len(x1)):
        A = np.array([[x1[i]], [y1[i]], [1]])
        init_trans = np.matmul(T_init, A)
        x1[i] = init_trans[0, 0]
        y1[i] = init_trans[1, 0]

    for i in range(len(x1)):
        keypoint_distances = []
        coor_image1.append([X1[i], Y1[i]])
        for j in range(len(x2)):
            a = np.array((x1[i], y1[i]))
            b = np.array((x2[j], y2[j]))
            dist = np.linalg.norm(a - b)
            keypoint_distances.append(dist)

        coor_image2.append([x2[np.argmin(keypoint_distances)], y2[np.argmin(keypoint_distances)]])
        keypoint_distances = []

    coor_image1 = np.array(coor_image1)
    coor_image2 = np.array(coor_image2)

    A, B = getAB(coor_image1, coor_image2)
    T = np.linalg.lstsq(A, B, rcond=None)[0]
    T_matrix = np.array([[T[0, 0], T[1, 0], T[4, 0]], [T[2, 0], T[3, 0], T[5, 0]], [0, 0, 1]])

    return T_matrix


def gray(image):
    gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gr

imgPath = '../input/'
save_path = '../output/'

objList = ['apple', 'bat', 'bell', 'bird', 'Bone', 'bottle', 'brick',
           'butterfly', 'camel', 'car', 'carriage', 'cattle', 'cellular_phone',
           'chicken', 'children', 'device7', 'dog', 'elephant', 'face', 'fork', 'hammer',
           'Heart', 'horse', 'jar', 'turtle']

numObj = len(objList)

image_pairs = createImagepairs(objList, imgPath)

for i in range(len(image_pairs)):
    start = timeit.default_timer()
    i1 = image_pairs[i][0]
    i2 = image_pairs[i][1]
    aligned_T = alignimages(i1, i2)
    aligned_img = np.zeros((i1.shape[0], i1.shape[1]))

    x1, y1 = np.nonzero(i1)[0], np.nonzero(i1)[1]
    for j in range(len(x1)):
        affine = aligned_T[0:2, 0:2]
        trans = aligned_T[0:2, 2:3]
        prev = np.array([[x1[j]], [y1[j]]])
        next = np.matmul(affine, prev) + trans
        if next[0, 0] <= aligned_img.shape[0] and next[1, 0] <= aligned_img.shape[1]:
            aligned_img[int(next[0, 0]), int(next[1, 0])] = 255
    stop = timeit.default_timer()
    error = evalAlignment(aligned_img, i2)
    disp = displayAlignment(gray(i1), gray(i2), aligned_img, thick=False)
    print(objList[i].capitalize() + ": ")
    cv2.imshow("Image Alignment", disp)
    cv2.imwrite(save_path + objList[i] + '_aligned.png', disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Error: {:0.2f}".format(error))
    print("Runtime: {:0.2f}s".format(stop - start))
    print('-----------------')

