"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

import cv2
import numpy as np
#from typing import Tuple


class Mouse_Click_Correspondence(object):

    def __init__(self,path1='',path2='',img1='',img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2


    def click_event(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1=self.sx1
            sy1=self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x, y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self,event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2= self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self,path1,path2):
        # reading the image
        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 2)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        np.save('p1.npy', points_1)
        np.save('p2.npy', points_2)



def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """
    x,y = p0
    x1,y1 = p1
    sx = x1-x
    sy = y1-y
    return np.sqrt(sx*sx+sy*sy)


def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    height, width = image.shape[:2]
    corners = [(0,0), (0,height-1), (width-1,0), (width-1,height-1)]
    return corners


""" def find_markers(image, template=None):
    Finds four corner markers.

    Use a combination of circle finding and/or corner finding and/or convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
   
    temp = template.copy()
    #temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp = temp.astype(np.float64)
    img = image.copy()
    img2 = img.copy()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_list = []

    canny = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(img, 150, 200)
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT,1,  20, param1=50,param2=25,minRadius=20,maxRadius=45)
    lencircles = 0
    if circles is not None:
        lencircles = len(circles)
    if lencircles < 4:
        print("nocircles")
        img = img.astype(np.float64)
        tempS0 = temp.shape[0]
        tempS1 = temp.shape[1]
        maxvalues = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        tempsum = np.sum(np.square(temp))
        for i in range(img.shape[0]-tempS0):
            for j in range(img.shape[1]-tempS1):
                splice = img[i:i+tempS0,j:j+tempS1]
                prod = np.multiply(splice,temp)
                norm = np.sqrt(np.multiply(np.sum(np.square(splice)), tempsum)).astype(np.float64)
                sumofProd = np.sum(prod)/norm
                #print(maxvalues[3][0])
                same = 0
                templist = []
                if sumofProd > maxvalues[3][0]:
                    templist.append([sumofProd, i+16, j+16])
                    for point in maxvalues:
                        #print(euclidean_distance((point[1],point[2]),(i+16, j+16)))
                        if euclidean_distance((point[1],point[2]),(i+16, j+16)) < 5:
                            same = 1
                            templist.append(point)
                            maxvalues.remove(point)
                    if not same:
                        maxvalues.append([sumofProd, i+16, j+16])
                        maxvalues.sort(key=lambda x: x[0], reverse = True)
                        maxvalues.pop()
                    else:
                        templist.sort(key=lambda x: x[0], reverse = True)
                        templist.pop()
                        maxvalues.extend(templist)
                        maxvalues.sort(key=lambda x: x[0], reverse = True)
    else:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(img2, (x, y), r, (0, 0, 255), 4)
        cv2.imwrite("img.png", img2)
        img = img.astype(np.float64)
        tempS0 = temp.shape[0]
        htempS0 = int(np.floor(tempS0/2))
        tempS1 = temp.shape[1]
        htempS1 = int(np.floor(tempS1/2))
        maxvalues = [[0],[0],[0],[0]]
        tempsum = np.sum(np.square(temp))
        for c in range(len(circles)):
            neighborhoodC = (circles[c][0], circles[c][1])
            currentMax = 0
            arr = [0,0,0]
            for i in range(circles[c][1]-htempS0-10,circles[c][1]-htempS0+20):
                for j in range(circles[c][0]-htempS1-10,circles[c][0]-htempS1+20):
                    #splice = img[i:i+tempS0,j:j+tempS1]
                    splice = img[i:i+tempS0,j:j+tempS1]
                    try:
                        prod = np.multiply(splice,temp)
                        norm = np.sqrt(np.multiply(np.sum(np.square(splice)), tempsum)).astype(np.float64)
                        sumofProd = np.sum(prod)/norm
                        #print(maxvalues[3][0])
                        if sumofProd > currentMax:
                            arr = [sumofProd, i+htempS0, j+htempS1]
                            currentMax = sumofProd
                    except:
                        pass
            if arr[0] < .85:
                temp = cv2.rotate(temp, cv2.cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite("temp.png", temp)
                for i in range(circles[c][1]-htempS0-10,circles[c][1]-htempS0+20):
                    for j in range(circles[c][0]-htempS1-10,circles[c][0]-htempS1+20):
                        #splice = img[i:i+tempS0,j:j+tempS1]
                        splice = img[j:j+tempS0,i:i+tempS1]
                        try:
                            prod = np.multiply(splice,temp)
                            norm = np.sqrt(np.multiply(np.sum(np.square(splice)), tempsum)).astype(np.float64)
                            sumofProd = np.sum(prod)/norm
                            #print(maxvalues[3][0])
                            if sumofProd > currentMax:
                                arr = [sumofProd, i+htempS0, j+htempS1]
                                currentMax = sumofProd
                        except:
                            pass
            if arr[0] > maxvalues[3][0]:
                maxvalues.append(arr)
                maxvalues.sort(key=lambda x: x[0], reverse = True)
                maxvalues.pop()
    maxvalues.sort(key=lambda x: x[2])
    if maxvalues[0][1] > maxvalues[1][1]:
        out_list.append((maxvalues[1][2], maxvalues[1][1]))
        out_list.append((maxvalues[0][2], maxvalues[0][1]))
    else:
        out_list.append((maxvalues[0][2], maxvalues[0][1]))
        out_list.append((maxvalues[1][2], maxvalues[1][1]))
    if maxvalues[2][1] > maxvalues[3][1]:
        out_list.append((maxvalues[3][2], maxvalues[3][1]))
        out_list.append((maxvalues[2][2], maxvalues[2][1]))
    else:
        out_list.append((maxvalues[2][2], maxvalues[2][1]))
        out_list.append((maxvalues[3][2], maxvalues[3][1]))
    print(out_list)
    return out_list """
def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding and/or corner finding and/or convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    temp = template.copy()
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp = temp.astype(np.float32)
    img = image.copy()
    img = cv2.medianBlur(img, 5)
    img2 = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_list = []
    maxvalues = []
    img = img.astype(np.float32)
    tempS0 = temp.shape[0]
    htempS0 = int(np.floor(tempS0/2))
    tempS1 = temp.shape[1]
    htempS1 = int(np.floor(tempS1/2))
    
    tempsum = np.sum(np.square(temp))
    """ matches = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    matchlocations = np.argwhere( matches >= .8)
    if len(matchlocations)== 4:
        for i in range(4):
            maxvalues.append([0,matchlocations[i][0]+htempS0,matchlocations[i][1]+htempS1]) """

    corners = cv2.cornerHarris(img, 2, 7, 0.07)
    indices = np.argwhere(corners>0.10*corners.max())
    Z = np.vstack(indices)
    Z = np.float32(Z)
    
    #cluster
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,4,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    maxvalues = []
    for c in range(len(Z)):
        img2[int(Z[c,0]), int(Z[c,1])] = [0,0,255]
    for c in range(len(center)):
        img2[int(center[c,0]), int(center[c,1])] = [0,255,0]
    cv2.imwrite("img.png", img2)
    searchrange = 10
    #check neighborhood
    for c in range(len(center)):
        cx,cy = (int(center[c,0]), int(center[c,1]))
        currentMax = 0
        arr = [0,0,0]
        #temprot = cv2.rotate(temp, cv2.cv2.ROTATE_90_CLOCKWISE)
        for i in range(cx-htempS0-searchrange,cx-htempS0+searchrange):
            for j in range(cy-htempS1-searchrange,cy-htempS1+searchrange):
                try:
                    #splice = img[i:i+tempS0,j:j+tempS1]
                    splice = img[i:i+tempS0,j:j+tempS1]
                    
                    prod1 = np.multiply(splice,temp)
                    #prod2 = np.multiply(splice,temprot)
                    norm = np.sqrt(np.multiply(np.sum(np.square(splice)), tempsum)).astype(np.float64)
                    sumofProd = np.sum(prod1)/norm
                    #sumofProd2 = np.sum(prod2)/norm
                    #print(maxvalues[3][0])
                    #maxProd = max(sumofProd,sumofProd2)
                    if sumofProd > currentMax:
                        currentMax = sumofProd
                        arr = [sumofProd, i+htempS0, j+htempS1]
                except:
                    pass
        maxvalues.append(arr)
    #print(maxvalues)
    #if it doesn't work, try rotating the template
    if maxvalues[0][0] < .83 and maxvalues[1][0] < .83 and maxvalues[2][0] < .83 and maxvalues[3][0] < .83:
        maxvalues = []
        temprot = cv2.rotate(temp, cv2.cv2.ROTATE_90_CLOCKWISE)
        for c in range(len(center)):
            cx,cy = (int(center[c,0]), int(center[c,1]))
            currentMax = 0
            arr = [0,0,0]
            img = img.astype(np.float64)
            tempS0 = temp.shape[0]
            htempS0 = int(np.floor(tempS0/2))
            tempS1 = temp.shape[1]
            htempS1 = int(np.floor(tempS1/2))
            
            tempsum = np.sum(np.square(temp))
            #temprot = cv2.rotate(temp, cv2.cv2.ROTATE_90_CLOCKWISE)
            for i in range(cx-htempS0-searchrange,cx-htempS0+searchrange):
                for j in range(cy-htempS1-searchrange,cy-htempS1+searchrange):
                    try:
                        #splice = img[i:i+tempS0,j:j+tempS1]
                        splice = img[i:i+tempS0,j:j+tempS1]
                        
                        prod1 = np.multiply(splice,temp)
                        prod2 = np.multiply(splice,temprot)
                        norm = np.sqrt(np.multiply(np.sum(np.square(splice)), tempsum)).astype(np.float64)
                        sumofProd = np.sum(prod1)/norm
                        sumofProd2 = np.sum(prod2)/norm
                        #print(maxvalues[3][0])
                        maxProd = max(sumofProd,sumofProd2)
                        if maxProd > currentMax:
                            currentMax = maxProd
                            arr = [sumofProd, i+htempS0, j+htempS1]
                    except:
                        pass
            maxvalues.append(arr)
    maxvalues.sort(key=lambda x: x[2])
    if maxvalues[0][1] > maxvalues[1][1]:
        out_list.append((maxvalues[1][2], maxvalues[1][1]))
        out_list.append((maxvalues[0][2], maxvalues[0][1]))
    else:
        out_list.append((maxvalues[0][2], maxvalues[0][1]))
        out_list.append((maxvalues[1][2], maxvalues[1][1]))
    if maxvalues[2][1] > maxvalues[3][1]:
        out_list.append((maxvalues[3][2], maxvalues[3][1]))
        out_list.append((maxvalues[2][2], maxvalues[2][1]))
    else:
        out_list.append((maxvalues[2][2], maxvalues[2][1]))
        out_list.append((maxvalues[3][2], maxvalues[3][1]))
    #print(out_list)
    return out_list

def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """
    img = image.copy()
    color = (0, 0, 255)
    out_image = cv2.line(img, markers[0], markers[1], color, thickness)
    out_image = cv2.line(out_image, markers[1], markers[3], color, thickness)
    out_image = cv2.line(out_image, markers[3], markers[2], color, thickness)
    out_image = cv2.line(out_image, markers[2], markers[0], color, thickness)
    return out_image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """
    out_image = imageB.copy()
    try:
        inv = np.linalg.inv(homography)
    except:
        inv = homography = np.identity(3)
    for i in range(imageB.shape[0]):
        for j in range(imageB.shape[1]):
            #print((i,j))
            avector = np.array([j,i,1])
            avector = np.dot(inv,avector)
            avector = avector/avector[2]
            aj = int(avector[0])
            ai = int(avector[1])
            try:
                if ai > 0 and aj > 0:
                    out_image[i,j,:] = imageA[ai,aj,:]
            except:
                pass
    return out_image


def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """

    b = np.zeros((9))
    b[8] = 1
    A = np.zeros((9,9))
    for i in range(4):
        x = srcPoints[i][0]
        y = srcPoints[i][1]
        x1 = dstPoints[i][0]
        y1 = dstPoints[i][1]
        A[2*i,:] = [x, y, 1, 0, 0, 0, -x1*x, -x1*y, -x1]
        A[2*i+1] = [0, 0, 0, x, y, 1, -y1*x, -y1*y, -y1]
    A[8,8] = 1
    try:
        ret = np.linalg.solve(A, b)
        #print(ret)
        homography = np.zeros((3,3))
        homography[0,:] = ret[0:3]
        homography[1,:] = ret[3:6]
        homography[2,:] = ret[6:9]
    except:
        homography = np.identity(3)
    return homography


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """

    # Open file with VideoCapture and set result to 'video'. (add 1 line)
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            yield frame
        else:
            break
    
    # Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None



class Automatic_Corner_Detection(object):

    def __init__(self):

        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)


    def filter(self, img, filter, padding=(0,0)):
        image = img.copy()
        if padding == (0,0):
            return cv2.filter2D(image, -1, filter)
        px = padding[0]
        py = padding[1]
        image = cv2.copyMakeBorder(image, px, px, py, py, cv2.BORDER_CONSTANT, None, value = 0)
        image = cv2.filter2D(image, -1, filter)
        #print(image[px:-px, py:-py])
        return image[px:-px, py:-py]
        

    def gradients(self, image_bw):

        Ix = self.filter(image_bw, self.SOBEL_X, padding = (1,1))
        Iy = self.filter(image_bw, self.SOBEL_Y, padding = (1,1))
        return Ix, Iy

    def get_gaussian(self, ksize, sigma):

        gauss = cv2.getGaussianKernel(ksize, sigma)
        return np.matmul(gauss,gauss.T)

    
    def second_moments(self, image_bw, ksize=7, sigma=10):

        sx2, sy2, sxsy = None, None, None
        Ix, Iy = self.gradients(image_bw)
        print(Ix)
        Ix2, Iy2, Ixy = Ix*Ix, Iy*Iy, Ix*Iy
        k = int(np.floor(ksize/2))
        k = (k,k)
        gkernel = self.get_gaussian(ksize, sigma)

        sx2 = self.filter(Ix2, gkernel, k)
        sy2 = self.filter(Iy2, gkernel, k)
        sxy = self.filter(Ixy, gkernel, k)
        return sx2, sy2, sxy

    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.04):
        '''
        Harris Corner Response Map
        Parameters:
            image_bw: np.array, Greyscale Image
            ksize: int, size of gaussian
            sigma: int, sigma for gaussian blurring
            alpha: float, trace constant
        Returns:
            R: np.array, normalized response map calculated
        '''    
        
        sx2, sy2, sxsy = self.second_moments(image_bw, ksize, sigma)
        det = sx2 * sy2 - sxsy** 2
        trace = sx2 + sy2
        R = det - alpha * trace * trace
        minR = np.min(R)
        norm = np.max(R) - minR
        if norm > 0:
            R = R - minR
            R = R/norm
        return R

    def pool2d(self, A, kernel_size, stride, padding, pool_mode='max'):
        '''
        2D Pooling
        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        '''
        # Padding
        A = np.pad(A, padding, mode='constant')

        # Window view of A
        output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                        (A.shape[1] - kernel_size)//stride + 1)
        kernel_size = (kernel_size, kernel_size)
        A_w = np.lib.stride_tricks.as_strided(A, shape = output_shape + kernel_size, 
                            strides = (stride*A.strides[0],
                                    stride*A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        if pool_mode == 'max':
            return A_w.max(axis=(1,2)).reshape(output_shape)
        elif pool_mode == 'avg':
            return A_w.mean(axis=(1,2)).reshape(output_shape)
            

    def nms_maxpool_numpy(self, R: np.ndarray, k, ksize):
        """Pooling function that takes in an array input
        Args:
            R (np.ndarray): Harris Response Map
            k (int): the number of corners that are to be detected with highest probability
            ksize (int): pooling size
        Return:
            x: x indices of the corners
            y: y indices of the corners
        """
    
        R[R<np.median(R)] = 0
        p = self.pool2d(R, ksize, 1, ksize//2)
        R[R!=p] = 0
        flat = R.reshape(-1)
        flat = np.sort(flat)[::-1]
        threshold_value = flat[k]
        x,y = np.where(R > threshold_value)
        #print((x,y))
        return y, x

    def harris_corner(self,image_bw, k=100):
        """Harris Corner Detection Function that takes in an image and detects the most likely k corner points.
        Args:
            image_bw (np.array): black and white image
            k (int): top k number of corners with highest probability to be detected by Harris Corner
        RReturn:
            x: x indices of the top k corners
            y: y indices of the top k corners
        """   
        R = self.harris_response_map(image_bw)
        return self.nms_maxpool_numpy(R, k, 7)




    def calculate_num_ransac_iterations(
            self,prob_success: float, sample_size: int, ind_prob_correct: float):

        num_samples = None

        p = prob_success
        s = sample_size
        e = 1 - ind_prob_correct

        num_samples = np.log(1 - p) / np.log(1 - (1 - e) ** s)
        print('Num of iterations', int(num_samples))

        return int(round(num_samples))




    def ransac_homography_matrix(self, matches_a: np.ndarray, matches_b: np.ndarray):

        p = 0.999
        s = 8
        sample_size_iter = 11
        e = 0.5
        threshold = 1
        numi = self.calculate_num_ransac_iterations(p, s, e)

        org_matches_a = matches_a
        org_matches_b = matches_b
        print('matches', org_matches_a.shape, org_matches_b.shape)
        matches_a = np.hstack([matches_a, np.ones([matches_a.shape[0], 1])])
        matches_b = np.hstack([matches_b, np.ones([matches_b.shape[0], 1])])
        in_list = []
        in_sum = 0
        best_in_sum = -99
        inliers = []
        final_inliers = []

        y = Image_Mosaic().get_homography_parameters(org_matches_b, org_matches_a)

        best_F = np.full_like(y, 1)
        choice = np.random.choice(org_matches_a.shape[0], sample_size_iter)
        print('s',org_matches_b[choice].shape,matches_b[choice].shape)
        best_inliers = np.dot(matches_a[choice], best_F) - matches_b[choice]
        print('inliers shape',best_inliers.shape,best_inliers)

        count = 0
        for i in range(min(numi, 20000)):
            
            choice = np.random.choice(org_matches_a.shape[0], sample_size_iter)
            match1, match2 = matches_a[choice], matches_b[choice]


            F = Image_Mosaic().get_homography_parameters(match2, match1)

            count += 1
            inliers = np.dot(matches_a[choice], F)- matches_b[choice]

            inliers = inliers[np.where(abs(inliers) <= threshold)]

            in_sum = abs(inliers.sum())
            best_in_sum = max(in_sum, best_in_sum)
            best_inliers = best_inliers if in_sum < best_in_sum else inliers

            if abs(in_sum) >= best_in_sum:
                # helper to debug
                # print('insum', in_sum)
                pass

            best_F = best_F if abs(in_sum) < abs(best_in_sum) else F


        for j in range(matches_a.shape[0]):
            final_liers = np.dot(matches_a[j], best_F) - matches_b[j]
            final_inliers.append(abs(final_liers) < threshold)

        final_inliers = np.stack(final_inliers)

        inliers_a = org_matches_a[np.where(final_inliers[:,0]==True) and np.where(final_inliers[:,1]==True)]
        inliers_b = org_matches_b[np.where(final_inliers[:,0]==True) and np.where(final_inliers[:,1]==True)]

        print('best F', best_F.shape, inliers_a.shape, inliers_b.shape, best_F, inliers_a, inliers_b)

        return best_F, inliers_a, inliers_b






class Image_Mosaic(object):

    def __int__(self):
        pass
    """ def getBoundingCorners(corners_1, corners_2, homography):
        x = []
        y = []
        corners_1_hom = corners_1
        #print(corners_2)
        for i in range(4):
            point = corners_1[i,0,:]
            point = np.append(point,1)
            point = np.matmul(homography,point)
            x.append(point[1]/point[2])
            y.append(point[0]/point[2])
            x.append(corners_2[i,0,1])
            y.append(corners_2[i,0,0])
        #print([np.min(x),np.min(y)],[np.max(x),np.max(y)])
        
        #min_xy = np.ndarray(np.min(x),np.min(y),dtype=np.float64)
        min_xy = np.zeros((2), dtype=np.float64)
        max_xy = np.zeros((2), dtype=np.float64)
        min_xy[1] = np.min(x)
        min_xy[0] = np.min(y)
        max_xy[1] = np.max(x)
        max_xy[0] = np.max(y)
        #print(min_xy)
        #print(max_xy)
        return min_xy , max_xy """
    def getBoundingCorners(self, srcimg, destimg, homography):
        srccorners = get_corners_list(srcimg)
        destcorners = get_corners_list(destimg)
        #[top-left, bottom-left, top-right, bottom-right]
        x = []
        y = []
        #print(corners)
        for i in range(len(destcorners)):
            point = np.array([destcorners[i][0], destcorners[i][1], 1])
            print("asdfasfd")
            print(point)
            point = np.matmul(homography,point)
            x.append(point[0]/point[2])
            x.append(srccorners[i][0])
            y.append(point[1]/point[2])
            y.append(srccorners[i][1])
        #minx miny maxx maxy
        return int(np.min(x)), int(np.min(y)), int(np.max(x)), int(np.max(y))
    """ def getImageCorners(image):
        corners = np.zeros((4, 1, 2), dtype=np.float32)
        #print("shape")
        #print(np.shape(image))
        corners[0,0,:] = [0,0]
        corners[1,0,:] = [0,np.shape(image)[0]]
        corners[2,0,:] = [np.shape(image)[1],0]
        corners[3,0,:] = [np.shape(image)[1],np.shape(image)[0]]
        #print(corners)
        return corners """

    def image_warp_inv(self, im_src, im_dst, homography):
        """ corners_1 = self.getImageCorners(im_dst)
        corners_2 = self.getImageCorners(im_src)
        print(corners_1) """
        minx , miny, maxx, maxy = self.getBoundingCorners(im_src, im_dst, homography)
        canvas_size = tuple((maxx-minx, maxy- miny))
        """ translationmatrix = [[1, 0, self.minx * -1],
                            [0, 1, self.miny * -1],
                            [0, 0, 1]]
        newhomography = np.matmul(translationmatrix,homography) """
        img_warped = np.zeros((canvas_size[1], canvas_size[0], im_src.shape[2]))
        img_warped = project_imageA_onto_imageB(im_dst, img_warped, homography)

        return img_warped

    def output_mosaic(self, img_src, img_warped):
        im_mos_out = img_warped.copy()
        for i in range(img_src.shape[0]):
            for j in range(img_src.shape[1]):
                if im_mos_out[i,j,:].all() == 0:
                    im_mos_out[i,j,:] = img_src[i,j,:]
        
        return im_mos_out

    def get_homography_parameters(self, points2, points1):
        """
        leverage your previous implementation of 
        find_four_point_transform() for this part.
        """
        return find_four_point_transform(points2, points1)





