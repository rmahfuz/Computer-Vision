import numpy as np
import cv2, math, scipy.signal
import matplotlib.pyplot as plt

path = '../Users/rmahfuz/Desktop/661/HW04/'

def harris(img, sigma):  
    #finding smallest even integer greater than 4*sigma (size of haar filter)
    size = int(4*sigma) + 1
    if size % 2 != 0:
        size += 1
    assert size % 2 == 0
    haar_x = np.hstack([-1*np.ones((size, int(size/2))), np.ones((size, int(size/2)))])
    haar_y = np.vstack([np.ones((int(size/2), size)), -1*np.ones((int(size/2), size))])
    img_x = scipy.signal.convolve2d(img, haar_x, mode = 'same')
    img_y = scipy.signal.convolve2d(img, haar_y, mode = 'same')
    img_x = np.divide(np.subtract(img_x, np.min(img_x)), np.subtract(np.max(img_x), np.min(img_x)))
    img_y = np.divide(np.subtract(img_y, np.min(img_y)), np.subtract(np.max(img_y), np.min(img_y)))
    print('img_x.shape = ', img_x.shape)
    d_x2 = np.square(img_x)
    d_y2 = np.square(img_y)
    d_xy = np.multiply(img_x, img_y)
    #print(img_x.shape)
    win_size = int(5*sigma)
    if win_size % 2 == 0: #ensuring that window size is even
        win_size += 1
    corner_pts = []
    ratio_store = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            C = np.zeros((2,2))
            for k in range(i - int(win_size/2), i+int(win_size/2)+1):
                for l in range(j-int(win_size/2), j+int(win_size/2)+1):
                    if k < img.shape[0] and l < img.shape[1]:
                        C[0][0] += d_x2[k][l]
                        C[0][1] += d_xy[k][l]
                        C[1][0] += d_xy[k][l]
                        C[1][1] += d_y2[k][l]
            if np.linalg.matrix_rank(C) == 2:
                ratio_store[i][j] = np.linalg.det(C)/np.square(np.matrix.trace(C))
            else:
                ratio_store[i][j] = -1
    print('finished calculating Cs')
    fil_win = 29 #filter window size
    #thresh = 2*np.mean(ratio_store)
    thresh = 0.01
    #Filtering out non-local maxima
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x_min = max(0, i-int(fil_win/2))
            x_max = min(i+int(fil_win/2)+1, img.shape[0])
            y_min = max(0,j-int(fil_win/2))
            y_max = min(j+int(fil_win/2)+1, img.shape[1])
            max_ratio = np.max(ratio_store[x_min:x_max,y_min:y_max])
            # if local maximum and greater than threshold
            if ratio_store[i,j] == max_ratio and ratio_store[i,j] > thresh:
                pad_size = int(fil_win/2)
                if i >= pad_size and i < img.shape[0] - pad_size and j >= pad_size and j < img.shape[1] - pad_size:
                    corner_pts.append((i,j))
    print('finished calculating corner points')
    return corner_pts

def ssd(cor_pts1, cor_pts2, img1, img2): #thresh: try 5000
    '''returns a list of point correspondences, where every element is of the form [(pt_x1, pt_y1), (pt_x2, pt_y2)]'''
    fil_win = 31 #filter window length
    pad_len = int(fil_win/2)
    ssd_store = np.zeros((len(cor_pts1), len(cor_pts2))) #array to store all SSDs
    ssd_store = ssd_store - 1
    for i in range(len(cor_pts1)):
        for j in range(len(cor_pts2)):
            cor1 = cor_pts1[i]; cor2 = cor_pts2[j]
            
            x_lo1 = max(0, cor1[0]-pad_len)
            x_hi1 = min(cor1[0]+pad_len+1, img1.shape[0])
            y_lo1 = max(0,cor1[1]-pad_len)
            y_hi1 = min(cor1[1]+pad_len+1, img1.shape[1])

            x_lo2 = max(0, cor2[0]-pad_len)
            x_hi2 = min(cor2[0]+pad_len+1, img2.shape[0])
            y_lo2 = max(0,cor2[1]-pad_len)
            y_hi2 = min(cor2[1]+pad_len+1, img2.shape[1])

            if x_hi1 - x_lo1 == x_hi2 - x_lo2 and y_hi1 - y_lo1 == y_hi2 - y_lo2:
                ssd_store[i,j] = np.sum(np.square(np.subtract(img1[x_lo1:x_hi1,y_lo1:y_hi1], img2[x_lo2:x_hi2,y_lo2:y_hi2])))
            else:
                ssd_store[i,j] = -1
                

    to_ret = [];ssds = []; track = np.ones(len(cor_pts2))
    thresh = np.min(ssd_store[ssd_store > 0])
    for i in range(len(ssd_store)):
        cur = ssd_store[i]
        to_find = cur[cur>0]
        if len(to_find) > 0:
            j = np.argmin(to_find)
            #if min(to_find) < 2000000 and abs(cor_pts1[i][0] - cor_pts2[j][0]) < 20 and abs(cor_pts1[i][1] - cor_pts2[j][1]) < 35:
            if abs(cor_pts1[i][0] - cor_pts2[j][0]) < 35 and abs(cor_pts1[i][1] - cor_pts2[j][1]) < 80 and track[j] == 1:
                ssds.append(min(to_find))
                to_ret.append([cor_pts1[i], cor_pts2[j]])
                track[j] = 0
    #plt.plot(list(range(len(ssds))), ssds)
    #plt.show()
            
    return (to_ret, ssds)
#-------------------------------------------------------------------------------------------------------------------------------
def ncc(cor_pts1, cor_pts2, img1, img2): 
    #returns a list of point correspondences, where every element is of the form [(pt_x1, pt_y1), (pt_x2, pt_y2)]
    fil_win = 31 #filter window length
    pad_len = int(fil_win/2)
    ncc_store = np.zeros((len(cor_pts1), len(cor_pts2))) #array to store all NCCs
    ncc_store = ncc_store - 2
    for i in range(len(cor_pts1)):
        for j in range(len(cor_pts2)):
            cor1 = cor_pts1[i]; cor2 = cor_pts2[j]
            
            x_lo1 = max(0, cor1[0]-pad_len)
            x_hi1 = min(cor1[0]+pad_len+1, img1.shape[0])
            y_lo1 = max(0,cor1[1]-pad_len)
            y_hi1 = min(cor1[1]+pad_len+1, img1.shape[1])

            x_lo2 = max(0, cor2[0]-pad_len)
            x_hi2 = min(cor2[0]+pad_len+1, img2.shape[0])
            y_lo2 = max(0,cor2[1]-pad_len)
            y_hi2 = min(cor2[1]+pad_len+1, img2.shape[1])

            if x_hi1 - x_lo1 == x_hi2 - x_lo2 and y_hi1 - y_lo1 == y_hi2 - y_lo2:
                mean1 = np.mean(img1[x_lo1:x_hi1,y_lo1:y_hi1])
                mean2 = np.mean(img2[x_lo2:x_hi2,y_lo2:y_hi2])
                term1 = np.subtract(img1[x_lo1:x_hi1,y_lo1:y_hi1], mean1)
                term2 = np.subtract(img2[x_lo2:x_hi2,y_lo2:y_hi2], mean2)
                ncc_store[i,j] = np.divide(np.sum(np.multiply(term1, term2)),
                                           np.sqrt(np.multiply(np.sum(np.square(term1)), np.sum(np.square(term2)))))
                #np.sum(np.square(np.subtract(img1[x_lo1:x_hi1,y_lo1:y_hi1], img2[x_lo2:x_hi2,y_lo2:y_hi2])))                

    to_ret = [];nccs = []
    #thresh = np.min(ssd_store[ncc_store > 0])
    track = np.ones(len(cor_pts2))
    for i in range(len(ncc_store)):
        cur = ncc_store[i]
        to_find = cur[cur>=-1]
        if len(to_find) > 0:
            j = np.argmax(to_find)
            if abs(cor_pts1[i][0] - cor_pts2[j][0]) < 35 and abs(cor_pts1[i][1] - cor_pts2[j][1]) < 80 and track[j] == 1:# and max(to_find) > 0.45:
                nccs.append(max(to_find))
                to_ret.append([cor_pts1[i], cor_pts2[j]])
                track[j] = 0

    return (to_ret, nccs)
#-------------------------------------------------------------------------------------------------------------------------------
def make_gray(img):
    new_img = []
    for i in range(img.shape[0]): #Making the image grayscale
        new_img.append(list(map(lambda x: sum(x)/3, img[i])))
    img = np.array(new_img)
    return img
#-------------------------------------------------------------------------------------------------------------------------------
def use_harris(pair, sigma):
    color_img1 = cv2.imread(path + 'pictures/pair' + str(pair) + '/1.jpg')
    color_img2 = cv2.imread(path + 'pictures/pair' + str(pair) + '/2.jpg')

    img1 = make_gray(color_img1)
    img2 = make_gray(color_img2)

    '''img1_corners = harris(img1, sigma = sigma)
    img2_corners = harris(img2, sigma = sigma)
    
    # To write the corners points in files:
    
    print('number of corner points of image 1 are: ', len(img1_corners))
    with open(path + 'pictures/pair' + str(pair) + '/points1.txt', 'w') as fh:
        for i in range(len(img1_corners)):
            fh.write(str(img1_corners[i][0]) + ', ' + str(img1_corners[i][1]) + '\n' )
    
    #img2_corners = harris(img2, sigma = sigma)
    print('number of corner points of image 2 are: ', len(img2_corners))

    with open(path + 'pictures/pair' + str(pair) + '/points2.txt', 'w') as fh:
        for i in range(len(img2_corners)):
            fh.write(str(img2_corners[i][0]) + ', ' + str(img2_corners[i][1]) + '\n')'''

    # To read corner points from file
    img1_corners = []; img2_corners = []
    
    with open(path + 'pictures/pair' + str(pair) + '/points1.txt', 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            img1_corners.append(list(map(lambda x: int(x.strip()), line.split(','))))
    print('len(img1_corners) = ', len(img1_corners))

    with open(path + 'pictures/pair' + str(pair) + '/points2.txt', 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            img2_corners.append(list(map(lambda x: int(x.strip()), line.split(','))))
    print('len(img2_corners) = ', len(img2_corners))

    '''#Just indicating corners:
    color_img1 = cv2.imread(path + 'pictures/pair' + str(pair) + '/1.jpg')

    for corner in img1_corners:
        for i in range(corner[0] - 2, corner[0] + 3):
            for j in range(corner[1] - 2, corner[1] + 3):
                if i < img1.shape[0] and j < img1.shape[1]:
                    color_img1[i,j] = (0,0,255)
    cv2.imwrite(path + 'pictures/pair' + str(pair) + '/corners.jpg', color_img1)'''
            

    corresp, ssds = ncc(img1_corners, img2_corners, img1, img2) #correspondences. either ncc or ssd
    print('found correspondences of size ', len(corresp))

    
    stacked_img = np.hstack((color_img1, color_img2))
    i = 0
    for pt in corresp:
        cv2.line(stacked_img, (pt[0][1], pt[0][0]), (pt[1][1] + img1.shape[1], pt[1][0]), color = (255,0,0), thickness = 2)
        #cv2.line(stacked_img, (pt[0][1], pt[0][0]), (pt[1][1] + img1.shape[1], pt[1][0]), color = (0,0,ssds[i]*max(ssds)/255), thickness = 1)
        #cv2.putText(img = stacked_img, text = str(i), org = (pt[0][1], pt[0][0]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,ssds[i]*max(ssds)/255), thickness = 1)
        #cv2.putText(img = stacked_img, text = str(i), org = (pt[1][1]+img1.shape[1], pt[1][0]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,ssds[i]*max(ssds)/255), thickness = 1)
        i+=1
    cv2.imwrite(path + 'pictures/pair' + str(pair) + '/nccsigma'+str(sigma)+'.jpg', stacked_img)


    '''for corner in img1_corners:
        for i in range(corner[0] - 2, corner[0] + 3):
            for j in range(corner[1] - 2, corner[1] + 3):
                if i < img1.shape[0] and j < img1.shape[1]:
                    color_img1[i,j] = (0,0,255)
    cv2.imwrite(path + 'pictures/pair1/corners.jpg', color_img1)'''
    #plotting image1
    '''plt.imshow(img1)
    for i in range(len(corresp)):
        x = corresp[i][0][0]; y = corresp[i][0][1]
        plt.scatter(x, y, s = 11, c = 'r', marker = 'x')
        plt.text(x+0.3, y+0.3, str(i), fontsize=15)
    plt.savefig(path + 'pictures/pair' + str(pair) + '/corner1.png')
    
    plt.clf()

    #plotting image2
    plt.imshow(img2)
    for i in range(len(corresp)):
        x = corresp[i][1][0]; y = corresp[i][1][1]
        plt.scatter(x, y, s = 11, c = 'r', marker = 'x')
        plt.text(x+0.3, y+0.3, str(i), fontsize=15)
    plt.savefig(path + 'pictures/pair' + str(pair) + '/corner2.png')'''
    
if __name__ == '__main__':
    use_harris(pair = 4, sigma = 1.2)
    
        

    
