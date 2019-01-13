import numpy as np
import cv2

path = '../Users/rmahfuz/Desktop/661/HW04/'

def euclid(x,y):
    assert x.shape == y.shape
    return np.sqrt(np.sum(np.square(np.subtract(x,y))))

def sift(pair):

    color_img1 = cv2.imread(path + 'pictures/pair' + str(pair) + '/1.jpg')
    color_img2 = cv2.imread(path + 'pictures/pair' + str(pair) + '/2.jpg')
    print(color_img1.shape)
    
    with open(path + 'pictures/pair' + str(pair) + '//sift1.txt', 'r') as fh:
        txt = fh.read()
        old_pts1 = np.array(list(map(lambda x: float(x.strip()), txt.split(',')[:-1])))
        pts1 = old_pts1.reshape((int(len(old_pts1)/4), 4))

        with open(path + 'pictures/pair' + str(pair) + '//sift2.txt', 'r') as fh:
            txt = fh.read()
        old_pts2 = np.array(list(map(lambda x: float(x.strip()), txt.split(',')[:-1])))
        pts2 = old_pts2.reshape((int(len(old_pts2)/4), 4))

        #print(pts1.shape, pts2.shape)
        #assert pts1.shape == pts2.shape

        corresp = []
        for pt in pts1:
            min_dist = euclid(pt,pts2[0]); cor = pts2[0]
            for i in range(1,len(pts2)):
                dist = euclid(pt,pts2[i])
                if dist < min_dist:
                    min_dist = dist
                    cor = pts2[i]
            corresp.append([[int(pt[0]), int(pt[1])], [int(cor[0]), int(cor[1])]])


        stacked_img = np.hstack((color_img1, color_img2))
        i = 0
        for pt in corresp:
            cv2.line(stacked_img, (pt[0][1], pt[0][0]), (pt[1][1] + color_img1.shape[1], pt[1][0]), color = (0,0,255), thickness = 1)
            #cv2.line(stacked_img, (pt[0][1], pt[0][0]), (pt[1][1] + img1.shape[1], pt[1][0]), color = (0,0,ssds[i]*max(ssds)/255), thickness = 1)
            #cv2.putText(img = stacked_img, text = str(i), org = (pt[0][1], pt[0][0]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,255), thickness = 1)
            #cv2.putText(img = stacked_img, text = str(i), org = (pt[1][1]+color_img1.shape[1], pt[1][0]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,255), thickness = 1)
            i+=1 
        cv2.imwrite(path + 'pictures/pair' + str(pair) + '/sift.jpg', stacked_img)

if __name__ == '__main__':
    sift(pair = 4)
