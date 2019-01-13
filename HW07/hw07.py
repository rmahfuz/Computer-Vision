import numpy as np
import cv2, os, time
from scipy import spatial
path = '../Users/rmahfuz/Desktop/661/HW07/'
#========================================================================================================
def create_hist(img):
    '''returns the histogram as a list'''
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = [0]*10
    P = 8
    R = 1
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1): #iterating through each pixel
            intensities = []
            for p in range(P):
                del_u = R*np.cos(2*np.pi*p/P)
                del_v = R*np.sin(2*np.pi*p/P)

                intensities.append((1-del_u)*(1-del_v)*img[i-1,j+1] + (1-del_u)*del_v*img[i+1,j+1] \
                                   + del_u*(1-del_v)*img[i-1,j-1] + del_u*del_v*img[i+1,j-1])
            intensities = np.array(intensities)
            intensities = (intensities >= img[i][j]).astype(int)
            #finding minimum value pattern
            min_val = np.packbits(intensities)
            for k in range(len(intensities)):
                pattern = list(map(lambda x: (x+k)%8, [0,1,2,3,4,5,6,7]))
                cur_val = np.packbits(intensities[pattern])
                if cur_val < min_val:
                    min_val = cur_val
            code = np.binary_repr(min_val.item(), width = 8)
            #analyzing runs
            if code[0] == '1': #if all are 1
                hist[8] += 1
            else:
                k = 7
                while(code[k] == code[k-1] and k > 0):
                    k-=1
                if k == 0: #if all are 0
                    hist[0] += 1
                else:
                    break1 = k
                    k-=1
                    while(code[k] == code[k-1] and k > 0):
                        k-=1
                    if k == 0: #if exactly 2 runs
                        hist[break1] += 1
                    else: #if more than 2 runs
                        hist[9] += 1
    assert sum(hist) == (img.shape[0]-2) * (img.shape[1]-2)
    return hist
#========================================================================================================
def train(className):
    '''appends to hists.txt the histograms for a particular class' training images'''
    hists = []
    for file in os.listdir(path + 'imagesDatabaseHW7/training/{}/'.format(className)):
        img = cv2.imread(path + 'imagesDatabaseHW7/training/{}/'.format(className) + file)
        start = time.time()
        hists.append(create_hist(img))
        print('img {} of {} took {} seconds'.format(file, className, time.time() - start))
    with open(path + 'hists.txt', 'a') as fh:
        for hist in hists:
            to_write = ''
            for i in range(len(hist) - 1):
                to_write += str(hist[i]) + ','
            to_write += str(hist[-1])
            fh.write(to_write + '\n')
#========================================================================================================
def classify(img):
    hist = create_hist(img)
    with open(path + 'hists.txt', 'r') as fh:
        lines = fh.readlines()
        dists = []
        i = 0
        for line in lines:
            cur_hist = list(map(lambda x: int(x), line.split(',')))
            dists.append((spatial.distance.euclidean(cur_hist, hist), int(i/20)))
            i+=1
        dists.sort()
        count = [0]*5
        for item in dists[:5]:
            count[item[1]] += 1
        prediction = count.index(max(count))
        return prediction     
#========================================================================================================
def train_all():
    train('beach')
    train('building')
    train('car')
    train('mountain')
    train('tree')
#========================================================================================================
def classify_all():
    classNames = ['beach', 'building', 'car', 'mountain', 'tree']
    confusion = np.zeros((5,5))
    i = 0
    for file in os.listdir(path + 'imagesDatabaseHW7/testing/'):
        img = cv2.imread(path + 'imagesDatabaseHW7/testing/{}'.format(file))
        ans = classify(img)
        if classNames[ans] == file[:-6]:
            print('{} is correctly classified as {}'.format(file, classNames[ans]))
        else:
            print('{} is incorrectly classified as {}'.format(file, classNames[ans]))
        confusion[int(i/5), ans] += 1
        i += 1
    print('Confusion matrix: ', confusion)
#========================================================================================================
def main():
    train_all()
    classify_all()
#========================================================================================================
if __name__ == '__main__':
    main()
