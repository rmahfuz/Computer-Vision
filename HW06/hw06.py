import numpy as np
import cv2, sys
import matplotlib.pyplot as plt

path = '../Users/rmahfuz/Desktop/661/HW06/'
#==============================================================================================================              
def remove_sky_from_lighthouse(img):
    '''special function to remove sky from lighthouse after texture-based segmentation'''
    if img is None:
        img = cv2.imread(path + 'pictures/lighthouse/txt_segmented_color_iter2.jpg')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] > 100 and img[i][j][2] < 190:
                img[i][j] = [0,0,0]
    cv2.imwrite(path + 'pictures/lighthouse/removed_sky.jpg', img)
    return img
#==============================================================================================================              
def process_baby(img = None):
    '''special function to remove the white background after texture-based segmentation'''
    if img is None:
        img = cv2.imread(path + 'pictures/baby.jpg')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] > 210 and img[i][j][2] > 210 and img[i][j][2] > 210:
                img[i][j] = [0,0,0]
    cv2.imwrite(path + 'pictures/baby/refined.jpg', img)
    return img
#==============================================================================================================
def otsu(img):
    '''accepts a grayscale image. returns k'''
    assert len(img.shape) == 2
    L = np.max(img)
    N = img.shape[0]*img.shape[1]
    n = np.array(list(map(lambda i: len(img[img==i]), range(L))))
    p = n/N
    def omega(k):
        return np.sum(p[:int(k)])
    def mu(k):
        result = 0
        for i in range(int(k)):
            result += (i* p[i])
        return result
    mu_T = mu(L)
    def mu0(k):
        return mu(k)/omega(k)
    def mu1(k):
        return (mu_T - mu(k))/(1 - omega(k))
    def sigma_b_sq(k):
        return omega(k) * (1 - omega(k)) * np.square(mu1(k) - mu0(k))
    #Find all k's for which 0 < omega(k) < 1
    #k_range = list(range(L))[1:]
    k_range = np.array([])
    for k in range(L):
        if omega(k) > 0 and omega(k) < 1:
            k_range = np.append(k_range, k)
    #Within that range of k, find the k for which sigma_b_sq is maximum
    k_star = k_range[0]
    max_sigma = sigma_b_sq(k_star)
    for k in k_range:
        if sigma_b_sq(k) > max_sigma:
            max_sigma = sigma_b_sq(k)
            k_star = k
    #print('k_star = ', k_star)
    return k_star 
#==============================================================================================================
def segment(name, iteration = 0, img = None):
    '''segments by color. accepts name: a string. optionally accepts an image: img'''
    if img is None:
        img = cv2.imread(path + 'pictures/' + name +'.jpg')
    def segment_given_k(actual_image, k, order = 0):
        image = actual_image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if order == 0:
                    cmp = image[i,j] >= k
                else:
                    cmp = image[i,j] < k
                if cmp == True:
                    image[i,j] = 0
        return image
    k_blue = otsu(img[:,:,0]) #blue
    k_green = otsu(img[:,:,1]) #green
    k_red = otsu(img[:,:,2]) #red
    print('k_blue = {}, k_green = {}, k_red = {}'.format(k_blue, k_green, k_red))
    segmented_blue = segment_given_k(img[:,:,0].copy(), k_blue)
    segmented_green = segment_given_k(img[:,:,1].copy(), k_green)
    segmented_red = segment_given_k(img[:,:,2].copy(), k_red)
    new_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not(segmented_blue[i][j] != 0 and segmented_green[i][j] != 0 and segmented_red[i][j] != 0):
            #if segmented_blue[i][j] == 0 and segmented_green[i][j] == 0 and segmented_red[i][j] == 0:
                new_img[i][j] = 0

    cv2.imwrite(path + 'pictures/' + name + '/segmented_iter' + str(iteration) + '.jpg', new_img)
    #cv2.imwrite(path + 'pictures/' + name + '/exp2.jpg', new_img)

    return new_img
#==============================================================================================================
def segment_by_texture(name, iteration = 0, img = None):
    '''accepts name: a string. optionally accepts an image: img'''
    if img is None:
        img = cv2.imread(path + 'pictures/' + name +'.jpg')
    if len(img.shape) == 3: #convert to grayscale if img is bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def get_channel(N): #returns img. variance from NxN channel
        image = img.copy()
        win = int(N/2)
        for i in range(win, image.shape[0] - win):
            for j in range(win, image.shape[1] - win):
                patch = img[i-win:i+win+1, j-win:j+win+1]
                image[i,j] = np.var(patch)
        image = (image / np.max(image)) * 255
        return image.astype(int)
    def segment_given_k(actual_image, k, order = 0):
        image = np.zeros((actual_image.shape[0], actual_image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if order == 0:
                    cmp = actual_image[i,j] >= k
                else:
                    cmp = actual_image[i,j] < k
                if cmp == False:
                    image[i,j] = 255
        return image
    one = get_channel(3); two = get_channel(5); three = get_channel(7)
    k_one = otsu(one)
    k_two = otsu(two) 
    k_three = otsu(three) 
    print('k_one = {}, k_two = {}, k_three = {}'.format(k_one, k_two, k_three))
    segmented_one = segment_given_k(one, k_one)
    segmented_two = segment_given_k(two, k_two)
    segmented_three = segment_given_k(three, k_three)
    new_img = cv2.imread(path + 'pictures/' + name +'.jpg')
    #new_img = np.ones((img.shape[0], img.shape[1])); new_img = new_img * 255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not(segmented_one[i][j] != 0 and segmented_two[i][j] != 0 and segmented_three[i][j] != 0):
            #if segmented_blue[i][j] == 0 and segmented_green[i][j] == 0 and segmented_red[i][j] == 0:
                new_img[i][j] = [0,0,0]

    cv2.imwrite(path + 'pictures/' + name + '/txt_segmented_color_iter' + str(iteration) + '.jpg', new_img)
    return new_img
#==============================================================================================================
def contour(name, img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((img.shape[0], img.shape[1]))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i][j] != 0:
                #if any of the 8 neighbors is 0:
                if img[i-1,j-1] == 0 or img[i-1,j] == 0 or img[i-1,j+1] == 0 or img[i,j+1] == 0\
                or img[i+1,j+1] == 0 or img[i+1,j] == 0 or img[i+1,j-1] == 0 or img[i,j-1] == 0:
                    mask[i][j] = 255
    cv2.imwrite(path + 'pictures/' + name, mask)
    return mask              
#==============================================================================================================
def apply_k(name, k, img):
    def segment_given_k(actual_image, k, order = 0):
        image = actual_image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if order == 0:
                    cmp = image[i,j] >= k
                else:
                    cmp = image[i,j] < k
                if cmp == True:
                    image[i,j] = 0
        return image
    segmented_blue = segment_given_k(img[:,:,0].copy(), k)
    segmented_green = segment_given_k(img[:,:,1].copy(), k)
    segmented_red = segment_given_k(img[:,:,2].copy(), k)
    new_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not(segmented_blue[i][j] != 0 and segmented_green[i][j] != 0 and segmented_red[i][j] != 0):
                new_img[i][j] = 0

    cv2.imwrite(path + name, new_img)
    return new_img
#==============================================================================================================              
def main():
    #-----Lighthouse------------------------------------------------------------------------------------------------------

    img = segment('lighthouse')
    contour('lighthouse/contour_rgb.jpg', img)
    img = segment_texture('lighthouse')
    for i in range(3):
       img = segment_texture('lighthouse', i+1, img)
    contour('lighthouse/contour_txt.jpg', img)
    remove_sky_from_lighthouse(img)
    contour('lighthouse/contour_txt_removed_sky.jpg', img)
    
    #-----Ski-------------------------------------------------------------------------------------------------------------
    img = segment('ski')
    img = segment('ski', 1, img)
    contour('ski/contour_rgb_2iter.jpg', img)

    contour('lighthouse/contour_rgb.jpg', img)
    img = segment_texture('ski')
    img = segment_texture('ski', 1, img)
    contour('ski/contour_txt_1iter.jpg', img)

    #-----Baby-------------------------------------------------------------------------------------------------------------
    img = segment('baby')
    contour('baby/contour_segmented_color_iter0.jpg', img)
    img = segment('baby', 1, img)
    contour('baby/contour_segmented_color_iter1.jpg', img)

    #binary search
    apply_k('pictures/baby/binary_search177.jpg', 177, cv2.imread(path + 'pictures/baby.jpg'))
    apply_k('pictures/baby/binary_search216.jpg', 216, cv2.imread(path + 'pictures/baby.jpg'))
    img = apply_k('pictures/baby/binary_search235.jpg', 235, cv2.imread(path + 'pictures/baby.jpg'))
    contour('baby/contour_binary_search235.jpg', img)
    apply_k('pictures/baby/binary_search245.jpg', 245, cv2.imread(path + 'pictures/baby.jpg'))

    img = segment_texture('baby') #removed some of blanket
    contour('baby/contour_segmented_txt.jpg', img)
    img = process_baby(img)
    contour('baby/contour_segmented_txt_refined.jpg', img)
#==============================================================================================================              
if __name__ == '__main__':
    main()





