import numpy as np
import cv2, os
from scipy import spatial
from scipy.linalg import fractional_matrix_power


path = '../Users/rmahfuz/Desktop/661/HW10/'
num_classes=30; num_samples=21
#===PCA============================================================================================================
def process_train_pca(k): #k is num of principal components. returns y_train(k x num_samples(630)), W_k
    img_arr = []
    for fn in os.listdir(path + 'ECE661_2018_hw10_DB1/train'):
        #print(fn)
        img=cv2.imread(path + 'ECE661_2018_hw10_DB1/train/' + fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#; img=np.array(img, dtype=np.float32)
        img=img.flatten()
        #img=img/np.linalg.norm(img) #normalize to make it illumination-invariant
        img_arr.append(img)
        #print(img.shape)
        #print(img_arr.shape)
    img_arr = np.array(img_arr,dtype=np.float32) 
    #print(img_arr.shape)
    mean_img = np.mean(img_arr,axis=0) #mean of each column over all rows
    #reshaped=mean_img.reshape((128,128));cv2.imwrite(path+'mean_img_train.png', reshaped)
    X=(img_arr-mean_img).T
    for i in range(X.shape[1]):
        X[:,i] = X[:,i]/np.linalg.norm(X[:,i])
    #print('X.shape = ',X.shape)#(16384,630)
    w,v = np.linalg.eig(np.matmul(X.T,X)) #eigenvalues, eigenvectors
    #print('w.shape = ', w.shape)#(630,)
    #print('v.shape = ', v.shape)#(630,630)
    v=v.T
    v=v[np.argsort(w)[::-1]]
    #w.sort(); w=w[::-1]
    v=v.T #each column is an eigenvector
    W=np.matmul(X,v)
    #Normalize each column of W:
    for i in range(W.shape[1]):
        W[:,i] = W[:,i]/np.linalg.norm(W[:,i])
    #Extract the largest/first k columns
    W_k = W[:,:k]
    #print('W_k.shape = ', W_k.shape)#(16384,k)
    #compute the projected y_train:
    y_train = np.matmul(W_k.T,X)
    #print('y_train.shape = ', y_train.shape)#(k,630). Each column represents a sample
    return (y_train, W_k)
#-----------------------------------------------------------------------------------------------------------------
def calc_pca_accuracy(k):
    #(y_train, W_k, mean_img) = process_train(k)
    (y_train, W_k) = process_train_pca(k) #(k,630)
    
    img_arr = []
    #print(os.listdir(path + 'ECE661_2018_hw10_DB1/test'))
    for fn in os.listdir(path + 'ECE661_2018_hw10_DB1/test'):
        #print(fn)
        img=cv2.imread(path + 'ECE661_2018_hw10_DB1/test/' + fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#; img=np.array(img, dtype=np.float32)
        img=img.flatten()
        #img=img/np.linalg.norm(img) #normalize to make it illumination-invariant
        img_arr.append(img)
        #print(img.shape) #(128,128)
    img_arr = np.array(img_arr,dtype=np.float32)
    '''#print('img_arr_train.shape = {}, img_arr.shape = {}'.format(img_arr_train.shape, img_arr.shape))
    #img_arr = np.vstack((img_arr_train, img_arr))
    #print(img_arr.shape)
    mean_img = np.mean(img_arr,axis=0) #mean of each column over all rows
    #reshaped=mean_img.reshape((128,128));cv2.imwrite(path+'mean_img_test.png', reshaped)
    X=(img_arr-mean_img).T
    print('X.shape = ',X.shape)
    w,v = np.linalg.eig(np.matmul(X.T,X)) #eigenvalues, eigenvectors
    print('w.shape = ', w.shape)#(630,)
    print('v.shape = ', v.shape)#(630,630)
    v=v.T
    v=v[np.argsort(w)[::-1]]
    w.sort(); w=w[::-1]
    v=v.T #each column is an eigenvector
    W=np.matmul(X,v)
    #Normalize each column of W:
    for i in range(W.shape[1]):
        W[:,i] = W[:,i]/np.linalg.norm(W[:,i])
    #Extract the largest/first k columns
    W_k = W[:,:k]
    print('W_k.shape = ', W_k.shape)#(16384,k)
    #compute the projected y_test:
    y_test = np.matmul(W_k.T,X)
    print('y_test.shape = ', y_test.shape) #(k,630)'''
    mean_img = np.mean(img_arr,axis=0) #mean of each column over all rows
    X=(img_arr-mean_img).T#(16384,630)
    for i in range(X.shape[1]):
        X[:,i] = X[:,i]/np.linalg.norm(X[:,i])
    y_test = np.matmul(W_k.T,X)
    #print('y_test.shape = ', y_test.shape) #(k,630)
    #-------------------Nearest neighbors classification---------------------------
    #Now I have both y_test and y_train. For each column in y_test, I will try to match it with the nearest column in y_train.
    acc = [0]*num_classes
    for i in range(y_test.shape[1]):
        dist = []
        for j in range(y_train.shape[1]):
            dist.append(spatial.distance.euclidean(y_test[:,i],y_train[:,j]))
        min_idx=np.argmin(dist)
        #print('dist = ', dist); print('min_idx = ',min_idx, ', dist[min_idx] = ', dist[min_idx])
        #print(int(min_idx/21))
        if int(min_idx/21) == i:
            acc[i] += 1
    #print('accuracy per class = ', acc)
    print('PCA accuracy for k = ', k, ' = ', np.sum(acc)/630)
    return acc
#-----------------------------------------------------------------------------------------------------------------
def test_pca():
    for k in range(20):
        calc_pca_accuracy(k)
#===LDA============================================================================================================
def process_train_lda(k): #k is num of components. returns y_train(k x num_samples(630)), W_k
    img_arr = []
    for fn in os.listdir(path + 'ECE661_2018_hw10_DB1/train'):
        img=cv2.imread(path + 'ECE661_2018_hw10_DB1/train/' + fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#; img=np.array(img, dtype=np.float32)
        img=img.flatten()
        img=img/np.linalg.norm(img) #normalize to make it illumination-invariant
        img_arr.append(img)
        
    img_arr = np.array(img_arr,dtype=np.float32) #each row represents an image (630x16384) 
    mean_img = np.mean(img_arr,axis=0) #mean of each column over all rows (1x16384)
    X=(img_arr-mean_img).T#(
    #Finding mean of each class
    class_means = []
    for i in range(num_classes):
        class_means.append(np.mean(img_arr[i*num_samples:i*num_samples+num_samples-1], axis=0))
    class_means = np.array(class_means, dtype = np.float32) #(30x16384)
    M=[]
    for i in range(num_classes):
        M.append(class_means[i,:] - mean_img)
    M=np.array(M, dtype=np.float32); M=M.T #(16384,30)
    w,u=np.linalg.eig(np.matmul(M.T,M))
    u=u.T

    u=u[np.argsort(w)[::-1]] #sorting in descending order
    #w.sort(); w=w[::-1]
    u=u.T #each column is an eigenvector
    V=np.matmul(M,u)
    #Normalize each column of V:
    for i in range(V.shape[1]):
        V[:,i] = V[:,i]/np.linalg.norm(V[:,i])
    #Extract the largest/first k columns into Y
    Y = V[:,:k]
    #print('Y.shape = ', Y.shape) #(16384,k)
    #Finding D_B:
    fac=np.matmul(Y.T,M); D_B = np.matmul(fac, fac.T)
    #Finding Z:
    #tmp=np.linalg.matrix_power(D_B,-1);
    tmp = fractional_matrix_power(D_B, -0.5); Z=np.matmul(Y, tmp)
    #Finding eigenvectors of Z.T*S_w*Z:
    X_w = []
    for i in range(630):
        X_w.append(img_arr[i] - class_means[int(i/21)])
    X_w = np.array(X_w, dtype=np.float32).T 
    #print('X_w.shape = ', X_w.shape)#(16384,630)
    fac = np.matmul(Z.T, X_w); S_BW = np.matmul(fac.T, fac)
    w,u = np.linalg.eig(S_BW)
    u=u.T
    u=u[np.argsort(w)] #sorting in ascending order
    #w.sort();
    u=u.T #each column is an eigenvector
    U=np.matmul(S_BW,u)
    #Normalize each column of U:
    for i in range(U.shape[1]):
        U[:,i] = U[:,i]/np.linalg.norm(U[:,i])
    #Extract the smallest/first k columns into U_hat
    U_hat = U[:,:k]
    #print('U_hat.shape = ', U_hat.shape)#(630,k)
    #Finally finding W_p:
    W_k = np.matmul(U_hat.T, Z.T).T
    y_train = np.matmul(W_k.T,X)#(k,630)
    return (y_train, W_k)
#-----------------------------------------------------------------------------------------------------------------
def calc_lda_accuracy(k):
    (y_train, W_k) = process_train_lda(k) #(k,630)
    
    img_arr = []
    for fn in os.listdir(path + 'ECE661_2018_hw10_DB1/test'):
        img=cv2.imread(path + 'ECE661_2018_hw10_DB1/test/' + fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#; img=np.array(img, dtype=np.float32)
        img=img.flatten()
        img=img/np.linalg.norm(img) #normalize to make it illumination-invariant
        img_arr.append(img)
    img_arr = np.array(img_arr,dtype=np.float32)
    mean_img = np.mean(img_arr,axis=0) #mean of each column over all rows
    X=(img_arr-mean_img).T#(16384,630)
    for i in range(X.shape[1]):
        X[:,i] = X[:,i]/np.linalg.norm(X[:,i])
    y_test = np.matmul(W_k.T,X)
    #print('y_test.shape = ', y_test.shape) #(k,630)
    #-------------------Nearest neighbors classification---------------------------
    #Now I have both y_test and y_train. For each column in y_test, I will try to match it with the nearest column in y_train.
    acc = [0]*num_classes
    for i in range(y_test.shape[1]):
        dist = []
        for j in range(y_train.shape[1]):
            dist.append(spatial.distance.euclidean(y_test[:,i],y_train[:,j]))
        min_idx=np.argmin(dist)
        #print('dist = ', dist); print('min_idx = ',min_idx, ', dist[min_idx] = ', dist[min_idx])
        #print(int(min_idx/21))
        if int(min_idx/21) == i:
            acc[i] += 1
    #print('accuracy per class = ', acc)
    print('overall accuracy for k = ', k, ' = ', np.sum(acc)/630)
    return acc
#-----------------------------------------------------------------------------------------------------------------
def test_lda():
    #process_train_lda(15)
    #calc_lda_accuracy(15)
    for k in range(20):
        calc_lda_accuracy(k)
#==================================================================================================================
def main():
    test_pca()
    #test_lda()
#-----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
