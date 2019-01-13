import numpy as np
import cv2, os
from scipy import spatial

#path = '../Users/rmahfuz/Desktop/661/HW10/'
path = ''
num_max_strong=10 #maximum number of strong classifiers permitted
num_max_weak=100 #maximum number of weak classifiers permitted
thresh_positive = 1 #acceptable positive detection rate
thresh_falsePositive = 0.5 #acceptable false positive rate
#=======================================================================
def find_features(filePath, saveName):
    def findRec(intImg, corner):
        one = intImg[int(corner[0,0]), int(corner[0,1])]
        two = intImg[int(corner[1,0]),int(corner[1,1])]
        three = intImg[int(corner[2,0]), int(corner[2,1])]
        four = intImg[int(corner[3,0]), int(corner[3,1])]
        return four+one-two-three
    feature=[]
    h_size=np.linspace(2,20,10, dtype=np.int32)
    v_size=np.linspace(2,40,20, dtype=np.int32)
    for fn in os.listdir(path+filePath):
        #print(fn)
        img=cv2.imread(path+filePath+fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Finding integral representation of image:
        intImg = np.zeros((img.shape[0]+1, img.shape[1]+1))
        intImg[1:,1:] = np.cumsum(np.cumsum(img,axis=0),axis=1)
        feature_temp = []
        #Find horizontal feature:
        for j in range(len(h_size)):
            width=h_size[j]
            for k in range(img.shape[0]):
                for m in range(img.shape[1]-int(width)+1):
                    corner0 = np.array([[k,m],[k,m+width/2],
                                       [k+1,m],[k+1,m+width/2]])
                    corner1 = np.array([[k,m+width/2],[k,m+width],
                                       [k+1,m+width/2],[k+1,m+width]])
                    rec0=findRec(intImg, corner0)
                    rec1=findRec(intImg, corner1)
                    feature_temp.append(rec1-rec0)
        #Find vertical feature:
        for j in range(len(v_size)):
            height=v_size[j]
            for k in range(img.shape[0]-int(height)+1):
                for m in range(img.shape[1]-1):
                    corner1 = np.array([[k,m],[k,m+2],
                                       [k+height/2,m],[k+height/2,m+2]])
                    corner0 = np.array([[k+height/2,m],[k+height/2,m+2],
                                       [k+height,m],[k+height,m+2]])
                    rec1=findRec(intImg, corner1)
                    rec0=findRec(intImg, corner0)
                    feature_temp.append(rec1-rec0)
        feature.append(feature_temp)
    feature = np.array(feature).T
    np.save(path + saveName, feature)#write feature into file
    print('feature.shape = ', feature.shape)
    return feature
#=======================================================================
def adaBoost(feature_all, num_pos, current_idx, stage, num_max_weak):
    def find_bestWeak(feature, weights, labels, num_pos):
        bestWeak = dict()
        (num_features, num_img) = feature.shape
        T_plus = np.repeat(np.sum(weights[:num_pos]), num_img)
        T_minus = np.repeat(np.sum(weights[num_pos:]), num_img)
        bestWeak['min_err'] = np.inf
        for i in range(num_features):
            current_feature = feature[i,:]
            sorted_features = np.sort(current_feature)
            sorted_feature_idx = np.argsort(current_feature)
            sorted_weights = weights[sorted_feature_idx]
            sorted_labels = labels[sorted_feature_idx]
            S_plus = np.cumsum(sorted_weights*sorted_labels)
            S_minus = np.cumsum(sorted_weights) - S_plus
            error1 = S_plus + (T_minus - S_minus)
            error2 = S_minus + (T_plus - S_plus)
            #print('len(error1) = {}, len(error2) = {}'.format(len(error1), len(error2)))
            e = []
            for j in range(len(error1)):
                e.append(min(error1[j], error2[j]))
            #e = np.min(error1, error2) #finding the error
            min_error = np.min(e) #finding best threshold
            thresh = np.argmin(e)
            polarity = -1 if error1[thresh] <= error2[thresh] else 1
            #obtain classification result
            classification_result = np.zeros((num_img, 1))
            if polarity == -1:
                classification_result[thresh:] = 1
            else:
                classification_result[:thresh] = 1
            classification_result[sorted_feature_idx] = classification_result

            if min_error < bestWeak['min_err']:
                bestWeak['min_err'] = min_error
                bestWeak['polarity'] = polarity
                bestWeak['feature'] = i
                bestWeak['result'] = classification_result
                if thresh == 0: #a little smaller than the smallest
                    bestWeak['thresh'] = sorted_features[thresh] - 0.001
                elif thresh == len(sorted_features): #a little larger than the largest
                    bestWeak['thresh'] = sorted_features[thresh] + 0.001
                else: #between that feature value and the previous feature value
                    bestWeak['thresh'] = 0.5*(sorted_features[thresh] + sorted_features[thresh-1])
        return bestWeak
    #--------------------------------------------------------------------
    feature = feature_all[:, current_idx]
    num_neg = len(current_idx) - num_pos
    #Initializing weights and labels:
    weights = [0.5*(1.0/num_pos)]*num_pos
    weights.extend([0.5*(1.0/num_neg)]*num_neg)
    labels = [1]*num_pos; labels.extend([0]*num_neg); labels=np.array(labels)

    alpha = np.zeros((num_max_weak, 1))
    weak = np.zeros((4, num_max_weak))
    weak_result = np.zeros((len(current_idx), num_max_weak)) #(feature, thresh, polarity, alpha)
    strong_result = np.zeros((len(current_idx),1))
    positive_accuracy = [] #of strong classifier at the end of each stage
    negative_FP = [] #of strong classifier at the end of each stage

    for t in range(num_max_weak):
	#print('weights = {}, np.sum(weights) = {}'.format(weights, np.sum(weights)))
        weights = weights/np.sum(weights) #normalizing the weights
        best_weak = find_bestWeak(feature, weights, labels, num_pos)
        err = best_weak['min_err']
        weak[:3,t] = [best_weak['feature'], best_weak['thresh'], best_weak['polarity']]
        weak_result[:,t] = best_weak['result'].flatten()
        #compute beta
        beta = err/(1-err)
        alpha[t,0] = np.log(1/beta)
        weak[3,t] = alpha[t,0]
        #update weights
        e = []
        for j in range(len(weak_result)):
            e.append(int(weak_result[j,t] == labels[j]))
        e = np.array(e)
        #e = np.array([weak_result==labels], dtype = np.int32)
        print('e = {}'.format(e))
        for i in range(len(weights)):
            weights[i] = weights[i]*pow(beta,1-e[i])
        #compute strong classifier result
        strong_tmp = np.matmul(weak_result[:,:t], alpha[:t,0])
        thresh = np.min(strong_tmp[:num_pos])
        for i in range(len(current_idx)):
            strong_result[i] = 1 if strong_tmp[i] >=thresh else 0
        positive_accuracy.append(np.sum(strong_result[:num_pos])/num_pos)
        negative_FP.append(np.sum(strong_result[num_pos:])/num_neg)
        if positive_accuracy[t] >= thresh_positive and negative_FP[t] <= thresh_falsePositive:
            break
    strong = dict()
    strong['updated_idx'] = np.arange(num_pos, dtype = np.int32).tolist()
    remaining = np.nonzero(strong_result[num_pos:])[0]+num_pos
    strong['updated_idx'].extend(remaining.tolist()) #the images classified as positive
    strong['num_weak'] = t #number of weak classifiers
    strong['parameters'] = weak #collection of weak classifiers
    return strong
#=======================================================================
def train():
    feature_pos = np.load(path + 'train_pos.npy')
    feature_neg = np.load(path + 'train_neg.npy')
    feature_all = np.hstack((feature_pos, feature_neg))
    num_pos = feature_pos.shape[1]
    num_neg = feature_neg.shape[1]
    train_result = []
    current_idx = np.arange(num_pos+num_neg)
    for i in range(num_max_strong):
        print('starting stage {}'.format(i))
        strong = adaBoost(feature_all, num_pos, current_idx, i, num_max_weak)
        current_idx = strong['updated_idx']
        neg_idx = []
        for j in range(len(current_idx)):
            if current_idx[j] > num_pos:
                neg_idx.append(j)
        train_result.append(strong)
        if len(neg_idx) == 0:
            break
        num_pos = len(current_idx) - len(neg_idx)
    np.save(path + 'training_result.npy', np.array(train_result))
#=======================================================================
def test():
    def strong_predict(feature_pos, feature_neg, feature_idx, thresh, polarity, alpha, num_weak):
	feature_all = np.hstack((feature_pos, feature_neg))
	num_pos = feature_pos.shape[1]
	num_neg = feature_neg.shape[1]
	num_img = num_pos + num_neg
	#calculating weak classifier result
	weak_result = np.zeros((num_img, num_weak))
	for i in range(num_weak):
	    current_feature = feature_all[int(feature_idx[i]),:]
	    for j in range(num_img):
		if polarity[i]*current_feature[j] <= polarity[i]*thresh[i]:
		    weak_result[j,i] = 1 #otherwise zero by default
	#calculating strong classifier result
	strong_result = np.zeros((num_img, 1))
	strong_tmp = np.matmul(weak_result, alpha.T)
	strong_thresh = 0.5*np.sum(alpha)
	for i in range(num_img):
	    if strong_tmp[i] >= strong_thresh:
		strong_result[i] = 1
	return strong_result
    #-----------------------------------------------------------------------
    feature_pos = np.load(path + 'test_pos.npy')
    feature_neg = np.load(path + 'test_neg.npy')
    train_result = np.load(path + 'training_result.npy')
    num_test_pos = feature_pos.shape[1]
    num_test_neg = feature_neg.shape[1]
    num_stages = len(train_result)
    false_positive = 0; true_negative = 0
    fp = np.zeros((num_stages, 1)); fn = np.zeros((num_stages, 1))
    for i in range(num_stages):
        current_stage = train_result[i]
        num_weak = current_stage['num_weak']
        weak = current_stage['parameters'] #collection of weak classifiers
        feature_idx = weak[0,:num_weak]
        weak_thresh = weak[1,:num_weak]
        polarity = weak[2,:num_weak]
        alpha = weak[3,:num_weak]
        predicted_labels = strong_predict(feature_pos, feature_neg, feature_idx,
                                      weak_thresh, polarity, alpha, num_weak)
        num_pos = feature_pos.shape[1]
        num_neg = feature_neg.shape[1]
        #calculating false positive and false negative for this stage
	#print(predicted_labels)
	if len(np.nonzero(predicted_labels[:num_pos])[0]) == 0:
	    false_positive += 0
	else:
	    false_positive += num_pos-len(np.nonzero(predicted_labels[:num_pos])[0])
	if len(np.nonzero(predicted_labels[num_pos:])[0]) == 0:
	    true_negative += 0
	else:
	    true_negative += num_neg - len(np.nonzero(predicted_labels[num_pos:])[0])
        fp[i] = (num_test_neg-true_negative)/num_test_neg #misclassified negative
        fn[i] = false_positive/num_test_pos #misclassified positive
        #update features
        feature_pos = feature_pos[:,np.nonzero(predicted_labels[:num_pos])[0]]
        feature_neg = feature_neg[:,np.nonzero(predicted_labels[num_pos:])[0]]
    print('fp = ', fp)
    print('fn = ', fn)
#=======================================================================
def main():
    '''find_features('ECE661_2018_hw10_DB2/train/positive/', 'train_pos.npy')
    find_features('ECE661_2018_hw10_DB2/train/negative/', 'train_neg.npy')
    find_features('ECE661_2018_hw10_DB2/test/positive/', 'test_pos.npy')
    find_features('ECE661_2018_hw10_DB2/test/negative/', 'test_neg.npy')'''
    #train()
    test()

if __name__ == '__main__':
    main()
