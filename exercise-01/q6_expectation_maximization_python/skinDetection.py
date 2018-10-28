import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood
import math 
pi = np.pi

def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes = skin_K = 3
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    n_weights, n_means, n_covariances = estGaussMixEM(ndata, K, n_iter, epsilon)
    s_weights, s_means, s_covariances = estGaussMixEM(sdata, K, n_iter, epsilon)
 
    s_N = ndata.shape[0]
    n_N = sdata.shape[0]
    img_row = img.shape[0]
    img_col = img.shape[1]
    D = ndata.shape[1]
    s_pdf = np.zeros([img_row, img_col])
    n_pdf = np.zeros([img_row, img_col])
    result = np.zeros([img_row, img_col, D])

    #print('skin weights:\n', s_weights, 'skin means:\n', s_means, 'skin covariances:\n', s_covariances)
    #print('non-skin weights:\n', n_weights, 'non-skin means:\n', n_means, 'non-skin covariances:\n', n_covariances)
    
    for row in range(img_row):
        for col in range(img_col):
            s_pdf[row][col] = getPdf(s_means, s_covariances, img[row][col], K, D)
            n_pdf[row][col] = getPdf(n_means, n_covariances, img[row][col], K, D)
            #print('s_pdf[row][col]:\n', s_pdf[row][col])
            #print('n_pdf[row][col]:\n', n_pdf[row][col])
            if (s_pdf[row][col] - 2 * n_pdf[row][col]) > theta:
                # this pixel probably has skin color, so categorize this as white
                result[row][col] = [255, 255, 255]
            else:
                # this pixel probably does NOT have skin color, so categorize this as black
                result[row][col] = [0, 0, 0]

    #####Insert your code here for subtask 1g#####
    return result

def getPdf(means, covariances, X, K, D):
    pdf = 0

    for k in range(0, K, 1):
        dis = np.matrix(X - means[k])
        inv_cov = np.linalg.inv(covariances[:, :, k])
        det_cov = np.linalg.det(covariances[:, :, k])
        mul = np.linalg.multi_dot([dis, inv_cov, dis.T])
        pdf =+ (1/(pow(2 * pi, D/2) * math.sqrt(det_cov))) * math.exp(-1/2 * mul)

    return pdf

