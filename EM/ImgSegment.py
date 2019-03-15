"""
-*- coding: utf-8 -*-
@author: Kuang Hao
"""
import numpy as np
import cv2 # opencv: https://pypi.python.org/pypi/opencv-python
from scipy.stats import multivariate_normal

def read_data(filename, is_RGB, visualize=False, save=False, save_name=None):
# read the text data file
#   data, image = read_data(filename, is_RGB) read the data file named 
#   filename. Return the data matrix with same shape as data in the file. 
#   If is_RGB is False, the data will be regarded as Lab and convert to  
#   RGB format to visualise and save.
#
#   data, image = read_data(filename, is_RGB, visualize)  
#   If visualize is True, the data will be shown. Default value is False.
#
#   data, image = read_data(filename, is_RGB, visualize, save)  
#   If save is True, the image will be saved in an jpg image with same name
#   as the text filename. Default value is False.
#
#   data, image = read_data(filename, is_RGB, visualize, save, save_name)  
#   The image filename.
#
#   Example: data, image = read_data("1_noise.txt", True)
#   Example: data, image = read_data("cow.txt", False, True, True, "segmented_cow.jpg")

    with open(filename, "r") as f:
        lines = f.readlines()

    data = []

    for line in lines:
        data.append(list(map(float, line.split(" "))))

    data = np.asarray(data).astype(np.float32)

    N, D = data.shape

    cols = int(data[-1, 0] + 1)
    rows = int(data[-1, 1] + 1)
    channels = D - 2
    img_data = data[:, 2:]

    # In numpy, transforming 1d array to 2d is in row-major order, which is different from the way image data is organized.
    image = np.reshape(img_data, [cols, rows, channels]).transpose((1, 0, 2))

    if visualize:
        if channels == 1:
            # for visualizing grayscale image
            cv2.imshow("", image)
        else:
            # for visualizing RGB image
            cv2.imshow("", cv2.cvtColor(image, cv2.COLOR_Lab2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save:
        if save_name is None:
            save_name = filename[:-4] + ".jpg"
        assert save_name.endswith(".jpg") or save_name.endswith(".png"), "Please specify the file type in suffix in 'save_name'!"

        if channels == 1:
            # for saving grayscale image
            cv2.imwrite(save_name, image)
        else:
            # for saving RGB image
            cv2.imwrite(save_name, (cv2.cvtColor(image, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))

    if not is_RGB:
        image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)

    return data, image

def write_data(data, filename):
# write the matrix into a text file
#   write_data(data, filename) write 2d matrix data into a text file named
#   filename.
#
#   Example: write_data(data, "cow.txt")

    lines = []
    for i in range(len(data)):
        lines.append(" ".join([str(int(data[i, 0])), str(int(data[i, 1]))] + ["%.6f" % v for v in data[i, 2:]]) + "\n")

    with open(filename, "w") as f:
        f.writelines(lines)

def Kmeans(obs_data,cluster,epsilon):
    # returns matrix r: expressing which observed data point belongs to which cluster
    #vectors cluster: mean vector of each cluster
    N = len(obs_data)
    K = len(cluster)
    tmp = np.ones(cluster.shape)

    def Astep(obs_data,cluster):
        #Assignment step
        r = np.zeros((N,K))
        min_ix = np.argmin(np.array([[np.linalg.norm(x-mu) for mu in cluster] for x in obs_data]),axis = 1)
        r[np.arange(N),min_ix] = 1
        return r

    def Ustep(obs_data,r):
        #Update step
        Nk= np.sum(r,axis=0)
        Nk[Nk==0]=1
        cluster = np.array([np.sum([r[n, k] * obs_data[n] for n in range(N)], axis=0) / Nk[k] for k in range(K)])
        return cluster

    while (np.abs((cluster - tmp))>epsilon).all():
        tmp = cluster.copy()
        r = Astep(obs_data,cluster)
        cluster = Ustep(obs_data,r)
    return np.reshape(cluster,(2,3))


def EM_algorithm(animal):
    # Input: file = file name to open
    data,image = read_data(animal + ".txt",True)
    N = len(data)
    d = data.shape[-1] - 2
    obs_data = data[:,2:].reshape(N,d,1)
    mus = np.random.randn(2,d,1)
    X = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
    clust = Kmeans(obs_data,mus,1e-5)
    # mean value to compute multivariate normal distribution
    mean1 = clust[1]
    mean2 = clust[0]
    # Threshold
    eps = 0.5             
    # covariance to compute multivariate normal distribution
    cov1 = np.cov(np.asarray([[13, 20, 29], [13, 23, 37], [13, 23, 29]]))

    cov2 = np.cov(np.asarray([[9, -58, 7], [8, -7, 10], [6, -4, 6]]))
    # mixing co-efficient
    mix1 = 0.4              
    mix2 = 0.6
    # number of samples
    N = image.shape[0] * image.shape[1]         

    log_likelihoods = []

    print("EM algorithm for: "+str(animal))

    # Start EM algorithm
    while(1):
        N1 = 0
        N2 = 0
        resp1_list = []
        resp2_list = []
        mu_sum1 = [0, 0, 0]
        mu_sum2 = [0, 0, 0]

        for y in image:
            for x in y:
                prob1 = multivariate_normal.pdf(x, mean=mean1, cov=cov1, allow_singular=True)      # gaussian density 1

                prob2 = multivariate_normal.pdf(x, mean=mean2, cov=cov2, allow_singular=True)      # gaussian density 2

                Numerator1 = mix1 * prob1
                Numerator2 = mix2 * prob2

                denom = Numerator1 + Numerator2
                #responsibility for 1st cluster
                resp1 = Numerator1 / denom  
                #responsibility for 2nd cluster
                resp2 = Numerator2 / denom  

                resp1_list.append(resp1)
                resp2_list.append(resp2)
                mu_sum1 += resp1 * x
                mu_sum2 += resp2 * x

                N1 += resp1
                N2 += resp2

        # Update mean values
        mu_new1 = mu_sum1 / N1
        mu_new2 = mu_sum2 / N2

        var_1 = np.zeros((3, 3))
        var_2 = np.zeros((3, 3))

        i = 0
        for y in image:
            for x in y:
                var_1 += resp1_list[i] * np.outer((x - mu_new1), (x - mu_new1))
                var_2 += resp2_list[i] * np.outer((x - mu_new2), (x - mu_new2))
                i = i + 1
        # Update covariances
        var_new1 = var_1 / N1
        var_new2 = var_2 / N2
        # Update mix co-efficients
        mix_new1 = N1 / N 
        mix_new2 = N2 / N

        mean1 = mu_new1
        mean2 = mu_new2

        cov1 = var_new1
        cov2 = var_new2

        mix1 = mix_new1
        mix2 = mix_new2
        #Calculate Log Likelihood
        Z = [0, 0]
        ll = 0
        sumList=[]
        for y in image:
            for x in y:
                prob1 = multivariate_normal.pdf(x, mu_new1, var_new1, allow_singular=True)

                prob2 = multivariate_normal.pdf(x, mu_new2, var_new2, allow_singular=True)

                sum = (mix_new1 * prob1) + (mix_new2 * prob2)
                sumList.append(np.log(sum))

            ll = np.sum(np.asarray(sumList))


        log_likelihoods.append(ll)

        if len(log_likelihoods) < 2: continue
        if np.abs(ll - log_likelihoods[-2]) < eps: break
        #Break loop if log likelihoods dont change more than threshold over 2 iterations

    print("End iteration for: " + str(animal))

    #Write to File
    back_data = data.copy()
    front_data = data.copy()
    mask_data = data.copy()

    for i in range(0,len(data)-1):

        cell = data[i]
        point = [cell[2], cell[3], cell[4]]
        prob1 = multivariate_normal.pdf(point, mean=mean1, cov=cov1, allow_singular=True)

        resp1 = mix1 * prob1
        prob2 = multivariate_normal.pdf(point, mean=mean2, cov=cov2, allow_singular=True)
        resp2 = mix2 * prob2

        resp1 = resp1/(resp1+resp2)
        resp2 = resp2/(resp1+resp2)


        if (resp1 < resp2):
            back_data[i][2] = back_data[i][3] = back_data[i][4] = 0
            mask_data[i][2] = mask_data[i][3] = mask_data[i][4] = 0

        else:
            front_data[i][2] = front_data[i][3] = front_data[i][4] = 0
            mask_data[i][2] = 100
            mask_data[i][3] = mask_data[i][4] = 0

    data_process(back_data, front_data, mask_data, animal)
    print("Finish: "+str(animal))

def data_process(data1, data2, data3, filename):
    # Write and save data for background, foreground and mask
    write_data(data1,"output/" + str(filename) + "_back.txt")
    read_data("output/" + str(filename) + "_back.txt", False, save=True, save_name="output/"+str(filename)+"_background.jpg")
    
    write_data(data2,"output/" + str(filename) + "_fore.txt")
    read_data("output/" + str(filename) + "_fore.txt", False, save=True, save_name="output/"+str(filename)+"_foreground.jpg")

    write_data(data3,"output/" + str(filename) + "_mask.txt")
    read_data("output/" + str(filename) + "_mask.txt", False, save=True, save_name="output/"+str(filename)+"_masked.jpg")


def main():
    EM_algorithm("cow")
    EM_algorithm("fox")
    EM_algorithm("owl")
    EM_algorithm("zebra")

if __name__ == "__main__":
    main()