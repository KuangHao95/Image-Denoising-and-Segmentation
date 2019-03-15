# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:05:23 2018

@author: Zhang Liquan
"""

import numpy as np
import cv2 # opencv: https://pypi.python.org/pypi/opencv-python
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import expit as sigmoid
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

# loading data/image with noise
data, image = read_data("1_noise.txt", is_RGB = True,visualize=False)

# tranform 3d to 2d array with respective color indicator
A = image[:,:,0]
image_binary = +1*(A == 255) + -1*(A == 0)
[M, N] = image_binary.shape

#mean-field parameters
nl  = 2  #noise level
y = image_binary # + nl*np.random.randn(M, N) #y_i ~ N(x_i; sigma^2);
W = 1  #coupling strength (w_ij)
rate = 0.5  #update smoothing rate
max_iter = 15 # number of iteratoins
LB = np.zeros(max_iter) # lower bound of maximum likelihood


Lp = np.reshape(multivariate_normal.logpdf(y.flatten(), mean=+1, cov=nl**2), (M, N)) # positive
Ln = np.reshape(multivariate_normal.logpdf(y.flatten(), mean=-1, cov=nl**2), (M, N)) # negative
logodds = Lp - Ln
mv = 2*sigmoid(logodds) - 1 # initial mean value of node i
a = mv + 0.5 * (logodds)
qp = sigmoid(+2*a)  #q_i(x_i=+1)
qn = sigmoid(-2*a)  #q_i(x_i=-1)

#for i in tqdm(range(max_iter)):
for i in tqdm(range(max_iter)):
    mvNew = mv
    for ix in range(N):
        for iy in range(M):
            pos = iy + M*ix
            neighborhood = pos + np.array([-1,1,-M,M])            
            boundary_idx = [iy!=0,iy!=M-1,ix!=0,ix!=N-1]
            neighborhood = neighborhood[np.where(boundary_idx)[0]]            
            xx, yy = np.unravel_index(pos, (M,N), order='F')
            nx, ny = np.unravel_index(neighborhood, (M,N), order='F')
            
            Mfi = W*np.sum(mv[nx,ny])  # mean filed influence on node i      
            mvNew[xx,yy] = (1-rate)*mvNew[xx,yy] + rate*np.tanh(Mfi + 0.5*logodds[xx,yy])
            LB[i] = LB[i] + 0.5*(Mfi * mvNew[xx,yy])

    mv = mvNew
            
    a = mv + 0.5 * logodds
    qp = sigmoid(+2*a) #q_i(x_i=+1)
    qn = sigmoid(-2*a) #q_i(x_i=-1)         
                                             
    LB[i] = LB[i] + np.sum(qp*Lp + qn*Ln) #+ np.sum(Hx)
            
#end for
mv_binary = +250*(mv>0)+ 0*(mv<0)  

plt.figure()
plt.imshow(mv_binary)

plt.figure()
plt.plot(LB, color='b', lw=2.0)

denoise = data
denoise[:,2] = mv_binary.flatten(order='F')

write_data(denoise,"1_denoise.txt")
