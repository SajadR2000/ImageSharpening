import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread('flowers.blur.png')
if img is None:
    print("Couldn't load the image")

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = img.astype(np.float64)

# a:
###########################################################
# Setting Parameters#######################################
###########################################################
sigma = 2.0
d = int(sigma * 3)
alpha = 2
###########################################################

###########################################################
# Creating Gaussian Filter#################################
###########################################################
x_axis = np.arange(-d, d+1).reshape((-1, 1))
y_axis = np.arange(-d, d+1).reshape((1, -1))
x_filter = np.exp(-np.square(x_axis) / 2 / sigma ** 2)
y_filter = np.exp((-np.square(y_axis) / 2 / sigma ** 2))
gaussian_filter = x_filter @ y_filter
gaussian_filter = gaussian_filter / gaussian_filter.sum()
###########################################################

###########################################################
# Saving Gaussian Filter###################################
###########################################################
gaussian_out = np.zeros((100*(2*d+1), 100*(2*d+1)))
for i in range(2*d+1):
    for j in range(2*d+1):
        gaussian_out[100*i:100*i+100, 100*j:100*j+100] = gaussian_filter[i, j]

plt.imsave('res01.jpg', gaussian_out, cmap='gray')
###########################################################

###########################################################
# Smoothed Image, Unsharp Mask, and Final Result###########
###########################################################
smoothed_img = cv.filter2D(img, -1, gaussian_filter)
unsharp_mask = img - smoothed_img
final = img + alpha * unsharp_mask
final[final > 255] = 255
final[final < 0] = 0
final = final.astype(np.uint8)
###########################################################

###########################################################
# Saving Previous Images###################################
###########################################################
# Changing range of the intensities to display negative values.
unsharp_mask_display = unsharp_mask.copy()
unsharp_mask_display = unsharp_mask_display - unsharp_mask_display.min()
unsharp_mask_display = unsharp_mask_display / unsharp_mask_display.max() * 255
unsharp_mask_display = cv.cvtColor(unsharp_mask_display.astype(np.uint8), cv.COLOR_RGB2GRAY)

plt.imsave('res02.jpg', smoothed_img.astype(np.uint8))
plt.imsave('res03.jpg', unsharp_mask_display.astype(np.uint8), cmap='gray')
plt.imsave('res04.jpg', final.astype(np.uint8))
###########################################################

# b:
###########################################################
# Setting Parameters#######################################
###########################################################
sigma = 0.7
d = int(sigma * 6)
k = 4
###########################################################

###########################################################
# Creating Gaussian Filter#################################
###########################################################
x_axis = np.arange(-d, d+1).reshape((-1, 1))
y_axis = np.arange(-d, d+1).reshape((1, -1))
x_filter = np.exp(-np.square(x_axis) / 2 / sigma ** 2)
y_filter = np.exp((-np.square(y_axis) / 2 / sigma ** 2))
gaussian_filter = x_filter @ y_filter
gaussian_filter = gaussian_filter / gaussian_filter.sum()
###########################################################

###########################################################
# Gaussian Filter Laplacian################################
###########################################################
unsharp_mask_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
gaussian_laplacian = cv.filter2D(gaussian_filter, -1, unsharp_mask_filter)
###########################################################

###########################################################
# Saving Gaussian Filter Laplacian#########################
###########################################################
gaussian_laplacian_display_temp = gaussian_laplacian - gaussian_laplacian.min()
gaussian_laplacian_display_temp = gaussian_laplacian_display_temp / gaussian_laplacian_display_temp.max() * 255
scale = 100
gaussian_laplacian_display = np.zeros((scale*(2*d+1), scale*(2*d+1)))
for i in range(2*d+1):
    for j in range(2*d+1):
        gaussian_laplacian_display[scale*i:scale*i+scale, scale*j:scale*j+scale] = gaussian_laplacian_display_temp[i, j]
plt.imsave('res05.jpg', gaussian_laplacian_display.astype(np.uint16), cmap='gray')
###########################################################

###########################################################
# Unsharp Mask and Final Result
###########################################################
unsharp_mask_b = cv.filter2D(img, -1, gaussian_laplacian)
final_result_b = img - k * unsharp_mask_b
final_result_b[final_result_b > 255] = 255
final_result_b[final_result_b < 0] = 0
###########################################################

###########################################################
# Saving Previous Results##################################
###########################################################
unsharp_mask_b_display = unsharp_mask_b - unsharp_mask_b.min()
unsharp_mask_b_display = unsharp_mask_b_display / unsharp_mask_b_display.max() * 255
unsharp_mask_b_display = cv.cvtColor(unsharp_mask_b_display.astype(np.uint8), cv.COLOR_RGB2GRAY)
plt.imsave('res06.jpg', unsharp_mask_b_display.astype(np.uint8), cmap='gray')
plt.imsave('res07.jpg', final_result_b.astype(np.uint8))
###########################################################


# c:
###########################################################
# Setting Parameters#######################################
###########################################################
D0 = 100
k_c = 4
###########################################################

###########################################################
# Calculating dft of the image#############################
###########################################################
img_dft = np.zeros(img.shape, dtype=np.complex128)
for i in range(3):
    img_dft[:, :, i] = np.fft.fftshift(np.fft.fft2(img[:, :, i]))
###########################################################

###########################################################
# Saving dft of the image##################################
###########################################################
# Note the I save log-magnitude of the image dft
log_abs_img_dft = np.log(np.abs(img_dft))
log_abs_img_dft = log_abs_img_dft / log_abs_img_dft.max() * 255
log_abs_img_dft = log_abs_img_dft.astype(np.uint8)
plt.imsave("res08.jpg", log_abs_img_dft)
###########################################################

###########################################################
# Creating high pass filter################################
###########################################################
x_axis = np.arange(img.shape[0]) - img.shape[0] // 2
y_axis = np.arange(img.shape[1]) - img.shape[1] // 2
x_filter = np.exp(-np.square(x_axis) / 2 / D0 ** 2).reshape((-1, 1))
y_filter = np.exp(-np.square(y_axis) / 2 / D0 ** 2).reshape((1, -1))
lpf = x_filter @ y_filter
hpf = 1 - lpf
###########################################################

###########################################################
# Saving high pass filter##################################
###########################################################
plt.imsave('res09.jpg', hpf, cmap='gray')
###########################################################

###########################################################
# This is for parameter selection##########################
###########################################################
smoothed_img_dft = np.zeros(img_dft.shape, dtype=np.complex128)
for i in range(3):
    smoothed_img_dft[:, :, i] = img_dft[:, :, i] * lpf

smoothed_img = np.zeros(img.shape, dtype=np.float64)
for i in range(3):
    smoothed_img[:, :, i] = np.real(np.fft.ifft2(np.fft.ifftshift(smoothed_img_dft[:, :, i])))
###########################################################

###########################################################
# Sharpening the image#####################################
###########################################################
sharp_img_dft = np.zeros(img.shape, dtype=np.complex128)
for i in range(3):
    sharp_img_dft[:, :, i] = (1 + k_c * hpf) * img_dft[:, :, i]

sharp_img = np.zeros(img.shape, dtype=np.float64)
for i in range(3):
    sharp_img[:, :, i] = np.real(np.fft.ifft2(np.fft.ifftshift(sharp_img_dft[:, :, i])))
sharp_img[sharp_img > 255] = 255
sharp_img[sharp_img < 0] = 0
sharp_img = sharp_img.astype(np.uint8)
###########################################################

###########################################################
# Saving final results#####################################
###########################################################
sharp_img_dft_log_magnitude = np.log(np.abs(sharp_img_dft))
sharp_img_dft_log_magnitude = sharp_img_dft_log_magnitude / sharp_img_dft_log_magnitude.max() * 255
sharp_img_dft_log_magnitude = sharp_img_dft_log_magnitude.astype(np.uint8)
plt.imsave('res10.jpg', sharp_img_dft_log_magnitude)
plt.imsave('res11.jpg', sharp_img)
###########################################################

# d:
###########################################################
# Setting Parameters#######################################
###########################################################
k_d = 2
###########################################################

###########################################################
# Filter###################################################
###########################################################
x_axis = np.arange(img.shape[0]) - img.shape[0] // 2
x_axis = x_axis.reshape((-1, 1))
y_axis = np.arange(img.shape[1]) - img.shape[1] // 2
y_axis = y_axis.reshape((1, -1))
x_filter = np.repeat(x_axis, img.shape[1], axis=1)
y_filter = np.repeat(y_axis, img.shape[0], axis=0)
filter_dft = np.square(x_filter) + np.square(y_filter)
filter_dft = 4 * np.pi ** 2 * filter_dft
###########################################################

###########################################################
# Filtering################################################
###########################################################
img_dft = np.zeros(img.shape, dtype=np.complex128)
for i in range(3):
    img_dft[:, :, i] = np.fft.fftshift(np.fft.fft2(img[:, :, i]))

unsharp_mask_dft = np.zeros(img.shape, dtype=np.complex128)
for i in range(3):
    unsharp_mask_dft[:, :, i] = filter_dft * img_dft[:, :, i]

unsharp_mask_dft_display = unsharp_mask_dft.copy()
unsharp_mask_dft_display[unsharp_mask_dft_display == 0] = 1
unsharp_mask_dft_log_abs = np.log(np.abs(unsharp_mask_dft_display))
unsharp_mask_dft_log_abs = unsharp_mask_dft_log_abs / unsharp_mask_dft_log_abs.max() * 255

unsharp_mask = np.zeros(img.shape, dtype=np.float64)
for i in range(3):
    unsharp_mask[:, :, i] = np.real(np.fft.ifft2(np.fft.ifftshift(unsharp_mask_dft[:, :, i])))

unsharp_mask_display = unsharp_mask.copy()
unsharp_mask_display = unsharp_mask_display - unsharp_mask_display.min()
unsharp_mask_display = unsharp_mask_display / unsharp_mask_display.max() * 255
unsharp_mask_display = cv.cvtColor(unsharp_mask_display.astype(np.uint8), cv.COLOR_RGB2GRAY)

unsharp_mask = unsharp_mask / unsharp_mask.max() * 255

sharp_img = img + k_d * unsharp_mask
sharp_img[sharp_img > 255] = 255
sharp_img[sharp_img < 0] = 0
###########################################################

###########################################################
# Saving the Results#######################################
###########################################################
plt.imsave('res12.jpg', unsharp_mask_dft_log_abs.astype(np.uint8))
plt.imsave('res13.jpg', unsharp_mask_display.astype(np.uint8), cmap='gray')
plt.imsave('res14.jpg', sharp_img.astype(np.uint8))
###########################################################
