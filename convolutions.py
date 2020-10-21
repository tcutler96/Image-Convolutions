import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random


# convolves image with given kernel
def convolve(image, kernel):
    input = image.copy()
    # dimensions of image and kernel
    image_height, image_width = input.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]
    # reshapes kernel to match image shape
    if input.ndim == 3:
        kernel = kernel.reshape(kernel.shape[0], kernel.shape[1], 1)
        if input.shape[2] == 4:
            # removes alpha channel from rgb image
            input = np.delete(input, -1, axis=2)
    # creates empty array for output image
    output = np.zeros_like(input)
    # adds border padding to image so kernel window is always well contained within image
    x_pad = (kernel_width - 1) // 2
    y_pad = (kernel_height - 1) // 2
    # edge options are 'constant', 'edge' 'mean', 'wrap'
    padding_option = 'edge'
    if input.ndim == 3:
        input = np.pad(input, ((y_pad, y_pad), (x_pad, x_pad), (0, 0)), padding_option)
    elif input.ndim == 2:
        input = np.pad(input, ((y_pad, y_pad), (x_pad, x_pad)), padding_option)
    # loops over image
    for height in range(y_pad, image_height + y_pad):
        for width in range(x_pad, image_width + x_pad):
            # region effected by kernel
            region = input[height - y_pad:height + y_pad + 1, width - x_pad:width + x_pad + 1]
            # apply the kernel the the region
            applied = np.multiply(region, kernel)
            # sum respective values
            summed = applied.sum(axis=1).sum(axis=0)
            # store values on output
            output[height - y_pad, width - x_pad] = summed
    return output


# applies multiple kernels to an image in turn or individually
def multi_convolve(image, kernels, in_turn=True):
    if in_turn:
        for kernel in kernels:
            output = convolve(image, kernel)
    else:
        output = []
        for kernel in kernels:
            output.append(convolve(image, kernel))
    return output


# colour maps rgb image to grey scale
def rgb_to_grey(input):
    if input.ndim == 3:
        output = np.dot(input[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        output = input
    return output


# colour maps hsv image to rgb
def hsv_to_rgb(input):
    h = np.floor(input[..., 0] / 60.0) % 6
    v = input[..., 2].astype('float')
    f = (input[..., 0] / 60.0) - np.floor(input[..., 0] / 60.0)
    p = v * (1.0 - input[..., 1])
    q = v * (1.0 - (f * input[..., 1]))
    t = v * (1.0 - ((1.0 - f) * input[..., 1]))

    output = np.zeros(input.shape)
    output[h == 0, :] = np.dstack((v, t, p))[h == 0, :]
    output[h == 1, :] = np.dstack((q, v, p))[h == 1, :]
    output[h == 2, :] = np.dstack((p, v, t))[h == 2, :]
    output[h == 3, :] = np.dstack((p, q, v))[h == 3, :]
    output[h == 4, :] = np.dstack((t, p, v))[h == 4, :]
    output[h == 5, :] = np.dstack((v, p, q))[h == 5, :]
    return output


# ensures kernel size is a positive, odd integer
def validate_size(size):
    size = int(size)
    if size < 1:
        size = 1
    if size % 2 == 0:
        size -= 1
    return size


# gets identity kernel
def identity():
    kernel = np.array([[1]])
    return kernel


# gets sharpening kernel
def sharpen():
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return kernel


# gets box blur kernel
def box_blur(size, separate=True):
    size = validate_size(size)
    if separate:  # 1d kernels
        kernel = np.full((size, 1), 1 / size)
        kernel = [kernel, kernel.transpose((1, 0))]
    else:  # 2d kernel
        kernel = np.full((size, size), 1 / (size * size))
    return kernel


# gets mesh array of coordinates
def get_coord_array(x_size, y_size):
    x_half = x_size // 2
    y_half = y_size // 2
    x_coords, y_coords = np.meshgrid(np.linspace(-y_half, y_half, y_size), np.linspace(-x_half, x_half, x_size))
    coords = np.array([x_coords, y_coords])
    return coords


# computes standard deviation from size of kernel
def get_sigma(size):
    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    return sigma


# gets gaussian blur kernel
def gaussian_blur(size, separate=True):
    size = validate_size(size)
    # computes standard deviation from size of kernel
    sigma = get_sigma(size)
    if separate:  # 1d kernels
        coords = get_coord_array(size, 1)
        # applies formula
        kernel = (1 / np.sqrt(2 * np.pi * sigma * sigma)) * (np.e ** -(coords[1, :, :] * coords[1, :, :] / (
                2 * sigma * sigma)))
        # scales kernel values such that they sum to 1
        kernel /= sum(sum(kernel))
        kernel = [kernel, kernel.transpose((1, 0))]
    else:  # 2d kernel
        coords = get_coord_array(size, size)
        # applies formula
        kernel = (1 / (2 * np.pi * sigma * sigma)) * (np.e ** - ((coords[0, :, :] * coords[0, :, :] +
                                                                  coords[1, :, :] * coords[1, :, :]) /
                                                                 (2 * sigma * sigma)))
        # scales kernel values such that they sum to 1
        kernel /= sum(sum(kernel))
    return kernel


# gets the sobel operator
def sobel(separate=False):
    if separate:  # 1d kernels (x1, x2, y1, y2)
        kernel = [np.array([[-1], [-2], [-1]]), np.array([[1, 0, -1]]),
                  np.array([[1], [0], [-1]]), np.array([[-1, -2, -1]])]
    else:  # 2d kernels (x, y)
        kernel = [np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                  np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])]
    return kernel


# gets the prewitt operator
def prewitt(separate=False):
    if separate:  # 1d kernels (x1, x2, y1, y2)
        kernel = [np.array([[1], [1], [1]]), np.array([[-1, 0, 1]]),
                  np.array([[1], [0], [-1]]), np.array([[-1, -1, -1]])]
    else:  # 2d kernels (x, y)
        kernel = [np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                  np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])]
    return kernel


# gets the laplacian operator
def laplacian():  # 2d kernels (x, y)
    kernel = [np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
              np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
              np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]])]
    return kernel


# allocates any remaining kernel sum such that it sums to zero and remains symmetrical
def allocate_kernel_sum(kernel, size):
    while sum(sum(kernel)) != 0:
        kernel_sum = sum(sum(kernel))
        kernel_sign = np.sign(kernel_sum)
        if abs(kernel_sum) > 8:  # allocates to ring around center element
            kernel[size // 2 - 1: size // 2 + 2, size // 2 - 1: size // 2 + 2] -= 1 * kernel_sign
            kernel[size // 2, size // 2] += 1 * kernel_sign
        elif abs(kernel_sum) > 4:  # allocates to adjacent neighbours of center element
            kernel[size // 2 - 1, size // 2] -= 1 * kernel_sign
            kernel[size // 2 + 1, size // 2] -= 1 * kernel_sign
            kernel[size // 2, size // 2 - 1] -= 1 * kernel_sign
            kernel[size // 2, size // 2 + 1] -= 1 * kernel_sign
        else:  # allocates to center element
            kernel[size // 2, size // 2] -= 1 * kernel_sign
    return kernel


# gets the laplacian of gaussian operator
def laplacian_of_gaussian(size):
    size = validate_size(size)
    # computes standard deviation from size of kernel
    sigma = get_sigma(size)
    coords = get_coord_array(size, size)
    # applies formula
    kernel = (1 - ((coords[0, :, :] * coords[0, :, :] + coords[1, :, :] * coords[1, :, :]) /
                   (2 * sigma * sigma))) * (np.e ** - ((coords[0, :, :] * coords[0, :, :] +
                                                        coords[1, :, :] * coords[1, :, :]) / (2 * sigma * sigma)))
    kernel = np.around(kernel, 5)
    # scales outer most but one kernel values to 1
    kernel *= 1 / kernel[1, 0]
    kernel = np.rint(kernel)
    # allocates remaining sum
    kernel = allocate_kernel_sum(kernel, size)
    return kernel


# gets the gradient magnitude of edges
def edge_magnitude(x, y):
    output = np.sqrt(x ** 2 + y ** 2)
    return output


# gets the angle orientation of edges
def edge_angle(x, y):
    output = np.arctan2(y, x)
    # converts angles to degrees
    output *= 180 / np.pi
    return output


# colours edges according to their angle
def edge_colour(mag, ang, scale=1, randomise=False):
    angle = ang.copy()
    # scales colour domain
    if scale != 1:
        angle *= scale
    # randomises colour domain
    if randomise:
        angle = angle + random.randint(-180, 180)
    output = np.ones((angle.shape[0], angle.shape[1], 3))  # saturation value set to 1
    output[:, :, 0] = angle  # angle taken as hue value
    output[:, :, 2] = mag  # magnitude taken as brightness value
    output = hsv_to_rgb(output)  # convert to rgb colour space
    return output


# applies non maximum suppression to thin out edges
def non_max_suppression(mag, ang):
    magnitude = mag.copy()
    angle = ang.copy()
    image_height, image_width = magnitude.shape
    output = np.zeros((image_height, image_width))
    magnitude = np.pad(magnitude, ((1, 1), (1, 1)))
    # make angles positive and round to nearest 45 degrees
    angle[angle < 0] += 180
    angle[(angle > 157.5) | (angle <= 22.5)] = 0
    angle[(angle > 22.5) & (angle <= 67.5)] = 45
    angle[(angle > 67.5) & (angle <= 112.5)] = 90
    angle[(angle > 112.5) & (angle <= 157.5)] = 135
    # loop over all pixels
    for height in range(1, image_height + 1):
        for width in range(1, image_width + 1):
            # get current pixels magnitude and angle
            pixel_mag = magnitude[height, width]
            pixel_angle = angle[height - 1, width - 1]
            # get magnitude of neighbour pixels in direction of angle
            if pixel_angle == 0:
                neighbour_mag_1 = magnitude[height, width - 1]
                neighbour_mag_2 = magnitude[height, width + 1]
            elif pixel_angle == 45:
                neighbour_mag_1 = magnitude[height - 1, width - 1]
                neighbour_mag_2 = magnitude[height + 1, width + 1]
            elif pixel_angle == 90:
                neighbour_mag_1 = magnitude[height - 1, width]
                neighbour_mag_2 = magnitude[height + 1, width]
            elif pixel_angle == 135:
                neighbour_mag_1 = magnitude[height - 1, width + 1]
                neighbour_mag_2 = magnitude[height + 1, width - 1]
            # keep current pixel magnitude if it is the most intense among neighbours, else leave as zero
            if pixel_mag >= neighbour_mag_1 and pixel_mag >= neighbour_mag_2:
                output[height - 1, width - 1] = pixel_mag
    return output


# applies single/ double value thresholding to identity strong/ weak pixels
def threshold(image, th_ratio_high=0.5, th_ratio_low=None):
    strong_value = 1
    weak_value = 0.25
    output = np.zeros(image.shape)
    # strong pixels
    th_high = np.max(image) * th_ratio_high
    output[image >= th_high] = strong_value
    # weak pixels
    if th_ratio_low is not None:
        th_low = np.max(image) * th_ratio_low
        output[(image < th_high) & (image >= th_low)] = weak_value
    return output


# applies hysteresis to track edges
def hysteresis(image):
    strong_value = 1
    weak_value = 0.25
    input = image.copy()
    input = np.pad(input, ((1, 1), (1, 1)))
    output = image.copy()
    # get weak pixels
    weak = np.where(input == weak_value)
    # loop through weak pixels
    for index, height in enumerate(weak[0]):
        width = weak[1][index]
        # checks if pixel is surrounded by a strong pixel
        if np.any(input[height - 1:height + 2, width - 1:width + 2] == strong_value):  # keep
            output[height - 1, width - 1] = strong_value
        else:  # discard
            output[height - 1, width - 1] = 0
    return output


# applies the canny process
def canny(image, colour=False, return_all=False):
    input = image.copy()
    grey = rgb_to_grey(input)
    if return_all:
        output = [[grey, 'Grey Scale']]
    g_blur = convolve(grey, gaussian_blur(5, False))
    if return_all:
        output.append([g_blur, 'Gaussian Blur'])
    kernel = sobel(False)
    sx = convolve(g_blur, kernel[0])
    if return_all:
        output.append([sx, 'Sobel X'])
    sy = convolve(g_blur, kernel[1])
    if return_all:
        output.append([sy, 'Sobel Y'])
    mag = edge_magnitude(sx, sy)
    if return_all:
        output.append([mag, 'Magnitude'])
    ang = edge_angle(sx, sy)
    n_max = non_max_suppression(mag, ang)
    if return_all:
        output.append([n_max, 'Non-Maximum Supression'])
    d_th = threshold(n_max, 0.15, 0.05)
    if return_all:
        output.append([d_th, 'Double Threshold'])
    hyst = hysteresis(d_th)
    if return_all:
        output.append([hyst, 'Hysteresis'])
    else:
        output = [[hyst, 'Hysteresis']]
    if colour:
        col = edge_colour(hyst, ang)
        if return_all:
            output.append([col, 'Coloured'])
        else:
            output = [[col, 'Coloured']]
    return output


# show cases a number of basic image manipulations
def showcase_basic(image):
    # original, identity, grey scale, sharpen, box blur
    images_to_plot = [[image, 'Original'], [convolve(image, identity()), 'Identity'],
                      [rgb_to_grey(image), 'Grey Scale'], [convolve(image, sharpen()), 'Sharpen'],
                      [convolve(image, box_blur(5, False)), 'Box Blur']]
    plot_images(images_to_plot, 1, 'Basic Image Manipulations')


# shows cases a number of edge detecting operators
def showcase_edge(image):
    # original
    images_to_plot = [[image, 'Original']]
    # sobel
    sx, sy = multi_convolve(rgb_to_grey(image), sobel(False), False)
    images_to_plot.append([sx, 'Sobel X'])
    images_to_plot.append([sy, 'Sobel Y'])
    sm = edge_magnitude(sx, sy)
    images_to_plot.append([sm, 'Sobel Gradient'])
    sa = edge_angle(sx, sy)
    images_to_plot.append([edge_colour(sm, sa), 'Sobel Coloured'])
    # laplacian
    images_to_plot.append([convolve(image, laplacian()[0]), 'Laplacian'])
    # grey scale
    images_to_plot.append([rgb_to_grey(image), 'Grey Scale'])
    # prewitt
    px, py = multi_convolve(rgb_to_grey(image), prewitt(False), False)
    images_to_plot.append([px, 'Prewitt X'])
    images_to_plot.append([py, 'Prewitt Y'])
    pm = edge_magnitude(px, py)
    images_to_plot.append([pm, 'Prewitt Gradient'])
    pa = edge_angle(px, py)
    images_to_plot.append([edge_colour(pm, pa), 'Prewitt Coloured'])
    # laplacian of gaussian
    images_to_plot.append([convolve(image, laplacian_of_gaussian(7)), 'Laplacian of Gaussian'])
    plot_images(images_to_plot, 2, 'Edge Detectors')


# showcases the steps involved with the canny process
def showcase_canny(image):
    # original
    images_to_plot = [[image, 'Original']]
    # canny process
    can = canny(image, True, True)
    for image_title in can:
        images_to_plot.append(image_title)
    plot_images(images_to_plot, 2, 'Canny Process')


# plots visualisation of hsv to rgb colour space
def hsv_colour_space(height=50):
    # -180 to 180 degrees
    angle = np.linspace(-180, 180, 361)
    hsv_gradient = np.ones((1, 361, 3))
    hsv_gradient[..., 0] = angle
    # converts hsv to rgb
    rgb_gradient = hsv_to_rgb(hsv_gradient)
    plt.imshow(rgb_gradient, extent=[-180, 180, 0, height])
    plt.title('HSV to RGB Colour Space')
    plt.xlabel('Angle ($\degree$)')
    plt.xticks(np.arange(-180, 181, 45))
    plt.yticks([])
    plt.show()


# plots given list of images on same figure
def plot_images(images, rows=1, main_title=None, size=(3, 2.5), show_axes=False, save=None):
    number_images = len(images)
    # determines plot shape from number of images and rows
    shape = [rows, int(np.ceil(number_images / rows))]
    while np.prod(shape) - shape[1] >= number_images:
        shape[0] -= 1
    # builds main plot
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(shape[1] * size[1], shape[0] * size[0]))
    # adds main plot title
    if main_title:
        fig.suptitle(main_title, fontsize=14)
    # flattens axes array
    if number_images > 1:
        axes = axes.ravel()
    else:
        axes = [axes]
    # loop through each sub plot
    for index, ax in enumerate(axes):
        try:  # attempt to get next image and title
            image, title = images[index]
            # ensure image values are in the range [0, 1]
            # print(title, image.min(), image.max())
            if np.min(image) < 0 or np.max(image) > 1:
                # clip values outside of range
                if title in ['Sharpen']:
                    image[image < 0] = 0
                    image[image > 1] = 1
                # normalises values into range
                else:
                    image = (image - np.min(image)) / (np.max(image) - np.min(image))
            # plots image with according colour map
            if image.ndim == 3:
                ax.imshow(image)
            elif image.ndim == 2:
                ax.imshow(image, cmap='gray')
            # plots title
            ax.set_title(title)
        except IndexError:
            pass
        # hide axes values
        if not show_axes:
            ax.set_axis_off()
    fig.tight_layout()
    if main_title:
        fig.subplots_adjust(top=0.84 + rows * 0.03)
    if save is not None:
        plt.savefig(f'{save}.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    image_path = 'Images/Convolutions/flower.png'
    input_image = mpimg.imread(image_path).astype('float')

    # hsv_colour_space()
    showcase_basic(image=input_image)
    showcase_edge(image=input_image)
    showcase_canny(image=input_image)
