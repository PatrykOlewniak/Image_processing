import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def load_image(path, show=False, normalize=True):
    image = plt.imread('files/images/eiffel-tower.jpg').astype(float)
    if normalize:   # -> float (0;1)
        image = image / 255.
    if show:
        plt.imshow(image)
        plt.show()
    return image


def show_converted_img_with_source(source_image, changed_image):
    plt.subplot(1, 2, 1)
    plt.imshow(source_image, interpolation='none', cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.imshow(changed_image, interpolation='none', cmap=plt.cm.gray)
    plt.show()


def convolve_2d(image, kernel, normalize=True, RGB=True):

    if RGB:
        # convolve 2d the kernel with each channel
        r = scipy.signal.convolve2d(image[:, :, 0], kernel, mode='same')
        g = scipy.signal.convolve2d(image[:, :, 1], kernel, mode='same')
        b = scipy.signal.convolve2d(image[:, :, 2], kernel, mode='same')
    #
        # stack the channels back into a 8-bit colour depth image and plot it
        changed_image = np.dstack([r, g, b])
        if normalize:
            changed_image = (changed_image * 255).astype(np.uint8) # return to 256 (if normalized) TODO: check if normalized
    else:
        pass # TODO: implement grey_scale imgs
        
    return changed_image


def smooth(image, kernel_size=3):
    print(kernel_size)
    kernel = np.ones((kernel_size, kernel_size))
    kernel /= 1.0 * kernel_size * kernel_size
    return convolve_2d(image, kernel, RGB=True)



if __name__ == '__main__':
    eiffelImg = "files/images/eiffel-tower.jpg"
    img = load_image(eiffelImg)
    changed_image = smooth(img,3)
    show_converted_img_with_source(img, changed_image)
