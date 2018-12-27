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

def smooth(image, kernel_size=3):
    # smooth kernel  - small smooth
    kernel = np.ones((kernel_size, kernel_size))
    kernel /= 1.0 * kernel_size * kernel_size



def convolve_2d(image, kernel_size=3):
    # smooth kernel  - small smooth
    kernel = np.ones((kernel_size, kernel_size))
    kernel /= 1.0 * kernel_size * kernel_size

    # convolve 2d the kernel with each channel
    r = scipy.signal.convolve2d(image[:, :, 0], kernel, mode='same')
    g = scipy.signal.convolve2d(image[:, :, 1], kernel, mode='same')
    b = scipy.signal.convolve2d(image[:, :, 2], kernel, mode='same')
#
    # stack the channels back into a 8-bit colour depth image and plot it
    image_changed = np.dstack([r, g, b])
    image_changed = (image_changed * 255).astype(np.uint8) # return to 256 (if normalized) TODO: check if normalized

    plt.subplot(1, 2, 1)
    plt.imshow(image, interpolation='none', cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.imshow(image_changed, interpolation='none', cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    eiffelImg = "files/images/eiffel-tower.jpg"
    img = load_image(eiffelImg)
    convolve_2d(img)
