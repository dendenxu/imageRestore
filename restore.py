import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, Perceptron, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from multiprocessing import Process, Array
from progressbar import ProgressBar


def is_salt_noise(img, threshold=0.1):
    is_zero = img == 0.
    if np.sum(is_zero) >= np.prod(img.shape) * threshold:
        return True
    else:
        return False


def restore_image(img, size=2):
    # return haar_denoise(img)
    if is_salt_noise(img):
        print("Found salty image")
        return restore_by_mean_multi_core(img)
        # return restore_by_mean(img, size=size)
        # return restore_by_linear_regression(img)
        # return restore_by_linear_regression_multi_core(img)
    else:
        print("Found gaussian noised image")
        return mean_global_restore(img, kernel_size=(size, size))


def gaussian_noise_image(img, var=0.1, mean=0):
    # adding mean gaussian noise
    noise = np.random.normal(mean, var, img.shape)
    noise_img = img + noise
    # noise_img += -np.min(noise_img)
    noise_img /= 1. if img.dtype == np.double else 255
    noise_img = np.clip(noise_img, 0., 1.)
    return noise_img


def mean_global_restore(img, kernel_size=(3, 3)):
    # we've defined three ways to reduce the noise on image, calling OpenCV library
    # you can uncomment the lines to remove gaussian noise
    # blur = cv2.medianBlur((img * 255 if img.dtype == np.double else 1).astype(np.uint8),
    #                       kernel_size[0]) / 255 if img.dtype == np.double else 1
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # blur = cv2.blur(img, kernel_size)
    return blur


def noise_mask_image(img, noise_ratio):
    return noise_mask_salt_pepper(img, noise_ratio)
    # return gaussian_noise_image(img, np.max(img) * noise_ratio)


def noise_mask_salt_pepper(img, noise_ratio):
    """
    generates the "noisy" image according to the specific problem

    :param img: img np.ndarray
    :param noise_ratio: more noise? 0.4/0.6/0.8
    :return: noise_img is the image with noise, 0-1 np.array, data type=np.double shape=(height,width,channel) channel=RGB
    """
    # copy the original image (different memory location)
    noise_img = np.copy(img)
    # initialization
    noise_mask = np.ones_like(noise_img, dtype='double')
    # mask image according to the ratio
    noise_mask[:, :round(noise_img.shape[1] * noise_ratio), :] = 0.
    # shuffle every row in every channel
    for channel in range(noise_img.shape[2]):
        for row in range(noise_img.shape[0]):
            np.random.shuffle(noise_mask[row, :, channel])
    noise_img = noise_img * noise_mask

    return noise_img


def get_noise_mask(noise_img):
    """
    get the noise mask of noise_img, usually a np.array

    :param noise_img: image with noise
    :return: noise mask, as double, size of noise_img
    """
    # we consider every 0 point in the image as noise point
    return np.array(noise_img != 0., dtype='double')


def in_range_one(row, rows, size):
    """
    generate range based on current position and size, taking care of edge

    :param row: current row/column number
    :param rows: total number of rows/columns
    :param size: radius/size of our consideration
    :return row_beg. row_end: normally row-size, row+size, however edge is taken care of
    """
    row_beg = row - size if row - size >= 0 else 0
    row_end = row + size if row + size < rows else rows - 1
    return row_beg, row_end


def in_range_two(row, col, rows, cols, size):
    """
    calls in_range_one twice to get 2D range for current position

    :param row: current row number
    :param col: current column number
    :param rows: total number of rows
    :param cols: total number of columns
    :param size: radius/size of our consideration
    :return: flattened 2D range
    """
    return np.array((in_range_one(row, rows, size), in_range_one(col, cols, size))).flatten()


def restore_by_mean(noise_img, size=2):
    """
    restore image by calculating RGB means of surrounding pixels.

    :param noise_img: a "noisy" image
    :param size: radius of mean values computation, defaults to 4, we compute mean from pixels in a 2*size*size square
    :return: res_img is the image restored, 0-1 np.array, data type=np.double shape=(height,width,channel) channel=RGB
    """
    # copy the original image (different memory location)
    res_img = np.copy(noise_img)
    # obtain noise_mask
    noise_mask = get_noise_mask(noise_img)
    # obtain shape of image
    rows, cols, channels = noise_img.shape
    # obtain noise points, as np.array "white"
    whites = np.argwhere(noise_mask == 0.)
    # use a progress bar to indicate progress
    with ProgressBar(max_value=len(whites)) as bar:
        for i, (row, col, channel) in enumerate(whites):
            res_img[row, col, channel] = mean(row, col, channel, rows, cols, size, noise_img, noise_mask)
            bar.update(i)
    return res_img


def restore_by_mean_multi_core(noise_img, size=2):
    """
    restore image by mean, however, we try to utilize multi-core CPU here

    :param noise_img: same as restore_by_mean
    :param size: same as restore_by_mean
    :return: same as restore_by_mean
    """
    noise_img_shared = Array("d", noise_img.ravel().tolist(), lock=False)
    # copy the original image (different memory location)
    res_img_shared = Array("d", np.copy(noise_img).ravel().tolist(), lock=False)
    # obtain noise_mask
    noise_mask_shared = Array("d", get_noise_mask(noise_img).ravel().tolist(), lock=False)
    # obtain shape of image
    rows, cols, channels = noise_img.shape
    # obtain noise points, as np.array "white"
    whites = np.argwhere(noise_img == 0.)
    # partition the whites list to 12 (number of logical cores of my CPU)
    parts = np.array_split(whites, 12)
    # contains all started jobs for future manipulation
    jobs = []
    # use a progress bar to indicate progress
    # this takes like, forever, why?
    with ProgressBar(max_value=len(parts)) as bar_start:
        for i, partial_whites in enumerate(parts):
            # we update the bar before spawning the subprocess to time it more accurately
            # and this makes the user see feedback faster, which makes them happy. At least it makes me happy
            bar_start.update(i)
            job = Process(
                target=wrapper_mean,
                args=(rows, cols, channels, size, noise_img_shared, noise_mask_shared, res_img_shared, partial_whites))
            jobs.append(job)
            job.start()
    # this takes like, forever
    with ProgressBar(max_value=len(jobs)) as bar_join:
        for i, job in enumerate(jobs):
            # same logic as above
            # if this is under job.join(), the first job takes 10s to finish,
            # the user waits 10s without seeing any feedback on screen
            bar_join.update(i)
            job.join()

    # notice that "res_img_shared" was originally a shared Array,
    # which takes some procedure to be transformed back to np.array
    return np.array(res_img_shared).reshape((rows, cols, channels))


def wrapper_mean(rows, cols, channels, size, noise_img, noise_mask, res_img, partial_whites):
    """
    A wrapper around the computation of mean value so that we can utilize the ability of multi-core CPU

    :param rows: img.shape[0]
    :param cols: img.shape[1]
    :param channels: img.shape[2]
    :param size: radius of mean values square
    :param noise_img: "noisy" img, shared memory between processes
    :param noise_mask: extracted noise mask, shared memory
    :param res_img: the result img to be modified, shared memory
    :param partial_whites: the white points that this wrapper should take care of
    :return: nothing, the function modifies res_img directly
    """

    # get objects we need from the shared memory, since client code requires them to have a certain shape
    noise_img = np.array(noise_img).reshape((rows, cols, channels))
    noise_mask = np.array(noise_mask).reshape((rows, cols, channels))

    # manually compute the corresponding indices, faster that iterating with np.unravel_index
    indices = partial_whites[:, 0] * cols * channels + partial_whites[:, 1] * channels + partial_whites[:, 2]

    # similar to what we do in a normal "restore by mean" function
    for i, (row, col, channel) in enumerate(partial_whites):
        # here the previously computed indices are used and we call function mean directly
        # notice that res_img is unchanged shared memory variable, which is mutable and affects the real memory
        # since different wrapper are to take care of different pixels, no lock is needed
        res_img[indices[i]] = mean(row, col, channel, rows, cols, size, noise_img, noise_mask)


def mean(row, col, channel, rows, cols, size, noise_img, noise_mask):
    """
    separate actual atomic computation from pre-processing, pave the way for multi-threading

    :param row: current row
    :param col: current column
    :param rows: total number of rows
    :param cols: total number of columns
    :param size: radius, 2*size*size image
    :param channel: current channel
    :param noise_img: "noisy" image
    :param noise_mask: binary(as double) noise mask
    :return: the mean value for [row, col, channel]
    """

    # we introduce a while(1) loop so that we can expand our search windows until one with valid pixel(s) is found
    while True:
        # considering the boundary, and transfer the square horizontally and vertically
        row_beg, row_end, col_beg, col_end = in_range_two(row, col, rows, cols, size)

        # mean values is sum(all pixels)/sum(noise mask)
        # since white point won't affect total sum and sum of noise mask indicates number of valid pixels
        # of course we can get number of valid positions from noise_img directly
        # but in practice, computing this from sum of noise_mask proves to be much faster
        number = np.sum(noise_mask[row_beg:row_end, col_beg:col_end, channel])
        # we update size and continue loop before computing "total", which saves time
        if number == 0.:
            size *= 2
            continue
        total = np.sum(noise_img[row_beg:row_end, col_beg:col_end, channel])
        return total / number


def restore_by_linear_regression(noise_img, size=2):
    """
    restore image by quadratic linear regression

    :param noise_img: same as restore_by_mean
    :param size: same as restore_by_mean
    :return: same as restore_by_mean
    """
    # copy the original image (different memory location)
    res_img = np.copy(noise_img)
    # obtain shape of image
    rows, cols, channels = noise_img.shape
    # obtain noise points, as np.array "white"
    whites = np.argwhere(noise_img == 0.)
    # use a progress bar to indicate progress
    with ProgressBar(max_value=len(whites)) as bar:
        for i, (row, col, channel) in enumerate(whites):
            bar.update(i)
            res_img[row, col, channel] = linear_regression(row, col, channel, rows, cols, size, noise_img)
    return np.clip(res_img, 0., 1.)


def linear_regression(row, col, channel, rows, cols, size, noise_img, use_quadratic=False):
    """
    separate actual atomic computation from pre-processing, pave the way for multi-threading

    :param row: current row
    :param col: current column
    :param rows: total number of rows
    :param cols: total number of columns
    :param size: radius, 2*size*size image
    :param channel: current channel
    :param noise_img: "noisy" image
    :param use_quadratic: whether to use quadratic linear regression (utilize CPU more)
    :return: the predicted value for [row, col, channel]
    """
    while True:
        # considering the boundary, and transfer the square horizontally and vertically
        row_beg, row_end, col_beg, col_end = in_range_two(row, col, rows, cols, size)
        # get the "noisy" local image, flattened for better vectorized operations
        noise_img_local = noise_img[row_beg:row_end, col_beg:col_end, channel].ravel()
        # get valid positions
        x_train = np.argwhere(noise_img_local != 0.)
        if len(x_train) == 0:
            size *= 2
            continue
        y_train = noise_img_local[x_train]
        if use_quadratic:
            # quadratic linear regression
            quadratic = PolynomialFeatures(degree=3)
            x_train_quadratic = quadratic.fit_transform(x_train)
            regress_quadratic = LinearRegression()
            regress_quadratic.fit(x_train_quadratic, y_train)
            # predict
            test = quadratic.transform([[2 * size * size + size]])
            return regress_quadratic.predict(test)
        else:
            test = [[2 * size * size + size]]
            lr = Ridge().fit(x_train, y_train)
            # lr = ElasticNet().fit(x_train, y_train) # not converging
            # lr = Lasso().fit(x_train, y_train) # not converging
            # lr = LinearRegression().fit(x_train, y_train)
            # lr = RidgeCV().fit(x_train, y_train)
            # lr = Perceptron().fit(x_train, y_train)
            return lr.predict(test)


def restore_by_linear_regression_multi_core(noise_img, size=2):
    """
    restore image by quadratic linear regression, however, we try to utilize multi-core CPU here

    :param noise_img: same as restore_by_mean
    :param size: same as restore_by_mean
    :return: same as restore_by_mean
    """
    noise_img_shared = Array("d", noise_img.ravel().tolist(), lock=False)
    # copy the original image (different memory location)
    res_img_shared = Array("d", np.copy(noise_img).ravel().tolist(), lock=False)
    # obtain shape of image
    rows, cols, channels = noise_img.shape
    # obtain noise points, as np.array "white"
    whites = np.argwhere(noise_img == 0.)
    # partition the whites list to 12 (number of logical cores of my CPU)
    parts = np.array_split(whites, 12)
    # contains all started jobs for future manipulation
    jobs = []
    # use a progress bar to indicate progress
    # this takes like, forever, why?
    with ProgressBar(max_value=len(parts)) as bar_start:
        for i, partial_whites in enumerate(parts):
            # we update the bar before spawning the subprocess to time it more accurately
            # and this makes the user see feedback faster, which makes them happy. At least it makes me happy
            bar_start.update(i)
            job = Process(
                target=wrapper_linear_regression,
                args=(rows, cols, channels, size, noise_img_shared, res_img_shared, partial_whites))
            jobs.append(job)
            job.start()
    # this takes like, forever
    with ProgressBar(max_value=len(jobs)) as bar_join:
        for i, job in enumerate(jobs):
            # same logic as above
            # if this is under job.join(), the first job takes 10s to finish,
            # the user waits 10s without seeing any feedback on screen
            bar_join.update(i)
            job.join()

    # notice that "res_img_shared" was originally a shared Array,
    # which takes some procedure to be transformed back to np.array
    return np.clip(np.array(res_img_shared).reshape((rows, cols, channels)), 0., 1.)


def wrapper_linear_regression(rows, cols, channels, size, noise_img, res_img, partial_whites):
    """
    A wrapper around the computation of linear regression function so that we can utilize the ability of multi-core CPU

    :param rows: img.shape[0]
    :param cols: img.shape[1]
    :param channels: img.shape[2]
    :param size: radius of mean values square
    :param noise_img: "noisy" img, shared memory between processes
    :param res_img: the result img to be modified, shared memory
    :param partial_whites: the white points that this wrapper should take care of
    :return: nothing, the function modifies res_img directly
    """
    noise_img = np.array(noise_img).reshape((rows, cols, channels))
    indices = partial_whites[:, 0] * cols * channels + partial_whites[:, 1] * channels + partial_whites[:, 2]
    for i, (row, col, channel) in enumerate(partial_whites):
        res_img[indices[i]] = linear_regression(row, col, channel, rows, cols, size, noise_img, use_quadratic=False)


def padding(img):
    """
    Pad the img so that it's divided by 2

    :param img: the img to be padded, as np.ndarray
    :return: the padded image
    """

    # make copies in case nothing is changed
    img_v = np.copy(img)
    img_h = np.copy(img)
    if img.shape[0] % 2:
        # pad a horizontal line
        img_v = np.ndarray(shape=(img.shape[0] + 1, img.shape[1], img.shape[2]))
        img_v[:-1, :, :] = img
        img_v[-1, :, :] = img[-1, :, :]
    if img_v.shape[1] % 2:
        # pad a vertical line
        img_h = np.ndarray(shape=(img_v.shape[0], img_v.shape[1] + 1, img_v.shape[2]))
        img_h[:, :-1, :] = img_v
        img_h[:, -1, :] = img_v[:, -1, :]
    return img_h


def haar_encode(img):
    """
    compute the haar transform of a img, padding it first, so you may want to store the original shape

    :param img: the img to be transformed
    :return: the transformed img
    """

    # pad the image for shape consistency
    img = padding(img)
    # compute haar info on axis=1
    img_v = np.zeros_like(img, dtype="double")
    img_v[:, :img.shape[1] // 2, :] = (img[:, ::2, :] + img[:, 1::2, :]) / 2
    img_v[:, img.shape[1] // 2:, :] = (img[:, ::2, :] - img[:, 1::2, :]) / 2
    # compute haar info on axis=0
    img_h = np.zeros_like(img, dtype="double")
    img_h[:img.shape[0] // 2, :, :] = (img_v[::2, :, :] + img_v[1::2, :, :]) / 2
    img_h[img.shape[0] // 2:, :, :] = (img_v[::2, :, :] - img_v[1::2, :, :]) / 2
    # the transformed image is returned
    return img_h


def haar_decode(img, padding_size=(0, 0)):
    """
    reverse the change caused by haar_transform, with padding information provided

    :param img: the haar transformed image
    :param padding_size: padding add to height and width to be reversed
    :return: the "untransformed" image
    """

    # reverse haar info on axis=0
    img_v = np.zeros_like(img, dtype="double")
    img_v[::2, :, :] = img[:img.shape[0] // 2, :, :] + img[img.shape[0] // 2:, :, :]
    img_v[1::2, :, :] = img[:img.shape[0] // 2, :, :] - img[img.shape[0] // 2:, :, :]
    # reverse haar info on axis=1
    img_h = np.zeros_like(img, dtype="double")
    img_h[:, ::2, :] = img_v[:, :img.shape[1] // 2, :] + img_v[:, img.shape[1] // 2:, :]
    img_h[:, 1::2, :] = img_v[:, :img.shape[1] // 2, :] - img_v[:, img.shape[1] // 2:, :]
    # restore height and width according to padding information
    # print(img_h.shape)
    # print(padding_size, padding_size.dtype)
    if padding_size[0] != 0:
        img_h = img_h[:-padding_size[0], :, :]
    if padding_size[1] != 0:
        img_h = img_h[:, :-padding_size[1], :]
    # print(img_h.shape)
    # return the restored image
    return img_h


def haar_denoise(noise_img, threshold=0.1):
    """
    calls haar_transform and haar_transform_back to remove noise in an img, deleting pixels under a certain threshold

    :param noise_img: "noisy" img
    :param threshold: we'd assume img to be of "double" and threshold should be in [0, 1]
    :return: 
    """

    # remember pading size
    padding_size = (np.array(noise_img.shape) % 2)[:-1]
    # print(padding_size)
    # do transform
    res_img = haar_encode(noise_img)

    # throw away small value
    shape = res_img.shape
    # print(shape)
    res_img = np.ravel(res_img)
    abs_res_img = abs(res_img)
    nearly_white = np.argwhere(abs_res_img < threshold)
    res_img[nearly_white] = 0
    res_img = res_img.reshape(shape)
    # transform back
    res_img = haar_decode(res_img, padding_size)

    # return the denoised img
    return np.clip(res_img, 0., 1.)

# plt.subplot(131)
# plt.imshow(img)
# plt.subplot(132)
# plt.imshow(noise_img)
# plt.subplot(133)
# plt.imshow(res_img)
