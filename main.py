import os
import cv2
import numpy as np
from time import perf_counter
from multiprocessing import Process
from progressbar import ProgressBar
from PIL import Image
from scipy import spatial
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from restore import restore_by_mean, restore_by_mean_multi_core, restore_by_linear_regression, \
    restore_by_linear_regression_multi_core, \
    noise_mask_image, haar_denoise, \
    restore_image

SAMPLE_DIR = "samples"
OUTPUT_DIR = "output"
RESTORE_NAME_EXTENSION = "_res.png"
NOISE_NAME_EXTENSION = "_noi.png"
ORIGIN_NAME_EXTENSION = "_ori.png"
LOG_NAME_EXTENSION = "_output.txt"
PLOT_NAME_EXTENSION = "_plot.eps"


def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path)

    # 如果图片是三通道，采用 matplotlib 展示图像时需要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片
    plt.imshow(image)

    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()


def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)

    # 从给定数组的形状中删除一维的条目
    img = img.squeeze()

    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        print("Saving double array as uint8")
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max

        # 转换图片数组数据类型
        img = img.astype(np.uint8)

    # 将 RGB 方式转换为 BGR 方式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 生成图片
    cv2.imwrite(filename, img)


def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型 
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)

    # 图像数组数据放缩在 0-1 之间
    return image.astype(np.double) / info.max


def compute_error(res_img, img):
    """
    计算恢复图像 res_img 与原始图像 img 的 2-范数
    :param res_img:恢复图像 
    :param img:原始图像 
    :return: 恢复图像 res_img 与原始图像 img 的2-范数
    """
    # 初始化
    error = 0.0

    # 将图像矩阵转换成为np.narray
    res_img = np.array(res_img)
    img = np.array(img)

    # 如果2个图像的形状不一致，则打印出错误结果，返回值为 None
    if res_img.shape != img.shape:
        print("shape error res_img.shape and img.shape %s != %s" % (res_img.shape, img.shape))
        return None

    # 计算图像矩阵之间的评估误差
    error = np.sqrt(np.sum(np.power(res_img - img, 2)))

    return round(error, 3)


def calc_ssim(img, img_noise):
    """
    计算图片的结构相似度
    :param img: 原始图片， 数据类型为 ndarray, shape 为[长, 宽, 3]
    :param img_noise: 噪声图片或恢复后的图片，
                      数据类型为 ndarray, shape 为[长, 宽, 3]
    :return:
    """
    return ssim(img, img_noise,
                multichannel=True,
                data_range=img_noise.max() - img_noise.min())


def calc_csim(img, img_noise):
    """
    计算图片的 cos 相似度
    :param img: 原始图片， 数据类型为 ndarray, shape 为[长, 宽, 3]
    :param img_noise: 噪声图片或恢复后的图片，
                      数据类型为 ndarray, shape 为[长, 宽, 3]
    :return:
    """
    img = img.reshape(-1)
    img_noise = img_noise.reshape(-1)
    return 1 - spatial.distance.cosine(img, img_noise)


def read_img(path):
    img = Image.open(path)
    img = img.resize((150, 150))
    img = np.asarray(img, dtype="uint8")
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(img.dtype)
    # 图像数组数据放缩在 0-1 之间
    return img.astype(np.double) / info.max


def test_img(img_path):
    # 加载图片的路径和名称
    # img_path = 'eve.jpg'
    # img_path = 'A.png'

    # 读取原始图片
    img = read_image("/".join((SAMPLE_DIR, img_path)))

    # 展示原始图片
    # plot_image(image=img, image_title="original image")

    # 生成受损图片
    # 图像数据归一化
    nor_img = normalization(img)

    # 噪声比率
    noise_ratio = 0.6

    # 生成受损图片
    noise_img = noise_mask_image(nor_img, noise_ratio)

    # 展示受损图片
    # plot_image(image=noise_img, image_title="the noise_ratio = %s of original image" % noise_ratio)

    start_time = perf_counter()
    # 恢复图片
    res_img = restore_image(noise_img, size=2)
    # res_img = restore_by_mean(noise_img, size=2)
    # res_img = restore_by_mean_multi_core(noise_img, size=2)
    # res_img = restore_by_linear_regression(noise_img, size=2)
    # res_img = restore_by_linear_regression_multi_core(noise_img, size=2)
    # res_img = haar_denoise(noise_img)
    # res_img = cv2.fastNlMeansDenoisingColored((noise_img*255).astype("uint8")).astype("double") / 255
    end_time = perf_counter()

    # 计算恢复图片与原始图片的误差
    ori_img_path = "/".join((OUTPUT_DIR, img_path + ORIGIN_NAME_EXTENSION))
    noi_img_path = "/".join((OUTPUT_DIR, img_path + NOISE_NAME_EXTENSION))
    res_img_path = "/".join((OUTPUT_DIR, img_path + RESTORE_NAME_EXTENSION))
    res_log_path = "/".join((OUTPUT_DIR, img_path + LOG_NAME_EXTENSION))
    os.makedirs(os.path.dirname(res_img_path), exist_ok=True)
    with open(res_log_path, "w") as f:
        log_info = "\n".join((
            "Name of Image: " + img_path,
            "Noise Ratio: {:.3f}".format(noise_ratio),
            "Time to restore: {:.3f}s".format(end_time - start_time),
            "SSIM Similarity: {:.3f}".format(calc_ssim(res_img, nor_img)),
            "Cosine Similarity: {:.3f}".format(calc_csim(res_img, nor_img)),
            "Error Restore/Origin: {:.3f}".format(compute_error(res_img, nor_img)),
            "Error Noise/Origin: {:.3f}".format(compute_error(noise_img, nor_img)),
        ))
        f.write(log_info)

    # 展示恢复图片
    # plot_image(image=res_img, image_title="restore image")

    # 保存恢复图片
    save_image(res_img_path, res_img)
    save_image(noi_img_path, noise_img)
    save_image(ori_img_path, nor_img)
    plot_img(nor_img, noise_img, res_img, log_info, img_path)


def test_all(img_path):
    ori = read_image(img_path)
    nor_img = normalization(ori)
    noise_ratio = 0.6
    noise_img = noise_mask_image(nor_img, noise_ratio)
    times = [perf_counter()]
    res_img_mean = restore_by_mean(noise_img, size=2)
    # res_img_mean = haar_denoise(noise_img)
    times += [perf_counter()]
    res_img_mean_multi = restore_by_mean_multi_core(noise_img, size=2)
    # res_img_mean_multi = haar_denoise(noise_img)
    times += [perf_counter()]
    res_img_lr = restore_by_linear_regression(noise_img, size=2)
    # res_img_lr = haar_denoise(noise_img)
    times += [perf_counter()]
    res_img_lr_multi = restore_by_linear_regression_multi_core(noise_img, size=2)
    # res_img_lr_multi = haar_denoise(noise_img)
    times += [perf_counter()]
    res_img_haar = haar_denoise(noise_img)
    times += [perf_counter()]
    res_img_cv = cv2.fastNlMeansDenoisingColored((noise_img * 255).astype("uint8")).astype("double") / 255
    times += [perf_counter()]
    log_info = "\n".join((
        "Name of Image: " + img_path,
        "Noise Ratio: {:.3f}".format(noise_ratio),
        "SSIM Noise/Origin: {:.3f}".format(calc_ssim(noise_img, nor_img)),
        "Cosine Noise/Origin: {:.3f}".format(calc_csim(noise_img, nor_img)),
        "Error Noise/Origin: {:.3f}".format(compute_error(noise_img, nor_img)),
        "Error Mean/Origin: {:.3f}".format(compute_error(res_img_mean, nor_img)),
        "Error Mean Multi/Origin: {:.3f}".format(compute_error(res_img_mean_multi, nor_img)),
        "Error LR/Origin: {:.3f}".format(compute_error(res_img_lr, nor_img)),
        "Error LR Multi/Origin: {:.3f}".format(compute_error(res_img_lr_multi, nor_img)),
        "Error Haar/Origin: {:.3f}".format(compute_error(res_img_haar, nor_img)),
        "Error OpenCV/Origin: {:.3f}".format(compute_error(res_img_cv, nor_img)),
    ))
    times = np.array(times)
    times = times[1::] - times[0:-1]
    hspace = 0.5
    wspace = 0.5
    width = 10
    height = ori.shape[0] / ori.shape[1] * width * hspace / wspace
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    axi_ori = fig.add_subplot(331)
    axi_noi = fig.add_subplot(332)
    axi_log = fig.add_subplot(333)
    axi_res_mean = fig.add_subplot(334)
    axi_res_mean_multi = fig.add_subplot(335)
    axi_res_lr = fig.add_subplot(336)
    axi_res_lr_multi = fig.add_subplot(337)
    axi_res_haar = fig.add_subplot(338)
    axi_res_cv = fig.add_subplot(339)

    axi_ori.set_title("Original Image")
    axi_ori.imshow(ori)
    axi_noi.set_title("Noisy Image")
    axi_noi.imshow(noise_img)

    axi_res_mean.set_title("Mean")
    axi_res_mean.imshow(res_img_mean)
    axi_res_mean.set_xlabel("{:.3f}s".format(times[0]))
    axi_res_mean_multi.set_title("Mean Multi")
    axi_res_mean_multi.imshow(res_img_mean_multi)
    axi_res_mean_multi.set_xlabel("{:.3f}s".format(times[1]))
    axi_res_lr.set_title("Linear Regression")
    axi_res_lr.imshow(res_img_lr)
    axi_res_lr.set_xlabel("{:.3f}s".format(times[2]))
    axi_res_lr_multi.set_title("Linear Regression Multi")
    axi_res_lr_multi.imshow(res_img_lr_multi)
    axi_res_lr_multi.set_xlabel("{:.3f}s".format(times[3]))
    axi_res_haar.set_title("Haar")
    axi_res_haar.imshow(res_img_haar)
    axi_res_haar.set_xlabel("{:.3f}s".format(times[4]))
    axi_res_cv.set_title("OpenCV")
    axi_res_cv.imshow(res_img_cv)
    axi_res_cv.set_xlabel("{:.3f}s".format(times[5]))

    axi_log.set_xlim(0, ori.shape[1])
    axi_log.set_ylim(0, ori.shape[0])
    axi_log.text(0, ori.shape[0] // 2, log_info, family="monospace", style="italic", ha="left", va="center")
    axi_log.axis("off")
    fig.savefig("/".join((OUTPUT_DIR, img_path + PLOT_NAME_EXTENSION)))


def main():
    img_list = os.listdir(SAMPLE_DIR)
    jobs = []
    with ProgressBar(max_value=len(img_list)) as bar_start:
        for i, img_path in enumerate(img_list):
            bar_start.update(i)
            job = Process(target=test_img, args=(img_path,))
            job.start()
            jobs.append(job)
    with ProgressBar(max_value=len(jobs)) as bar_join:
        for i, job in enumerate(jobs):
            bar_join.update(i)
            job.join()


def plot_img(ori, noi, res, log, img_path):
    width = 10
    height = ori.shape[0] / ori.shape[1] / 4 * width
    fig = plt.figure(figsize=(width, height))
    axi_ori = fig.add_subplot(141)
    axi_noi = fig.add_subplot(142)
    axi_res = fig.add_subplot(143)
    axi_log = fig.add_subplot(144)
    axi_ori.set_title("Original Image")
    axi_ori.imshow(ori)
    axi_noi.set_title("Noisy Image")
    axi_noi.imshow(noi)
    axi_res.set_title("Restored Image")
    axi_res.imshow(res)
    axi_log.set_xlim(0, ori.shape[1])
    axi_log.set_ylim(0, ori.shape[0])
    axi_log.text(0, ori.shape[0] // 2, log, family="monospace", style="italic", ha="left", va="center")
    axi_log.axis("off")
    fig.savefig("/".join((OUTPUT_DIR, img_path + PLOT_NAME_EXTENSION)))


def plot_dir(dirname):
    files = os.listdir(dirname)
    files.sort()
    txts = [file_name for file_name in files if file_name.endswith(LOG_NAME_EXTENSION)]
    ress = [file_name for file_name in files if file_name.endswith(RESTORE_NAME_EXTENSION)]
    oris = [file_name for file_name in files if file_name.endswith(ORIGIN_NAME_EXTENSION)]
    nois = [file_name for file_name in files if file_name.endswith(NOISE_NAME_EXTENSION)]
    lst = np.transpose(np.array((txts, oris, nois, ress)))

    for txt, ori, noi, res in lst:
        with open(dirname + txt, "r") as f:
            txt = f.read()
        img_path = ori
        ori = read_image(ori)
        noi = read_image(noi)
        res = read_image(res)
        plot_img(ori, noi, res, txt, img_path)
