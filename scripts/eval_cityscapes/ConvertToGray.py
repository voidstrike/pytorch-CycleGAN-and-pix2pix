from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

object_to_gray = {
    'road': 0,
    'sidewalk': 1,
    'building': 2,
    'fence': 4,
    'pole': 5,
    'traffic-light': 6,
    'traffic-sign': 7,
    'vegetation': 8,
    'terrain': 9,
    'sky': 10,
    'person':11,
    'rider': 12,
    'car': 13,
    'truck': 14,
    'bus': 15,
    'train': 16,
    'motorbicycle':17,
    'bicycle': 18,
    'default': 255
}

object_to_rgb = {
    'road':(128, 64, 128),
    'sidewalk': (244, 35, 232),
    'building': (70, 70, 70),
    'fence': (190, 153, 153),
    'pole': (153, 153, 153),
    'traffic-light': (250, 170, 30),
    'traffic-sign': (220, 220, 0),
    'vegetation': (107, 142, 35),
    'terrain': (152, 251, 152),
    'sky': (70, 130, 180),
    'person': (220, 20, 60),
    'rider': (255, 0, 0),
    'car': (0, 0, 142),
    'truck': (0, 0, 70),
    'bus': (0, 60, 100),
    'train': (0, 80, 180),
    'motorbicycle': (0, 0, 230),
    'bicycle': (119, 11, 32),
    'default': (0, 0, 0)
}

def MahattanDistance(rgb_p, rgb_o):
    return abs(rgb_p[0] - rgb_o[0]) + abs(rgb_p[1] - rgb_o[1]) + abs(rgb_p[2] - rgb_o[2])

def AdjustImage(im):
    w, h = im.size
    s = int((w + h - abs(w - h)) / 2)
    l = int((w - s) / 2)
    u = int((h - s) / 2)
    im_new = im.crop((l, u, l+s, u+s))
    return im_new.resize((256, 256))

def ReturnGrayImage(path):
    img = Image.open(path).resize((256, 256))

    # img = AdjustImage(img)
    gray_matrix = np.zeros((256, 256), dtype=np.int64)
    # plt.imshow(img)
    # plt.show()

    # test_image = np.zeros((256, 256, 3))
    # print(img.getpixel((0, 0)))

    for i in range(256):
        for j in range(256):
            min_dis = 999
            min_obj = None
            for k in object_to_rgb.keys():
                temp_dis = MahattanDistance(img.getpixel((j, i)), object_to_rgb[k])
                if min_dis >= temp_dis:
                    min_dis = temp_dis
                    min_obj = k
            gray_matrix[i][j] = object_to_gray[min_obj]
            # test_image[j][i] = img.getpixel((i, j))
    # plt.imshow(gray_matrix)
    # plt.show()
    return gray_matrix

def SaveGrayImage(path, gray):
    plt.imshow(gray)
    plt.savefig(path)

def main():
    SaveGrayImage('gray.jpg',ReturnGrayImage('roadimage.jpg'))

if __name__ == '__main__':
    main()