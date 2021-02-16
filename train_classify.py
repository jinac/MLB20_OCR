import glob
import os
import numpy as np

import cv2


h_reg = 10
w_reg = 3
CLASS_DICT = {
    'dash': 10,
    'dot': 11,
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
}

def get_features(img, bbox, img_h, img_w):
    # img = cv2.resize(img, (w_reg, h_reg), interpolation=cv2.INTER_AREA)
    img = img / 255.
    # print(img)
    h = bbox[3] - bbox[1]
    w = bbox[2] - bbox[0]

    h_win = int(img_h / h_reg)
    w_win = int(img_w / w_reg)
    div = (h_win * w_win)
    # print(div, h_win, w_win)
    thresh_region = 0.6

    feat_ratio = h / w
    # img[img >= thresh_region] = 1.
    # feat_regions = img.flatten()
    # print(feat_regions)
    feat_regions = []
    for y in range(h_reg):
        for x in range(w_reg):
            win = img[y * h_win: (y+1) * h_win, x * w_win: (x+1) * w_win]
            occ_ratio = np.sum(win)
            # print(occ_ratio, div)
            occ_ratio /= div
            # print(occ_ratio)
            # feat_regions.append(occ_ratio)
            if occ_ratio >= thresh_region:
                feat_regions.append(1.)
            else:
                feat_regions.append(-1.)

    # print(np.array(feat_regions).reshape(h_reg, w_reg))
    return [feat_ratio, *feat_regions]

def classify(features, feat_mat):
    ratio = features[0]
    if ratio < 0.5:
        return CLASS_DICT['dash']
    if ratio > 0.5 and ratio < 1.3:
        return CLASS_DICT['dot']
    if ratio > 1.3 and ratio < 2.7:
        f = features[1:]
        # print(f)
        x = np.argmax(np.dot(feat_mat, features[1:]))
        # print(x)
        if x > 0:
            x += 1
        return x
        # return 'a number'
    if ratio > 2.7:
        return CLASS_DICT['1']

def main():
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_files = glob.glob(os.path.join(train_dir, '*.png'))
    test_files = glob.glob(os.path.join('chars', '*/*.png'))
    char_classes = [filename.split('/')[-1].split('.')[0] for filename in train_files]

    feat_mat = []
    feat_ratio = []
    for char_class, train_file in zip(char_classes, train_files):
        img = cv2.imread(train_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        h, w = img.shape
        bbox = [0, 0, w, h]
        feat = get_features(img, bbox, h, w)
        # print(char_class, feat[0])
        f = np.asarray(feat[1:])
        f = f.reshape(h_reg, w_reg)
        # print(f)
        if char_class not in ['dash', 'dot', '1']:
            feat_mat.append(feat[1:])
        feat_ratio.append(feat[0])
        # print(train_file, h, w)

    feat_mat = np.asarray(feat_mat)
    print(feat_mat)
    with open('digit_feat_map.npy', 'wb') as f:
        np.save(f, feat_mat)
    for i, char_class in enumerate(['0', '2', '3', '4', '5', '6', '7', '8', '9']):
        # print(feat_mat[i, :])
        print(char_class, feat_ratio[i], np.argmax(np.dot(feat_mat, feat_mat[i, :])))

    print(test_files)
    for test_file in test_files:
        img = cv2.imread(test_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        bbox = [0, 0, w, h]
        feat = get_features(img, bbox, h, w)
        # print(test_file, feat[0], np.argmax(np.dot(feat_mat, feat[1:])))
        print(test_file, classify(feat, feat_mat))
        # print(test_file, h, w)

if __name__ == '__main__':
    main()
