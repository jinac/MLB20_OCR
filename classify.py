import numpy as np


class CharClassifier(object):
    def __init__(self, feat_mat_filename='digit_feat_map2.npy'):
        self.feat_mat_filename = feat_mat_filename
        self.feat_mat_classes, self.feat_mat = self.load_feat_mat(feat_mat_filename)
        self.h_div = 10
        self.w_div = 3
        self.thresh_region = 0.6

    def load_feat_mat(self, feat_mat_filename):
        data = np.load(feat_mat_filename)
        return data['feat_mat_classes'], data['feat_mat']

    def get_features(self, img):
        if np.max(img) == 255.:
            img = img / 255.
        h, w = img.shape
        h_win = int(h / self.h_div)
        w_win = int(w / self.w_div)
        div = (h_win * w_win)

        feat_ratio = h / w
        feat_regions = []
        for y in range(self.h_div):
            for x in range(self.w_div):
                win = img[y * h_win: (y+1) * h_win, x * w_win: (x+1) * w_win]
                occ_ratio = np.sum(win)
                occ_ratio /= div
                if occ_ratio >= self.thresh_region:
                    feat_regions.append(1.)
                else:
                    feat_regions.append(-1.)
        return feat_regions

    def classify(self, img):
        h, w = img.shape
        ratio = h / w
        if ratio < 0.5:
            return '-'
        if ratio >= 0.5 and ratio < 1.3:
            return '.'
        if ratio >= 1.3 and ratio < 2.7:
            digit_features = self.get_features(img)
            # print(len(digit_features))
            # print(np.dot(self.feat_mat, digit_features))
            idx = np.argmax(np.dot(self.feat_mat, digit_features))
            # print(idx)
            return str(self.feat_mat_classes[idx])
        if ratio >= 2.7:
            return '1'

    def __call__(self, img):
        return self.classify(img)
