"""
MLB2020 OCR stats extraction script.
"""
import argparse

import cv2
import numpy as np

from classify import CharClassifier


TOP_Y1 = 0.1694915254237288
TOP_Y2 = 0.4067796610169492
BOTTOM_Y1 = 0.7627118644067796
BOTTOM_Y2 = 1.0

BIN_THRESH = 210

PX_DIST = 0.03508771929824561

HITTER_STATS = ['AT-BATS', 'RUNS', 'HR',
                'RBI', 'STEALS', 'AVERAGE',
                'OBP', 'SLG', 'OPS', 'WAR']
PITCHER_STATS = ['WIN-LOSS', 'SAVES', 'INNINGS',
                 'SO', 'WALKS', 'ERA', 'WHIP',
                 'K/9', 'BB/9', 'WAR']

def preprocess_img(img):
    """
    Preprocess image by converting to grayscale single color channel
    and applying gaussian blur for later threshold step.
    """
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bw = cv2.medianBlur(img_bw, 5)
    return img_bw

def morph_clean(img, kernel_shape=(3, 3)):
    """
    Clean image with morophological erosion and dilation.
    """
    kernel = np.ones(kernel_shape, np.uint8)
    img_erode = cv2.erode(img, kernel, iterations=1)
    img_morph_clean = cv2.dilate(img_erode, kernel, iterations=1)
    return img_morph_clean

def get_numbers_bbox(img):
    """
    Get bounding boxes around connected components in thresholded image.
    """
    ret, img_label = cv2.connectedComponents(img)
    labels = np.unique(img_label)
    boxes = []
    for label in labels:
        y, x = np.where(img_label == label)  # row (y), col (x)
        min_y = np.min(y)
        min_x = np.min(x)
        max_y = np.max(y)
        max_x = np.max(x)
        box_h_diff = np.abs(max_y - min_y)
        box_w_diff = np.abs(max_x - min_x)
        boxes.append([min_x, min_y, max_x, max_y])
    return boxes

def filter_rows(bboxes, height):
    """
    Filter out found thresholded items by row heuristics
    to return only the top row and bottom row bounding boxes.
    """
    if len(bboxes) > 0:
        top_row = []
        bottom_row = []

        # Filter for row by heuristic of pixels
        for bbox in bboxes:
            y1, y2 = bbox[1] / height, bbox[3] / height
            if y1 > TOP_Y1 and y1 < TOP_Y2 and y2 > TOP_Y1 and y2 < TOP_Y2:
                top_row.append(bbox)
            if y1 > BOTTOM_Y1 and y1 < BOTTOM_Y2 and y2 > BOTTOM_Y1 and y2 < BOTTOM_Y2:
                bottom_row.append(bbox)
        top_row.sort(key=lambda x:x[0])
        bottom_row.sort(key=lambda x:x[0])

    return top_row, bottom_row

def decode(img, classifier, row):
    """
    Use classifier to interpret characters in bounding box
    and separate characters into sections by horizontal pixel
    distance heuristic.
    """
    ret_row = []
    _, w = img.shape
    bbox = row[0]
    cur = [classifier(img[bbox[1]: bbox[3], bbox[0]: bbox[2]])]
    last = row[0][2]
    for bbox in row[1:]:
        char_class = classifier(img[bbox[1]: bbox[3], bbox[0]: bbox[2]])
        dist = (bbox[0] - last) / w
        if dist < PX_DIST:
            cur.append(char_class)
        else:
            ret_row.append(str(''.join(cur)))
            cur = [char_class]
        last = bbox[2]
    ret_row.append(str(''.join(cur)))
    return ret_row


def extract_stats(img):
    # Preprocess image.
    img_bw = preprocess_img(img)
    height, width = img_bw.shape

    # Threshold for numbers.
    ret, img_thresh = cv2.threshold(img_bw, BIN_THRESH, 255, cv2.THRESH_BINARY)

    # Erode and dilate to clean up.
    img_morph_clean = morph_clean(img_thresh)

    # Organize pixels into shapes, and derive bounding boxes from them.
    bboxes = get_numbers_bbox(img_morph_clean)
    img_clean = img_morph_clean.copy()
    img_morph_clean = cv2.cvtColor(img_morph_clean, cv2.COLOR_GRAY2RGB)

    # Values found in y-axis pixel histogram in MLBScreenOCR.ipynb.
    top_row, bottom_row = filter_rows(bboxes, height)

    # Classify shape to number and group by distance to neighboring horizontal box.
    char_classifier = CharClassifier()
    stats = decode(img_clean, char_classifier, top_row)
    bottom_row = decode(img_clean, char_classifier, bottom_row)
    stats.extend(bottom_row)

    # Decode numbers and return.
    stats_names = PITCHER_STATS if '-' in stats[0] else HITTER_STATS 
    stats = {k: v for k, v in zip(stats_names, stats)}
    return stats

def main():
    # Load arguments.
    parser = argparse.ArgumentParser(
        description='Script to extract stats from MLB2020 top players screen')
    parser.add_argument('img_filename', help='path of image to extract stats from.')
    parser.add_argument('crop_box', nargs=4, type=int, help='bounding box to crop to stats region of image')
    args = parser.parse_args()

    img_filename = args.img_filename
    crop_box = args.crop_box

    # Load image.
    img = cv2.imread(img_filename)

    # Crop according to crop box.
    x1, x2 = crop_box[0], crop_box[0] + crop_box[2]
    y1, y2 = crop_box[1], crop_box[1] + crop_box[3]
    img = img[y1:y2, x1:x2]

    stats = extract_stats(img)
    print(stats)

if __name__ == '__main__':
    main()
