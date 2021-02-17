import cv2
import numpy as np

from classify import CharClassifier

def preprocess_img(img):
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
    ret, img_label = cv2.connectedComponents(img)
    labels = np.unique(img_label)
    boxes = []
    for label in labels:
        y, x = np.where(img_label == label)  # row (y), col (x)
        # print(y, x)
        min_y = np.min(y)
        min_x = np.min(x)
        max_y = np.max(y)
        max_x = np.max(x)
        box_h_diff = np.abs(max_y - min_y)
        box_w_diff = np.abs(max_x - min_x)
        # print(min_y, min_x, max_x, max_y)
        # print(box_h_diff, box_w_diff)
        boxes.append([min_x, min_y, max_x, max_y])
        # if box_h_diff >= 30 and box_h_diff < height - 10 and box_w_diff >= 10 and box_w_diff < width - 10:
        #     boxes.append([min_x, min_y, max_x, max_y])
    
    return boxes

def filter_rows(bboxes, height, top_y1, top_y2, bottom_y1, bottom_y2):
    if len(bboxes) > 0:
        top_row = []
        bottom_row = []

        # Filter for row by heuristic of pixels
        # top_heuristic_y1, top_heuristic_y2 = 50 / height, 120 / height
        # bottom_heuristic_y1, bottom_heuristic_y2 = 225 / height, height / height
        # top_y1 = top_y1 / height
        # top_y2 = top_y2 / height
        # bottom_y1 = bottom_y1 / height
        # bottom_y2 = bottom_y2 / height
        for bbox in bboxes:
            y1, y2 = bbox[1] / height, bbox[3] / height
            if y1 > top_y1 and y1 < top_y2 and y2 > top_y1 and y2 < top_y2:
                top_row.append(bbox)
            if y1 > bottom_y1 and y1 < bottom_y2 and y2 > bottom_y1 and y2 < bottom_y2:
                bottom_row.append(bbox)
        top_row.sort(key=lambda x:x[0])
        bottom_row.sort(key=lambda x:x[0])

    return top_row, bottom_row


def main():
    # Load image
    # img = cv2.imread('divleague_crop.png')
    img = cv2.imread('divleague2_crop.png')
    img_bw = preprocess_img(img)

    height, width = img_bw.shape
    print(img_bw.shape)

    # Crop
    # y_1, y_2 = int(height / 2 - 1), int(height)
    # x_1, x_2 = 0, width
    # img = img[y_1:y_2, x_1:x_2]
    # print(img.shape)

    # Threshold for numbers.
    ret, img_thresh = cv2.threshold(img_bw, 210, 255, cv2.THRESH_BINARY)
    # ret, img_thresh = cv2.threshold(img_bw, 230, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # img_thresh2 = cv2.adaptiveThreshold(img_bw, 210, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # img_thresh = img_thresh + img_thresh2
    # ret, img_thresh = cv2.threshold(img_thresh, 10, 255, cv2.THRESH_BINARY)
    cv2.imwrite('test/thresh.jpg', img_thresh)

    # Erode and dilate to clean up.
    img_morph_clean = morph_clean(img_thresh)
    cv2.imwrite('test/morph_clean.jpg', img_morph_clean)

    # Organize pixels into shapes, and derive bounding boxes from them.
    bboxes = get_numbers_bbox(img_morph_clean)
    img_clean = img_morph_clean.copy()
    img_morph_clean = cv2.cvtColor(img_morph_clean, cv2.COLOR_GRAY2RGB)
    hist_x = np.zeros(width)
    hist_y = np.zeros(height)
    for bbox in bboxes:
        # img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), thickness=2)
        # img_morph_clean = cv2.rectangle(img_morph_clean, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), thickness=2)
        hist_x[bbox[0]: bbox[2]] += 1
        hist_y[bbox[1]: bbox[3]] += 1

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].set_title('X-axis pixel histogram')
    axs[0].set(xlabel='pixel coordinate', ylabel='count')
    axs[1].set_title('Y-axis pixel histogram')
    axs[1].set(xlabel='pixel coordinate', ylabel='count')
    axs[0].plot(np.arange(width), hist_x)
    axs[1].plot(np.arange(height), hist_y)
    plt.savefig('test/box_pixel_axis_hist1.png')

    # Group boxes by partitions by distance.
    print(hist_x.mean())
    print(hist_y.mean())

    x_mean = hist_x.mean()
    y_mean = hist_y.mean()
    x_bounds = []
    y_bounds = []

    # Values found in y-axis pixel histogram in MLBScreenOCR.ipynb.
    top_y1 = 0.1694915254237288
    top_y2 = 0.4067796610169492
    bottom_y1 = 0.7627118644067796
    bottom_y2 = 1.0
    # top_y1, top_y2 = 50, 120
    # bottom_y1, bottom_y2 = 225, height
    top_row, bottom_row = filter_rows(bboxes, height, top_y1, top_y2, bottom_y1, bottom_y2)
    # print(len(top_row), len(bottom_row))
    for bbox in top_row:
        # print(bbox)
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), thickness=2)
        img_morph_clean = cv2.rectangle(img_morph_clean, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), thickness=2)
    for bbox in bottom_row:
        # print(bbox)
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2)
        img_morph_clean = cv2.rectangle(img_morph_clean, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2)

    # boxes = np.array(bboxes)
    # for box in boxes:
    #     print(np.mean(box[:2]), np.mean(box[2:]), box)

    # Classify shape to number.
    char_classifier = CharClassifier()
    def decode(img, classifier, row):
        ret_row = []
        bbox = row[0]
        cur = [classifier(img[bbox[1]: bbox[3], bbox[0]: bbox[2]])]
        last = row[0][2]
        for bbox in row[1:]:
            char_class = classifier(img[bbox[1]: bbox[3], bbox[0]: bbox[2]])
            dist = bbox[0] - last
            print(dist, char_class)
            if dist < 40:
                cur.append(char_class)
            else:
                ret_row.append(str(''.join(cur)))
                cur = [char_class]
            last = bbox[2]
        ret_row.append(str(''.join(cur)))
        return ret_row

    stats = decode(img_clean, char_classifier, top_row)
    hitter = '-' in stats[0]
    print(stats, hitter)
    # print('TOP ROW:')
    # tmp_row = []
    # cur = []
    # last = 0
    # for bbox in top_row:
    #     char_class = char_classifier(img_clean[bbox[1]: bbox[3], bbox[0]: bbox[2]])
    #     dist = bbox[0] - last
    #     if not cur or dist < 90:
    #         cur.append(char_class)
    #     else:
    #         tmp_row.append(str(''.join(cur)))
    #         cur = [char_class]
    #     last = bbox[2]
    #     if char_class == '-':
    #         hitter = False
    #     print(char_class)
    # cur.append(char_class)
    # tmp_row.append(str(''.join(cur)))
    # print(tmp_row)

    bottom_row = decode(img_clean, char_classifier, bottom_row)
    stats.extend(bottom_row)
    print(bottom_row)
    # print('BOTTOM ROW:')
    # for bbox in bottom_row:
    #     print(char_classifier(img_clean[bbox[1]: bbox[3], bbox[0]: bbox[2]]))
    # print()

    # Decode numbers and return.
    # TODO: if 'dash', then pitcher, else hitter
    hitter_stat_names = ['AT-BATS', 'RUNS', 'HR', 'RBI',
                         'STEALS', 'AVERAGE', 'OBP', 'SLG',
                         'OPS', 'WAR']
    pitcher_stat_names = ['WIN-LOSS', 'SAVES', 'INNINGS', 'SO',
                          'WALKS', 'ERA', 'WHIP',
                          'K/9', 'BB/9', 'WAR']
    if hitter:
        stat_names = hitter_stat_names
    else:
        stat_names = pitcher_stat_names

    stats = {k: v for k, v in zip(stat_names, stats)}
    print(stats)

    cv2.imwrite('test/thresh_final.jpg', img_morph_clean)
    cv2.imwrite('test/out.jpg', img)

if __name__ == '__main__':
    main()
