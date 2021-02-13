from extract_stats import preprocess_img, morph_clean, get_numbers_bbox, filter_rows
import cv2

def main():
    img_filenames = ['divleague_crop.png', 'divleague2_crop.png']
    ex_i = 0
    for img_filename in img_filenames:
        print(img_filename)
        img = cv2.imread(img_filename)
        height, width, _ = img.shape
        img_bw = preprocess_img(img)
        ret, img_thresh = cv2.threshold(img_bw, 210, 255, cv2.THRESH_BINARY)
        img_morph_clean = morph_clean(img_thresh)
        bboxes = get_numbers_bbox(img_morph_clean)

        if len(bboxes) > 0:
            top_row = []
            bottom_row = []

            top_y1, top_y2 = 50, 120
            bottom_y1, bottom_y2 = 225, height
            top_row, bottom_row = filter_rows(bboxes, height, top_y1, top_y2, bottom_y1, bottom_y2)

            for bbox in top_row:
                cv2.imwrite('chars/{}.png'.format(ex_i), img_morph_clean[bbox[1]: bbox[3], bbox[0]: bbox[2]])
                ex_i += 1
            for bbox in bottom_row:
                cv2.imwrite('chars/{}.png'.format(ex_i), img_morph_clean[bbox[1]: bbox[3], bbox[0]: bbox[2]])
                ex_i += 1


if __name__ == '__main__':
    main()
