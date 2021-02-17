# MLB20_OCR

# Purpose
This repo contains scripting and jupyter notebook showing work to develop a simple OCR for MLB2020 top players stats screen. It was developed to reduce manual efforts for the Divergent League Baseball project.

# Arguments
The 5 (1 + 4) arguments:
1) The path to input image.
2-4) Bounding box to crop for stats in format: [top_left_pt_x_coordinate, top_left_pt_y_coordinate, crop_region_width, crop_region_height]

# Usage Example
```
python3 extract_stats.py source_imgs/divleague_crop.png 0 0 1140 295
python3 extract_stats.py source_imgs/divleague2_crop.png 0 0 1141 297
```