import os
from collections import namedtuple

# Where to save debug images and corner files
DEBUG_DIR = "calib_debug"

# Paths to images
CALIB_LEFT_GLOB  = "image_02/data/*.png"
CALIB_RIGHT_GLOB = "image_03/data/*.png"

# Provided ground-truth calibration file (for later comparison)
CALIB_FILE = "calib_cam_to_cam.txt"

# Camera IDs (strings just for labelling)
LEFT_ID  = "02"
RIGHT_ID = "03"

# 🔹 Single global square size (meters, or whatever unit you want)
SQUARE_SIZE = 9.950000e-02  # adjust to your real value

BoardDef = namedtuple("BoardDef", [
    "name",          # identifier, e.g. "B01"
    "pattern_size",  # (cols, rows) of INNER corners
    "roi_left",      # (x, y, w, h) in LEFT images
    "roi_right",     # (x, y, w, h) in RIGHT images
])

# Fill these with your measured ROIs.
# Values below are placeholders – adjust to your scene!
BOARDS = [
    BoardDef("B01", (7, 11),
             roi_left=(120, 120, 200, 200),
             roi_right=(70, 130, 210, 200)),

    BoardDef("B02", (7,  11),
             roi_left=(350, 170, 150, 150),
             roi_right=(240, 165, 220, 150)),

    BoardDef("B03", (7,  5),
             roi_left=(500, 270, 120, 120),
             roi_right=(420, 270, 130, 115)),

    BoardDef("B04", (7,  5),
             roi_left=(680, 300, 100, 120),
             roi_right=(600, 290, 100, 120)),

    BoardDef("B05", (7,  5),
             roi_left=(800, 300, 100, 120),
             roi_right=(710, 290, 100, 120)),

    BoardDef("B06", (7,  5),
             roi_left=(1100, 250, 100, 120),
             roi_right=(1000, 250, 100, 120)),

    BoardDef("B07", (7,  5),
             roi_left=(1320, 180, 70, 180),
             roi_right=(1200, 140, 100, 200)),

    BoardDef("B08", (5,  7),
             roi_left=(800, 120, 120, 100),
             roi_right=(700, 120, 130, 100)),

    BoardDef("B09", (5,  7),
             roi_left=(450, 70, 190, 80),
             roi_right=(350, 65, 200, 100)),

    BoardDef("B10", (5,  7),
             roi_left=(1030, 70, 190, 80),
             roi_right=(900, 65, 190, 80)),

    BoardDef("B11", (5,  7),
             roi_left=(500, 390, 190, 100),
             roi_right=(300, 385, 300, 100)),

    BoardDef("B12", (5,  7),
             roi_left=(1000, 380, 150, 100),
             roi_right=(885, 370, 150, 100)),

    BoardDef("B13", (15, 5),
             roi_left=(1200, 150, 120, 300),
             roi_right=(1115, 125, 95, 300)),
]

