import numpy as np

class ycb_video():
    def __init__(self):

        self.classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick', 'holiday_cup1', 'holiday_cup2', 'sanning_mug', \
                         '001_chips_can', 'block_red_big', 'block_green_big', 'block_blue_big', 'block_yellow_big', \
                         'block_red_small', 'block_green_small', 'block_blue_small', 'block_yellow_small', \
                         'block_red_median', 'block_green_median', 'block_blue_median', 'block_yellow_median',
                         'fusion_duplo_dude', 'cabinet_handle', 'industrial_dolly')
        self.num_classes = len(self.classes)

        self.class_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), \
                              (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (32, 0, 0), \
                              (150, 0, 0), (0, 150, 0), (0, 0, 150), (150, 150, 0), (75, 0, 0), (0, 75, 0), (0, 0, 75), (75, 75, 0), \
                              (200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0), (16, 16, 0), (16, 16, 16), (0, 180, 0)]

        self.nums = np.array([762, 1112, 1719, 2299, 2172, 1506, 1626, 2018, 2991, 1591, 1898, \
                     1107, 1104, 1800, 1619, 2305, 1335, 1716, 1424, 2218, 1873, 731, 1153, 1618, \
                     1401, 1444, 1607, 1448, 1558, 1164, 1124, 1952, 1254, 1567, 1554, 1668, \
                     2512, 2157, 3467, 3160, 2393, 2450, 2122, 2591, 2542, 2509, 2033, 2089, \
                     2244, 2402, 1917, 2009, 900, 837, 1929, 1830, 1226, 1872, 1720, 1864, \
                     754, 533, 680, 667, 668, 653, 801, 849, 884, 784, 1016, 951, 890, 719, 908, \
                     694, 864, 779, 689, 789, 788, 985, 743, 953, 986, 890, 897, 948, 453, 868, 842, 890]) - 1;

