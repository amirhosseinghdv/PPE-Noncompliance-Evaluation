from enum import IntFlag
import os
import cv2
import tqdm
import time

from keras.preprocessing import image
from matplotlib.pyplot import imshow
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import numpy 
import keras
import tensorflow
import tensorflow as tf
from tensorflow import keras


from config import opt
from models.yolox import Detector
from utils.util import mkdir, label_color, get_img_path


aval_aval = time.time()



human_yolo_GOOD_dict = {'0.19':0, '0.2':0}
human_yolo_UNCERTAIN_dict = {'0.19':0, '0.2':0}
human_yolo_BAD_dict = {'0.19':1, '0.2':1}

hat_yolo_GOOD_dict = {'0.19':0, '0.2':0}
hat_yolo_UNCERTAIN_dict = {'0.19':0, '0.2':0}
hat_yolo_BAD_dict = {'0.19':1, '0.2':1}

NOhat_yolo_GOOD_dict = {'0.19':0, '0.2':0}
NOhat_yolo_UNCERTAIN_dict = {'0.19':0, '0.2':0}
NOhat_yolo_BAD_dict = {'0.19':1, '0.2':1}

vest_yolo_GOOD_dict = {'0.19':0, '0.2':0}
vest_yolo_UNCERTAIN_dict = {'0.19':0, '0.2':0}
vest_yolo_BAD_dict = {'0.19':1, '0.2':1}

NOvest_yolo_GOOD_dict = {'0.19':0, '0.2':0}
NOvest_yolo_UNCERTAIN_dict = {'0.19':0, '0.2':0}
NOvest_yolo_BAD_dict = {'0.19':1, '0.2':1}

gloves_yolo_GOOD_dict = {'0.19':0, '0.2':0}
gloves_yolo_UNCERTAIN_dict = {'0.19':0, '0.2':0}
gloves_yolo_BAD_dict = {'0.19':1, '0.2':1}

NOgloves_yolo_GOOD_dict = {'0.19':0, '0.2':0}
NOgloves_yolo_UNCERTAIN_dict = {'0.19':0, '0.2':0}
NOgloves_yolo_BAD_dict = {'0.19':1, '0.2':1}


for x in range(21, 71):
    human_yolo_GOOD_dict[str(x/100)] = 2 * (x/100 - 0.2)
for x in range(71, 101):
    human_yolo_GOOD_dict[str(x/100)] = 1

for x in range(21, 71):
    human_yolo_BAD_dict[str(x/100)] = (-2 * (x/100 - 0.2)) + 1
for x in range(71, 101):
    human_yolo_BAD_dict[str(x/100)] = 0

for x in range(21, 46):
    human_yolo_UNCERTAIN_dict[str(x/100)] = 4 * (x/100 - 0.2)
for x in range(45, 71):
    human_yolo_UNCERTAIN_dict[str(x/100)] = (-4 * (x/100 - 0.45)) + 1
for x in range(71, 101):
    human_yolo_UNCERTAIN_dict[str(x/100)] = 0
for x in range(21, 71):
    hat_yolo_GOOD_dict[str(x/100)] = 1.8 * (x/100 - 0.2) + 0.1
for x in range(71, 101):
    hat_yolo_GOOD_dict[str(x/100)] = 1

for x in range(21, 71):
    hat_yolo_BAD_dict[str(x/100)] = (-1.8 * (x/100 - 0.2)) + 0.9
for x in range(71, 101):
    hat_yolo_BAD_dict[str(x/100)] = 0

for x in range(21, 46):
    hat_yolo_UNCERTAIN_dict[str(x/100)] = 3.4 * (x/100 - 0.2) + 0.1
for x in range(45, 71):
    hat_yolo_UNCERTAIN_dict[str(x/100)] = (-3.8 * (x/100 - 0.45)) + 0.95
for x in range(71, 101):
    hat_yolo_UNCERTAIN_dict[str(x/100)] = 0
for x in range(21, 81):
    NOhat_yolo_GOOD_dict[str(x/100)] = 8/6 * (x/100 - 0.2) + 0.2
for x in range(81, 101):
    NOhat_yolo_GOOD_dict[str(x/100)] = 1

for x in range(21, 81):
    NOhat_yolo_BAD_dict[str(x/100)] = (-8/6 * (x/100 - 0.2)) + 0.8
for x in range(81, 101):
    NOhat_yolo_BAD_dict[str(x/100)] = 0

for x in range(21, 51):
    NOhat_yolo_UNCERTAIN_dict[str(x/100)] = 7/3 * (x/100 - 0.2) + 0.2
for x in range(50, 81):
    NOhat_yolo_UNCERTAIN_dict[str(x/100)] = (-3 * (x/100 - 0.5)) + 0.9
for x in range(81, 101):
    NOhat_yolo_UNCERTAIN_dict[str(x/100)] = 0
for x in range(21, 71):
    vest_yolo_GOOD_dict[str(x/100)] = 1.4 * (x/100 - 0.2) + 0.3
for x in range(71, 101):
    vest_yolo_GOOD_dict[str(x/100)] = 1

for x in range(21, 71):
    vest_yolo_BAD_dict[str(x/100)] = (-1.4 * (x/100 - 0.2)) + 0.7
for x in range(71, 101):
    vest_yolo_BAD_dict[str(x/100)] = 0

for x in range(21, 46):
    vest_yolo_UNCERTAIN_dict[str(x/100)] = 2.2 * (x/100 - 0.2) + 0.3
for x in range(45, 71):
    vest_yolo_UNCERTAIN_dict[str(x/100)] = (-3.4 * (x/100 - 0.45)) + 0.85
for x in range(71, 101):
    vest_yolo_UNCERTAIN_dict[str(x/100)] = 0
for x in range(21, 71):
    NOvest_yolo_GOOD_dict[str(x/100)] = 1.8 * (x/100 - 0.2) + 0.1
for x in range(71, 101):
    NOvest_yolo_GOOD_dict[str(x/100)] = 1

for x in range(21, 71):
    NOvest_yolo_BAD_dict[str(x/100)] = (-1.8 * (x/100 - 0.2)) + 0.9
for x in range(71, 101):
    NOvest_yolo_BAD_dict[str(x/100)] = 0

for x in range(21, 46):
    NOvest_yolo_UNCERTAIN_dict[str(x/100)] = 3.4 * (x/100 - 0.2) + 0.1
for x in range(45, 71):
    NOvest_yolo_UNCERTAIN_dict[str(x/100)] = (-3.8 * (x/100 - 0.45)) + 0.95
for x in range(71, 101):
    NOvest_yolo_UNCERTAIN_dict[str(x/100)] = 0
for x in range(21, 101):
    gloves_yolo_GOOD_dict[str(x/100)] = 1 * (x/100 - 0.2) + 0.2

for x in range(21, 101):
    gloves_yolo_BAD_dict[str(x/100)] = (-1 * (x/100 - 0.2)) + 0.8


for x in range(21, 61):
    gloves_yolo_UNCERTAIN_dict[str(x/100)] = 1.75 * (x/100 - 0.2) + 0.2
for x in range(60, 101):
    gloves_yolo_UNCERTAIN_dict[str(x/100)] = (-2.25 * (x/100 - 0.6)) + 0.9
for x in range(21, 61):
    NOgloves_yolo_GOOD_dict[str(x/100)] = 2 * (x/100 - 0.2) + 0.2
for x in range(61, 101):
    NOgloves_yolo_GOOD_dict[str(x/100)] = 1

for x in range(21, 61):
    NOgloves_yolo_BAD_dict[str(x/100)] = (-2 * (x/100 - 0.2)) + 0.8
for x in range(61, 101):
    NOgloves_yolo_BAD_dict[str(x/100)] = 0

for x in range(21, 41):
    NOgloves_yolo_UNCERTAIN_dict[str(x/100)] = 3.5 * (x/100 - 0.2) + 0.2
for x in range(40, 61):
    NOgloves_yolo_UNCERTAIN_dict[str(x/100)] = (-4.5 * (x/100 - 0.4)) + 0.9
for x in range(61, 101):
    NOgloves_yolo_UNCERTAIN_dict[str(x/100)] = 0






hat_efficientnet_GOOD_dict = {'0':0}
hat_efficientnet_UNCERTAIN_dict = {'0':0}
hat_efficientnet_BAD_dict = {'0':1}

NOhat_efficientnet_GOOD_dict = {'0':0}
NOhat_efficientnet_UNCERTAIN_dict = {'0':0}
NOhat_efficientnet_BAD_dict = {'0':1}

vest_efficientnet_GOOD_dict = {'0':0}
vest_efficientnet_UNCERTAIN_dict = {'0':0}
vest_efficientnet_BAD_dict = {'0':1}

NOvest_efficientnet_GOOD_dict = {'0':0}
NOvest_efficientnet_UNCERTAIN_dict = {'0':0}
NOvest_efficientnet_BAD_dict = {'0':1}

gloves_efficientnet_GOOD_dict = {'0':0}
gloves_efficientnet_UNCERTAIN_dict = {'0':0}
gloves_efficientnet_BAD_dict = {'0':1}

NOgloves_efficientnet_GOOD_dict = {'0':0}
NOgloves_efficientnet_UNCERTAIN_dict = {'0':0}
NOgloves_efficientnet_BAD_dict = {'0':1}

for x in range(1, 101):
    hat_efficientnet_GOOD_dict[str(x/100)] = 0.7 * (x/100 - 0.01) + 0.3

for x in range(1, 101):
    hat_efficientnet_BAD_dict[str(x/100)] = (-0.7 * (x/100 - 0.01)) + 0.7


for x in range(1, 51):
    hat_efficientnet_UNCERTAIN_dict[str(x/100)] = 1.1 * (x/100 - 0.01) + 0.3
for x in range(51, 101):
    hat_efficientnet_UNCERTAIN_dict[str(x/100)] = (-1.7 * (x/100 - 0.5)) + 0.85
for x in range(1, 101):
    NOhat_efficientnet_GOOD_dict[str(x/100)] = 0.7 * (x/100 - 0.01) + 0.3

for x in range(1, 101):
    NOhat_efficientnet_BAD_dict[str(x/100)] = (-0.7 * (x/100 - 0.01)) + 0.7


for x in range(1, 51):
    NOhat_efficientnet_UNCERTAIN_dict[str(x/100)] = 1.1 * (x/100 - 0.01) + 0.3
for x in range(51, 101):
    NOhat_efficientnet_UNCERTAIN_dict[str(x/100)] = (-1.7 * (x/100 - 0.5)) + 0.85
for x in range(1, 101):
    vest_efficientnet_GOOD_dict[str(x/100)] = 0.8 * (x/100 - 0.01) + 0.2

for x in range(1, 101):
    vest_efficientnet_BAD_dict[str(x/100)] = (-0.8 * (x/100 - 0.01)) + 0.8


for x in range(1, 51):
    vest_efficientnet_UNCERTAIN_dict[str(x/100)] = 1.4 * (x/100 - 0.01) + 0.2
for x in range(51, 101):
    vest_efficientnet_UNCERTAIN_dict[str(x/100)] = (-1.8 * (x/100 - 0.5)) + 0.9
for x in range(1, 101):
    NOvest_efficientnet_GOOD_dict[str(x/100)] = 0.8 * (x/100 - 0.01) + 0.2

for x in range(1, 101):
    NOvest_efficientnet_BAD_dict[str(x/100)] = (-0.8 * (x/100 - 0.01)) + 0.8


for x in range(1, 51):
    NOvest_efficientnet_UNCERTAIN_dict[str(x/100)] = 1.4 * (x/100 - 0.01) + 0.2
for x in range(51, 101):
    NOvest_efficientnet_UNCERTAIN_dict[str(x/100)] = (-1.8 * (x/100 - 0.5)) + 0.9
for x in range(1, 101):
    gloves_efficientnet_GOOD_dict[str(x/100)] = 0.35 * (x/100 - 0.01) + 0.25

for x in range(1, 101):
    gloves_efficientnet_BAD_dict[str(x/100)] = (-0.35 * (x/100 - 0.01)) + 0.75

for x in range(1, 51):
    gloves_efficientnet_UNCERTAIN_dict[str(x/100)] = 0.85 * (x/100 - 0.01) + 0.25
for x in range(51, 101):
    gloves_efficientnet_UNCERTAIN_dict[str(x/100)] = (-0.55 * (x/100 - 0.5)) + 0.675
for x in range(1, 101):
    NOgloves_efficientnet_GOOD_dict[str(x/100)] = 0.35 * (x/100 - 0.01) + 0.25

for x in range(1, 101):
    NOgloves_efficientnet_BAD_dict[str(x/100)] = (-0.35 * (x/100 - 0.01)) + 0.75

for x in range(1, 51):
    NOgloves_efficientnet_UNCERTAIN_dict[str(x/100)] = 0.85 * (x/100 - 0.01) + 0.25
for x in range(51, 101):
    NOgloves_efficientnet_UNCERTAIN_dict[str(x/100)] = (-0.55 * (x/100 - 0.5)) + 0.675










human_FUZZY_GOOD_dict = {}
human_FUZZY_UNCERTAIN_dict = {}
human_FUZZY_BAD_dict = {}

NOhat_FUZZY_GOOD_dict = {}
NOhat_FUZZY_UNCERTAIN_dict = {}
NOhat_FUZZY_BAD_dict = {}

NOvest_FUZZY_GOOD_dict = {}
NOvest_FUZZY_UNCERTAIN_dict = {}
NOvest_FUZZY_BAD_dict = {}

NOgloves_FUZZY_GOOD_dict = {}
NOgloves_FUZZY_UNCERTAIN_dict = {}
NOgloves_FUZZY_BAD_dict = {}

for x in range(0, 51):
    human_FUZZY_GOOD_dict[str(x/100)] = 0
for x in range(51, 76):
    human_FUZZY_GOOD_dict[str(x/100)] =  (4 * (x/100 - 0.5)) 
for x in range(75, 101):
    human_FUZZY_GOOD_dict[str(x/100)] = 1

for x in range(0, 26):
    human_FUZZY_BAD_dict[str(x/100)] = 1
for x in range(25, 51):
    human_FUZZY_BAD_dict[str(x/100)] =  (-4 * (x/100 - 0.25)) +1
for x in range(50, 101):
    human_FUZZY_BAD_dict[str(x/100)] = 0

for x in range(0, 26):
    human_FUZZY_UNCERTAIN_dict[str(x/100)] = 0
for x in range(25, 51):
    human_FUZZY_UNCERTAIN_dict[str(x/100)] = (4 * (x/100 - 0.25)) 
for x in range(50, 76):
    human_FUZZY_UNCERTAIN_dict[str(x/100)] = (-4 * (x/100 - 0.5)) +1
for x in range(75, 101):
    human_FUZZY_UNCERTAIN_dict[str(x/100)] = 0
for x in range(0, 51):
    NOhat_FUZZY_GOOD_dict[str(x/100)] = 0
for x in range(51, 76):
    NOhat_FUZZY_GOOD_dict[str(x/100)] =  (4 * (x/100 - 0.5)) 
for x in range(75, 101):
    NOhat_FUZZY_GOOD_dict[str(x/100)] = 1

for x in range(0, 26):
    NOhat_FUZZY_BAD_dict[str(x/100)] = 1
for x in range(25, 51):
    NOhat_FUZZY_BAD_dict[str(x/100)] =  (-4 * (x/100 - 0.25)) +1
for x in range(50, 101):
    NOhat_FUZZY_BAD_dict[str(x/100)] = 0

for x in range(0, 26):
    NOhat_FUZZY_UNCERTAIN_dict[str(x/100)] = 0
for x in range(25, 51):
    NOhat_FUZZY_UNCERTAIN_dict[str(x/100)] = (4 * (x/100 - 0.25)) 
for x in range(50, 76):
    NOhat_FUZZY_UNCERTAIN_dict[str(x/100)] = (-4 * (x/100 - 0.5)) +1
for x in range(75, 101):
    NOhat_FUZZY_UNCERTAIN_dict[str(x/100)] = 0
for x in range(0, 51):
    NOvest_FUZZY_GOOD_dict[str(x/100)] = 0
for x in range(51, 76):
    NOvest_FUZZY_GOOD_dict[str(x/100)] =  (4 * (x/100 - 0.5)) 
for x in range(75, 101):
    NOvest_FUZZY_GOOD_dict[str(x/100)] = 1

for x in range(0, 26):
    NOvest_FUZZY_BAD_dict[str(x/100)] = 1
for x in range(25, 51):
    NOvest_FUZZY_BAD_dict[str(x/100)] =  (-4 * (x/100 - 0.25)) +1
for x in range(50, 101):
    NOvest_FUZZY_BAD_dict[str(x/100)] = 0

for x in range(0, 26):
    NOvest_FUZZY_UNCERTAIN_dict[str(x/100)] = 0
for x in range(25, 51):
    NOvest_FUZZY_UNCERTAIN_dict[str(x/100)] = (4 * (x/100 - 0.25)) 
for x in range(50, 76):
    NOvest_FUZZY_UNCERTAIN_dict[str(x/100)] = (-4 * (x/100 - 0.5)) +1
for x in range(75, 101):
    NOvest_FUZZY_UNCERTAIN_dict[str(x/100)] = 0
for x in range(0, 51):
    NOgloves_FUZZY_GOOD_dict[str(x/100)] = 0
for x in range(51, 76):
    NOgloves_FUZZY_GOOD_dict[str(x/100)] =  (4 * (x/100 - 0.5)) 
for x in range(75, 101):
    NOgloves_FUZZY_GOOD_dict[str(x/100)] = 1

for x in range(0, 26):
    NOgloves_FUZZY_BAD_dict[str(x/100)] = 1
for x in range(25, 51):
    NOgloves_FUZZY_BAD_dict[str(x/100)] =  (-4 * (x/100 - 0.25)) +1
for x in range(50, 101):
    NOgloves_FUZZY_BAD_dict[str(x/100)] = 0

for x in range(0, 26):
    NOgloves_FUZZY_UNCERTAIN_dict[str(x/100)] = 0
for x in range(25, 51):
    NOgloves_FUZZY_UNCERTAIN_dict[str(x/100)] = (4 * (x/100 - 0.25)) 
for x in range(50, 76):
    NOgloves_FUZZY_UNCERTAIN_dict[str(x/100)] = (-4 * (x/100 - 0.5)) +1
for x in range(75, 101):
    NOgloves_FUZZY_UNCERTAIN_dict[str(x/100)] = 0

result_FUZZY_VerySafe_dict = {}
result_FUZZY_safe_dict = {}
result_FUZZY_MediumDanger_dict = {}
result_FUZZY_Dangerous_dict = {}
result_FUZZY_VeryDangerous_dict = {}

for x in range(0, 16):
    result_FUZZY_VerySafe_dict[str(x/100)] = 1
for x in range(15, 31):
    result_FUZZY_VerySafe_dict[str(x/100)] =  (-1/0.15 * (x/100 - 0.15)) +1
for x in range(30, 101):
    result_FUZZY_VerySafe_dict[str(x/100)] = 0

for x in range(0, 16):
    result_FUZZY_safe_dict[str(x/100)] = 0
for x in range(15, 31):
    result_FUZZY_safe_dict[str(x/100)] = (1/0.15 * (x/100 - 0.15)) 
for x in range(30, 46):
    result_FUZZY_safe_dict[str(x/100)] = (-1/0.15 * (x/100 - 0.3)) +1
for x in range(45, 101):
    result_FUZZY_safe_dict[str(x/100)] = 0

for x in range(0, 36):
    result_FUZZY_MediumDanger_dict[str(x/100)] = 0
for x in range(35, 51):
    result_FUZZY_MediumDanger_dict[str(x/100)] = (1/0.15 * (x/100 - 0.35)) 
for x in range(50, 66):
    result_FUZZY_MediumDanger_dict[str(x/100)] = (-1/0.15 * (x/100 - 0.5)) +1
for x in range(65, 101):
    result_FUZZY_MediumDanger_dict[str(x/100)] = 0

for x in range(0, 56):
    result_FUZZY_Dangerous_dict[str(x/100)] = 0
for x in range(55, 71):
    result_FUZZY_Dangerous_dict[str(x/100)] = (1/0.15 * (x/100 - 0.55)) 
for x in range(70, 86):
    result_FUZZY_Dangerous_dict[str(x/100)] = (-1/0.15 * (x/100 - 0.7)) +1
for x in range(85, 101):
    result_FUZZY_Dangerous_dict[str(x/100)] = 0

for x in range(0, 71):
    result_FUZZY_VeryDangerous_dict[str(x/100)] = 0
for x in range(70, 86):
    result_FUZZY_VeryDangerous_dict[str(x/100)] =  (1/0.15 * (x/100 - 0.7)) 
for x in range(85, 101):
    result_FUZZY_VeryDangerous_dict[str(x/100)] = 1







def rule_human1(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo):
    output = min( max(human_yolo_BAD_dict[human_yolo] , human_yolo_UNCERTAIN_dict[human_yolo]) ,  max(hat_yolo_GOOD_dict[hat_yolo], NOhat_yolo_GOOD_dict[NOhat_yolo], vest_yolo_GOOD_dict[vest_yolo], NOvest_yolo_GOOD_dict[NOvest_yolo], gloves_yolo_GOOD_dict[gloves_yolo], NOgloves_yolo_GOOD_dict[NOgloves_yolo]))
    return output
def rule_human2(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo):
    output = min( max(human_yolo_BAD_dict[human_yolo] , human_yolo_UNCERTAIN_dict[human_yolo]) , min(1 - hat_yolo_GOOD_dict[hat_yolo], 1 - NOhat_yolo_GOOD_dict[NOhat_yolo], 1 - vest_yolo_GOOD_dict[vest_yolo], 1 - NOvest_yolo_GOOD_dict[NOvest_yolo], 1 - gloves_yolo_GOOD_dict[gloves_yolo], 1 - NOgloves_yolo_GOOD_dict[NOgloves_yolo]) , max(hat_yolo_UNCERTAIN_dict[hat_yolo], NOhat_yolo_UNCERTAIN_dict[NOhat_yolo], vest_yolo_UNCERTAIN_dict[vest_yolo], NOvest_yolo_UNCERTAIN_dict[NOvest_yolo], gloves_yolo_UNCERTAIN_dict[gloves_yolo], NOgloves_yolo_UNCERTAIN_dict[NOgloves_yolo]))
    return output
def rule_human3(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo):
    output = min( max(human_yolo_BAD_dict[human_yolo] , human_yolo_UNCERTAIN_dict[human_yolo]) ,  min(hat_yolo_BAD_dict[hat_yolo], NOhat_yolo_BAD_dict[NOhat_yolo], vest_yolo_BAD_dict[vest_yolo], NOvest_yolo_BAD_dict[NOvest_yolo], gloves_yolo_BAD_dict[gloves_yolo], NOgloves_yolo_BAD_dict[NOgloves_yolo]))
    return output
def rule_human4(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo):
    output = min( human_yolo_GOOD_dict[human_yolo] ,  min(hat_yolo_BAD_dict[hat_yolo], NOhat_yolo_BAD_dict[NOhat_yolo], vest_yolo_BAD_dict[vest_yolo], NOvest_yolo_BAD_dict[NOvest_yolo], gloves_yolo_BAD_dict[gloves_yolo], NOgloves_yolo_BAD_dict[NOgloves_yolo]))
    return output
def rule_human5(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo):
    output = min( human_yolo_GOOD_dict[human_yolo] ,  max(hat_yolo_GOOD_dict[hat_yolo], NOhat_yolo_GOOD_dict[NOhat_yolo], vest_yolo_GOOD_dict[vest_yolo], NOvest_yolo_GOOD_dict[NOvest_yolo], gloves_yolo_GOOD_dict[gloves_yolo], NOgloves_yolo_GOOD_dict[NOgloves_yolo], hat_yolo_UNCERTAIN_dict[hat_yolo], NOhat_yolo_UNCERTAIN_dict[NOhat_yolo], vest_yolo_UNCERTAIN_dict[vest_yolo], NOvest_yolo_UNCERTAIN_dict[NOvest_yolo], gloves_yolo_UNCERTAIN_dict[gloves_yolo], NOgloves_yolo_UNCERTAIN_dict[NOgloves_yolo]))
    return output
def rule_hat1(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( max(hat_efficientnet_UNCERTAIN_dict[hat_efficientnet] , hat_efficientnet_GOOD_dict[hat_efficientnet]) , max(hat_yolo_UNCERTAIN_dict[hat_yolo] , hat_yolo_GOOD_dict[hat_yolo]) , max(NOhat_yolo_BAD_dict[NOhat_yolo] , NOhat_yolo_UNCERTAIN_dict[NOhat_yolo]) )                 
    return output
def rule_hat2(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( hat_efficientnet_UNCERTAIN_dict[hat_efficientnet]  , hat_yolo_GOOD_dict[hat_yolo] , NOhat_yolo_GOOD_dict[NOhat_yolo] )                 
    return output
def rule_hat3(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( max(hat_efficientnet_UNCERTAIN_dict[hat_efficientnet] , hat_efficientnet_GOOD_dict[hat_efficientnet]) , max(hat_yolo_BAD_dict[hat_yolo] , hat_yolo_UNCERTAIN_dict[hat_yolo]) , NOhat_yolo_GOOD_dict[NOhat_yolo] )                 
    return output
def rule_hat4(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( max(hat_efficientnet_UNCERTAIN_dict[hat_efficientnet] , hat_efficientnet_GOOD_dict[hat_efficientnet]) , hat_yolo_BAD_dict[hat_yolo] , NOhat_yolo_BAD_dict[NOhat_yolo] )                 
    return output
def rule_hat5(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( max(hat_efficientnet_UNCERTAIN_dict[hat_efficientnet] , hat_efficientnet_GOOD_dict[hat_efficientnet]) , hat_yolo_BAD_dict[hat_yolo] , NOhat_yolo_UNCERTAIN_dict[NOhat_yolo] )                 
    return output
def rule_hat6(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( hat_efficientnet_GOOD_dict[hat_efficientnet]  , hat_yolo_GOOD_dict[hat_yolo] , NOhat_yolo_GOOD_dict[NOhat_yolo] )                 
    return output
def rule_hat7(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( min(hat_efficientnet_BAD_dict[hat_efficientnet] , NOhat_efficientnet_BAD_dict[NOhat_efficientnet]) , hat_yolo_GOOD_dict[hat_yolo] , NOhat_yolo_GOOD_dict[NOhat_yolo] )                 
    return output
def rule_hat8(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( min(hat_efficientnet_BAD_dict[hat_efficientnet] , NOhat_efficientnet_BAD_dict[NOhat_efficientnet]) , hat_yolo_GOOD_dict[hat_yolo] , NOhat_yolo_BAD_dict[NOhat_yolo] )                 
    return output
def rule_hat9(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( min(hat_efficientnet_BAD_dict[hat_efficientnet] , NOhat_efficientnet_BAD_dict[NOhat_efficientnet]) , hat_yolo_UNCERTAIN_dict[hat_yolo] , max(NOhat_yolo_BAD_dict[NOhat_yolo], NOhat_yolo_UNCERTAIN_dict[NOhat_yolo], NOhat_yolo_GOOD_dict[NOhat_yolo]) ) 
    return output
def rule_hat10(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( min(hat_efficientnet_BAD_dict[hat_efficientnet] , NOhat_efficientnet_BAD_dict[NOhat_efficientnet]) , max(hat_yolo_BAD_dict[hat_yolo] , hat_yolo_GOOD_dict[hat_yolo]) , NOhat_yolo_UNCERTAIN_dict[NOhat_yolo] )                 
    return output
def rule_hat11(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( min(hat_efficientnet_BAD_dict[hat_efficientnet] , NOhat_efficientnet_BAD_dict[NOhat_efficientnet]) , hat_yolo_BAD_dict[hat_yolo] , NOhat_yolo_GOOD_dict[NOhat_yolo] )                 
    return output
def rule_hat12(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( min(hat_efficientnet_BAD_dict[hat_efficientnet] , NOhat_efficientnet_BAD_dict[NOhat_efficientnet]) , hat_yolo_BAD_dict[hat_yolo] , NOhat_yolo_BAD_dict[NOhat_yolo] )                 
    return output
def rule_hat13(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( max(NOhat_efficientnet_UNCERTAIN_dict[NOhat_efficientnet] , NOhat_efficientnet_GOOD_dict[NOhat_efficientnet]) , max(NOhat_yolo_UNCERTAIN_dict[NOhat_yolo] , NOhat_yolo_GOOD_dict[NOhat_yolo]) , max(hat_yolo_BAD_dict[hat_yolo] , hat_yolo_UNCERTAIN_dict[hat_yolo]) )                 
    return output
def rule_hat14(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( NOhat_efficientnet_UNCERTAIN_dict[NOhat_efficientnet]  , hat_yolo_GOOD_dict[hat_yolo] , NOhat_yolo_GOOD_dict[NOhat_yolo] )                 
    return output
def rule_hat15(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( max(NOhat_efficientnet_UNCERTAIN_dict[NOhat_efficientnet] , NOhat_efficientnet_GOOD_dict[NOhat_efficientnet]) , max(NOhat_yolo_BAD_dict[NOhat_yolo] , NOhat_yolo_UNCERTAIN_dict[NOhat_yolo]) , hat_yolo_GOOD_dict[hat_yolo] )                 
    return output
def rule_hat16(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( max(NOhat_efficientnet_UNCERTAIN_dict[NOhat_efficientnet] , NOhat_efficientnet_GOOD_dict[NOhat_efficientnet]) , hat_yolo_BAD_dict[hat_yolo] , NOhat_yolo_BAD_dict[NOhat_yolo] )                 
    return output
def rule_hat17(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( max(NOhat_efficientnet_UNCERTAIN_dict[NOhat_efficientnet] , NOhat_efficientnet_GOOD_dict[NOhat_efficientnet]) , NOhat_yolo_BAD_dict[NOhat_yolo] , hat_yolo_UNCERTAIN_dict[hat_yolo] )                 
    return output
def rule_hat18(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet):
    output = min( NOhat_efficientnet_GOOD_dict[NOhat_efficientnet]  , hat_yolo_GOOD_dict[hat_yolo] , NOhat_yolo_GOOD_dict[NOhat_yolo] )                 
    return output
def rule_vest1(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( max(vest_efficientnet_UNCERTAIN_dict[vest_efficientnet] , vest_efficientnet_GOOD_dict[vest_efficientnet]) , max(vest_yolo_UNCERTAIN_dict[vest_yolo] , vest_yolo_GOOD_dict[vest_yolo]) , max(NOvest_yolo_BAD_dict[NOvest_yolo] , NOvest_yolo_UNCERTAIN_dict[NOvest_yolo]) )                 
    return output
def rule_vest2(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( vest_efficientnet_UNCERTAIN_dict[vest_efficientnet]  , vest_yolo_GOOD_dict[vest_yolo] , NOvest_yolo_GOOD_dict[NOvest_yolo] )                 
    return output
def rule_vest3(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( max(vest_efficientnet_UNCERTAIN_dict[vest_efficientnet] , vest_efficientnet_GOOD_dict[vest_efficientnet]) , max(vest_yolo_BAD_dict[vest_yolo] , vest_yolo_UNCERTAIN_dict[vest_yolo]) , NOvest_yolo_GOOD_dict[NOvest_yolo] )                 
    return output
def rule_vest4(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( max(vest_efficientnet_UNCERTAIN_dict[vest_efficientnet] , vest_efficientnet_GOOD_dict[vest_efficientnet]) , vest_yolo_BAD_dict[vest_yolo] , NOvest_yolo_BAD_dict[NOvest_yolo] )                 
    return output
def rule_vest5(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( max(vest_efficientnet_UNCERTAIN_dict[vest_efficientnet] , vest_efficientnet_GOOD_dict[vest_efficientnet]) , vest_yolo_BAD_dict[vest_yolo] , NOvest_yolo_UNCERTAIN_dict[NOvest_yolo] )                 
    return output
def rule_vest6(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( vest_efficientnet_GOOD_dict[vest_efficientnet]  , vest_yolo_GOOD_dict[vest_yolo] , NOvest_yolo_GOOD_dict[NOvest_yolo] )                 
    return output
def rule_vest7(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( min(vest_efficientnet_BAD_dict[vest_efficientnet] , NOvest_efficientnet_BAD_dict[NOvest_efficientnet]) , vest_yolo_GOOD_dict[vest_yolo] , NOvest_yolo_GOOD_dict[NOvest_yolo] )                 
    return output
def rule_vest8(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( min(vest_efficientnet_BAD_dict[vest_efficientnet] , NOvest_efficientnet_BAD_dict[NOvest_efficientnet]) , vest_yolo_GOOD_dict[vest_yolo] , NOvest_yolo_BAD_dict[NOvest_yolo] )                 
    return output
def rule_vest9(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( min(vest_efficientnet_BAD_dict[vest_efficientnet] , NOvest_efficientnet_BAD_dict[NOvest_efficientnet]) , vest_yolo_UNCERTAIN_dict[vest_yolo] , max(NOvest_yolo_BAD_dict[NOvest_yolo], NOvest_yolo_UNCERTAIN_dict[NOvest_yolo], NOvest_yolo_GOOD_dict[NOvest_yolo]) ) 
    return output
def rule_vest10(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( min(vest_efficientnet_BAD_dict[vest_efficientnet] , NOvest_efficientnet_BAD_dict[NOvest_efficientnet]) , max(vest_yolo_BAD_dict[vest_yolo] , vest_yolo_GOOD_dict[vest_yolo]) , NOvest_yolo_UNCERTAIN_dict[NOvest_yolo] )                 
    return output
def rule_vest11(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( min(vest_efficientnet_BAD_dict[vest_efficientnet] , NOvest_efficientnet_BAD_dict[NOvest_efficientnet]) , vest_yolo_BAD_dict[vest_yolo] , NOvest_yolo_GOOD_dict[NOvest_yolo] )                 
    return output
def rule_vest12(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( min(vest_efficientnet_BAD_dict[vest_efficientnet] , NOvest_efficientnet_BAD_dict[NOvest_efficientnet]) , vest_yolo_BAD_dict[vest_yolo] , NOvest_yolo_BAD_dict[NOvest_yolo] )                 
    return output
def rule_vest13(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( max(NOvest_efficientnet_UNCERTAIN_dict[NOvest_efficientnet] , NOvest_efficientnet_GOOD_dict[NOvest_efficientnet]) , max(NOvest_yolo_UNCERTAIN_dict[NOvest_yolo] , NOvest_yolo_GOOD_dict[NOvest_yolo]) , max(vest_yolo_BAD_dict[vest_yolo] , vest_yolo_UNCERTAIN_dict[vest_yolo]) )                 
    return output
def rule_vest14(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( NOvest_efficientnet_UNCERTAIN_dict[NOvest_efficientnet]  , vest_yolo_GOOD_dict[vest_yolo] , NOvest_yolo_GOOD_dict[NOvest_yolo] )                 
    return output
def rule_vest15(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( max(NOvest_efficientnet_UNCERTAIN_dict[NOvest_efficientnet] , NOvest_efficientnet_GOOD_dict[NOvest_efficientnet]) , max(NOvest_yolo_BAD_dict[NOvest_yolo] , NOvest_yolo_UNCERTAIN_dict[NOvest_yolo]) , vest_yolo_GOOD_dict[vest_yolo] )                 
    return output
def rule_vest16(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( max(NOvest_efficientnet_UNCERTAIN_dict[NOvest_efficientnet] , NOvest_efficientnet_GOOD_dict[NOvest_efficientnet]) , vest_yolo_BAD_dict[vest_yolo] , NOvest_yolo_BAD_dict[NOvest_yolo] )                 
    return output
def rule_vest17(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( max(NOvest_efficientnet_UNCERTAIN_dict[NOvest_efficientnet] , NOvest_efficientnet_GOOD_dict[NOvest_efficientnet]) , NOvest_yolo_BAD_dict[NOvest_yolo] , vest_yolo_UNCERTAIN_dict[vest_yolo] )                 
    return output
def rule_vest18(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet):
    output = min( NOvest_efficientnet_GOOD_dict[NOvest_efficientnet]  , vest_yolo_GOOD_dict[vest_yolo] , NOvest_yolo_GOOD_dict[NOvest_yolo] )                 
    return output
def rule_gloves1(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( max(gloves_efficientnet_UNCERTAIN_dict[gloves_efficientnet] , gloves_efficientnet_GOOD_dict[gloves_efficientnet]) , max(gloves_yolo_UNCERTAIN_dict[gloves_yolo] , gloves_yolo_GOOD_dict[gloves_yolo]) , max(NOgloves_yolo_BAD_dict[NOgloves_yolo] , NOgloves_yolo_UNCERTAIN_dict[NOgloves_yolo]) )                 
    return output
def rule_gloves2(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( gloves_efficientnet_UNCERTAIN_dict[gloves_efficientnet]  , gloves_yolo_GOOD_dict[gloves_yolo] , NOgloves_yolo_GOOD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves3(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( max(gloves_efficientnet_UNCERTAIN_dict[gloves_efficientnet] , gloves_efficientnet_GOOD_dict[gloves_efficientnet]) , max(gloves_yolo_BAD_dict[gloves_yolo] , gloves_yolo_UNCERTAIN_dict[gloves_yolo]) , NOgloves_yolo_GOOD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves4(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( max(gloves_efficientnet_UNCERTAIN_dict[gloves_efficientnet] , gloves_efficientnet_GOOD_dict[gloves_efficientnet]) , gloves_yolo_BAD_dict[gloves_yolo] , NOgloves_yolo_BAD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves5(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( max(gloves_efficientnet_UNCERTAIN_dict[gloves_efficientnet] , gloves_efficientnet_GOOD_dict[gloves_efficientnet]) , gloves_yolo_BAD_dict[gloves_yolo] , NOgloves_yolo_UNCERTAIN_dict[NOgloves_yolo] )                 
    return output
def rule_gloves6(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( gloves_efficientnet_GOOD_dict[gloves_efficientnet]  , gloves_yolo_GOOD_dict[gloves_yolo] , NOgloves_yolo_GOOD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves7(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( min(gloves_efficientnet_BAD_dict[gloves_efficientnet] , NOgloves_efficientnet_BAD_dict[NOgloves_efficientnet]) , gloves_yolo_GOOD_dict[gloves_yolo] , NOgloves_yolo_GOOD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves8(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( min(gloves_efficientnet_BAD_dict[gloves_efficientnet] , NOgloves_efficientnet_BAD_dict[NOgloves_efficientnet]) , gloves_yolo_GOOD_dict[gloves_yolo] , NOgloves_yolo_BAD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves9(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( min(gloves_efficientnet_BAD_dict[gloves_efficientnet] , NOgloves_efficientnet_BAD_dict[NOgloves_efficientnet]) , gloves_yolo_UNCERTAIN_dict[gloves_yolo] , max(NOgloves_yolo_BAD_dict[NOgloves_yolo], NOgloves_yolo_UNCERTAIN_dict[NOgloves_yolo], NOgloves_yolo_GOOD_dict[NOgloves_yolo]) ) 
    return output
def rule_gloves10(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( min(gloves_efficientnet_BAD_dict[gloves_efficientnet] , NOgloves_efficientnet_BAD_dict[NOgloves_efficientnet]) , max(gloves_yolo_BAD_dict[gloves_yolo] , gloves_yolo_GOOD_dict[gloves_yolo]) , NOgloves_yolo_UNCERTAIN_dict[NOgloves_yolo] )                 
    return output
def rule_gloves11(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( min(gloves_efficientnet_BAD_dict[gloves_efficientnet] , NOgloves_efficientnet_BAD_dict[NOgloves_efficientnet]) , gloves_yolo_BAD_dict[gloves_yolo] , NOgloves_yolo_GOOD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves12(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( min(gloves_efficientnet_BAD_dict[gloves_efficientnet] , NOgloves_efficientnet_BAD_dict[NOgloves_efficientnet]) , gloves_yolo_BAD_dict[gloves_yolo] , NOgloves_yolo_BAD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves13(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( max(NOgloves_efficientnet_UNCERTAIN_dict[NOgloves_efficientnet] , NOgloves_efficientnet_GOOD_dict[NOgloves_efficientnet]) , max(NOgloves_yolo_UNCERTAIN_dict[NOgloves_yolo] , NOgloves_yolo_GOOD_dict[NOgloves_yolo]) , max(gloves_yolo_BAD_dict[gloves_yolo] , gloves_yolo_UNCERTAIN_dict[gloves_yolo]) )                 
    return output
def rule_gloves14(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( NOgloves_efficientnet_UNCERTAIN_dict[NOgloves_efficientnet]  , gloves_yolo_GOOD_dict[gloves_yolo] , NOgloves_yolo_GOOD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves15(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( max(NOgloves_efficientnet_UNCERTAIN_dict[NOgloves_efficientnet] , NOgloves_efficientnet_GOOD_dict[NOgloves_efficientnet]) , max(NOgloves_yolo_BAD_dict[NOgloves_yolo] , NOgloves_yolo_UNCERTAIN_dict[NOgloves_yolo]) , gloves_yolo_GOOD_dict[gloves_yolo] )                 
    return output
def rule_gloves16(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( max(NOgloves_efficientnet_UNCERTAIN_dict[NOgloves_efficientnet] , NOgloves_efficientnet_GOOD_dict[NOgloves_efficientnet]) , gloves_yolo_BAD_dict[gloves_yolo] , NOgloves_yolo_BAD_dict[NOgloves_yolo] )                 
    return output
def rule_gloves17(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( max(NOgloves_efficientnet_UNCERTAIN_dict[NOgloves_efficientnet] , NOgloves_efficientnet_GOOD_dict[NOgloves_efficientnet]) , NOgloves_yolo_BAD_dict[NOgloves_yolo] , gloves_yolo_UNCERTAIN_dict[gloves_yolo] )                 
    return output
def rule_gloves18(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet):
    output = min( NOgloves_efficientnet_GOOD_dict[NOgloves_efficientnet]  , gloves_yolo_GOOD_dict[gloves_yolo] , NOgloves_yolo_GOOD_dict[NOgloves_yolo] )                 
    return output
def rule_final1(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final2(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output    

def rule_final3(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final4(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final5(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final6(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final7(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final8(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final9(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final10(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final11(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output    

def rule_final12(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final13(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final14(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final15(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final16(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final17(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final18(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final19(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final20(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output    

def rule_final21(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final22(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final23(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final24(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final25(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final26(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final27(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_GOOD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output


def rule_final28(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final29(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output    

def rule_final30(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final31(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final32(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final33(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final34(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final35(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final36(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final37(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final38(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output    

def rule_final39(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final40(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final41(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final42(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final43(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final44(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final45(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final46(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final47(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output    

def rule_final48(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final49(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final50(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final51(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final52(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final53(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final54(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_UNCERTAIN_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final55(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final56(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output    

def rule_final57(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final58(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final59(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final60(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final61(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final62(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final63(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_GOOD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final64(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final65(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output    

def rule_final66(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final67(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final68(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final69(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final70(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final71(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final72(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_UNCERTAIN_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final73(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final74(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output    

def rule_final75(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_GOOD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final76(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final77(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final78(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_UNCERTAIN_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output

def rule_final79(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_GOOD_dict[NOgloves_result] )                 
    return output

def rule_final80(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_UNCERTAIN_dict[NOgloves_result] )                 
    return output

def rule_final81(human_result, NOhat_result, NOvest_result, NOgloves_result):
    output = min( human_FUZZY_BAD_dict[human_result]  , NOhat_FUZZY_BAD_dict[NOhat_result] , NOvest_FUZZY_BAD_dict[NOvest_result], NOgloves_FUZZY_BAD_dict[NOgloves_result] )                 
    return output


def INFERENCE(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo, hat_efficientnet, NOhat_efficientnet, vest_efficientnet, NOvest_efficientnet, gloves_efficientnet, NOgloves_efficientnet):



    human_yolo = human_yolo
    hat_yolo = hat_yolo
    NOhat_yolo = NOhat_yolo
    vest_yolo = vest_yolo
    NOvest_yolo = NOvest_yolo
    gloves_yolo = gloves_yolo
    NOgloves_yolo = NOgloves_yolo
    hat_efficientnet = hat_efficientnet
    NOhat_efficientnet = NOhat_efficientnet
    vest_efficientnet = vest_efficientnet
    NOvest_efficientnet = NOvest_efficientnet
    gloves_efficientnet = gloves_efficientnet
    NOgloves_efficientnet = NOgloves_efficientnet

    if (human_yolo == '0.10') or (human_yolo == '0.20') or (human_yolo == '0.30') or (human_yolo == '0.40') or (human_yolo == '0.50') or (human_yolo == '0.60') or (human_yolo == '0.70') or (human_yolo == '0.80') or (human_yolo == '0.90'):
        human_yolo = human_yolo[:-1]
    if (hat_yolo == '0.10') or (hat_yolo == '0.20') or (hat_yolo == '0.30') or (hat_yolo == '0.40') or (hat_yolo == '0.50') or (hat_yolo == '0.60') or (hat_yolo == '0.70') or (hat_yolo == '0.80') or (hat_yolo == '0.90'):
        hat_yolo = hat_yolo[:-1]
    if (NOhat_yolo == '0.10') or (NOhat_yolo == '0.20') or (NOhat_yolo == '0.30') or (NOhat_yolo == '0.40') or (NOhat_yolo == '0.50') or (NOhat_yolo == '0.60') or (NOhat_yolo == '0.70') or (NOhat_yolo == '0.80') or (NOhat_yolo == '0.90'):
        NOhat_yolo = NOhat_yolo[:-1]
    if (vest_yolo == '0.10') or (vest_yolo == '0.20') or (vest_yolo == '0.30') or (vest_yolo == '0.40') or (vest_yolo == '0.50') or (vest_yolo == '0.60') or (vest_yolo == '0.70') or (vest_yolo == '0.80') or (vest_yolo == '0.90'):
        vest_yolo = vest_yolo[:-1]
    if (NOvest_yolo == '0.10') or (NOvest_yolo == '0.20') or (NOvest_yolo == '0.30') or (NOvest_yolo == '0.40') or (NOvest_yolo == '0.50') or (NOvest_yolo == '0.60') or (NOvest_yolo == '0.70') or (NOvest_yolo == '0.80') or (NOvest_yolo == '0.90'):
        NOvest_yolo = NOvest_yolo[:-1]
    if (gloves_yolo == '0.10') or (gloves_yolo == '0.20') or (gloves_yolo == '0.30') or (gloves_yolo == '0.40') or (gloves_yolo == '0.50') or (gloves_yolo == '0.60') or (gloves_yolo == '0.70') or (gloves_yolo == '0.80') or (gloves_yolo == '0.90'):
        gloves_yolo = gloves_yolo[:-1]
    if (NOgloves_yolo == '0.10') or (NOgloves_yolo == '0.20') or (NOgloves_yolo == '0.30') or (NOgloves_yolo == '0.40') or (NOgloves_yolo == '0.50') or (NOgloves_yolo == '0.60') or (NOgloves_yolo == '0.70') or (NOgloves_yolo == '0.80') or (NOgloves_yolo == '0.90'):
        NOgloves_yolo = NOgloves_yolo[:-1]
    if (hat_efficientnet == '0.10') or (hat_efficientnet == '0.20') or (hat_efficientnet == '0.30') or (hat_efficientnet == '0.40') or (hat_efficientnet == '0.50') or (hat_efficientnet == '0.60') or (hat_efficientnet == '0.70') or (hat_efficientnet == '0.80') or (hat_efficientnet == '0.90'):
        hat_efficientnet = hat_efficientnet[:-1]
    if (NOhat_efficientnet == '0.10') or (NOhat_efficientnet == '0.20') or (NOhat_efficientnet == '0.30') or (NOhat_efficientnet == '0.40') or (NOhat_efficientnet == '0.50') or (NOhat_efficientnet == '0.60') or (NOhat_efficientnet == '0.70') or (NOhat_efficientnet == '0.80') or (NOhat_efficientnet == '0.90'):
        NOhat_efficientnet = NOhat_efficientnet[:-1]
    if (vest_efficientnet == '0.10') or (vest_efficientnet == '0.20') or (vest_efficientnet == '0.30') or (vest_efficientnet == '0.40') or (vest_efficientnet == '0.50') or (vest_efficientnet == '0.60') or (vest_efficientnet == '0.70') or (vest_efficientnet == '0.80') or (vest_efficientnet == '0.90'):
        vest_efficientnet = vest_efficientnet[:-1]
    if (NOvest_efficientnet == '0.10') or (NOvest_efficientnet == '0.20') or (NOvest_efficientnet == '0.30') or (NOvest_efficientnet == '0.40') or (NOvest_efficientnet == '0.50') or (NOvest_efficientnet == '0.60') or (NOvest_efficientnet == '0.70') or (NOvest_efficientnet == '0.80') or (NOvest_efficientnet == '0.90'):
        NOvest_efficientnet = NOvest_efficientnet[:-1]
    if (gloves_efficientnet == '0.10') or (gloves_efficientnet == '0.20') or (gloves_efficientnet == '0.30') or (gloves_efficientnet == '0.40') or (gloves_efficientnet == '0.50') or (gloves_efficientnet == '0.60') or (gloves_efficientnet == '0.70') or (gloves_efficientnet == '0.80') or (gloves_efficientnet == '0.90'):
        gloves_efficientnet = gloves_efficientnet[:-1]
    if (NOgloves_efficientnet == '0.10') or (NOgloves_efficientnet == '0.20') or (NOgloves_efficientnet == '0.30') or (NOgloves_efficientnet == '0.40') or (NOgloves_efficientnet == '0.50') or (NOgloves_efficientnet == '0.60') or (NOgloves_efficientnet == '0.70') or (NOgloves_efficientnet == '0.80') or (NOgloves_efficientnet == '0.90'):
        NOgloves_efficientnet = NOgloves_efficientnet[:-1]



    if (human_yolo == '0.0') or (human_yolo == '0.00'):
        human_yolo = '0'
    if (hat_yolo == '0.0') or (hat_yolo == '0.00'):
        hat_yolo = '0'
    if (NOhat_yolo == '0.0') or (NOhat_yolo == '0.00'):
        NOhat_yolo = '0'
    if (vest_yolo == '0.0') or (vest_yolo == '0.00'):
        vest_yolo = '0'
    if (NOvest_yolo == '0.0') or (NOvest_yolo == '0.00'):
        NOvest_yolo = '0'
    if (gloves_yolo == '0.0') or (gloves_yolo == '0.00'):
        gloves_yolo = '0'
    if (NOgloves_yolo == '0.0') or (NOgloves_yolo == '0.00'):
        NOgloves_yolo = '0'
    if (hat_efficientnet == '0.0') or (hat_efficientnet == '0.00'):
        hat_efficientnet = '0'
    if (NOhat_efficientnet == '0.0') or (NOhat_efficientnet == '0.00'):
        NOhat_efficientnet = '0'
    if (vest_efficientnet == '0.0') or (vest_efficientnet == '0.00'):
        vest_efficientnet = '0'
    if (NOvest_efficientnet == '0.0') or (NOvest_efficientnet == '0.00'):
        NOvest_efficientnet = '0'
    if (gloves_efficientnet == '0.0') or (gloves_efficientnet == '0.00'):
        gloves_efficientnet = '0'
    if (NOgloves_efficientnet == '0.0') or (NOgloves_efficientnet == '0.00'):
        NOgloves_efficientnet = '0'                                                                                        






    humanGOOD_max = max( rule_human1(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo) , rule_human5(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo))
    humanUNCERTAIN_max = max( rule_human2(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo) , rule_human4(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo))
    humanBAD_max = rule_human3(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo)
   
  
    NOhatGOOD_max = max( rule_hat11(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat13(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat16(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat18(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) )
    NOhatUNCERTAIN_max = max( rule_hat2(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat3(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat5(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat7(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat9(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat10(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat12(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat14(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat15(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat17(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) )
    NOhatBAD_max = max( rule_hat1(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat4(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat6(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) , rule_hat8(hat_efficientnet, hat_yolo, NOhat_yolo, NOhat_efficientnet) )  
    NOvestGOOD_max = max( rule_vest11(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest13(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest16(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest18(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) )
    NOvestUNCERTAIN_max = max( rule_vest2(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest3(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest5(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest7(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest9(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest10(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest12(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest14(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest15(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest17(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) )
    NOvestBAD_max = max( rule_vest1(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest4(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest6(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) , rule_vest8(vest_efficientnet, vest_yolo, NOvest_yolo, NOvest_efficientnet) )
    NOglovesGOOD_max = max( rule_gloves11(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves13(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves16(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves18(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) )
    NOglovesUNCERTAIN_max = max( rule_gloves2(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves3(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves5(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves7(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves9(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves10(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves12(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves14(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves15(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves17(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) )
    NOglovesBAD_max = max( rule_gloves1(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves4(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves6(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) , rule_gloves8(gloves_efficientnet, gloves_yolo, NOgloves_yolo, NOgloves_efficientnet) )
   
   

    # print('humanGOOD_max', humanGOOD_max)
    # print('humanUNCERTAIN_max', humanUNCERTAIN_max)
    # print('humanBAD_max', humanBAD_max)
    # print('NOhatGOOD_max', NOhatGOOD_max)
    # print('NOhatUNCERTAIN_max', NOhatUNCERTAIN_max)
    # print('NOhatBAD_max', NOhatBAD_max)    
    # print('NOvestGOOD_max', NOvestGOOD_max)
    # print('NOvestUNCERTAIN_max', NOvestUNCERTAIN_max)
    # print('NOvestBAD_max', NOvestBAD_max)    
    # print('NOglovesGOOD_max', NOglovesGOOD_max)
    # print('NOglovesUNCERTAIN_max', NOglovesUNCERTAIN_max)
    # print('NOglovesBAD_max', NOglovesBAD_max)

    human_final_list = []
    NOhat_final_list = []
    NOvest_final_list = []
    NOgloves_final_list = []


    for i in range(0,101):
        humannn = max((humanGOOD_max * human_FUZZY_GOOD_dict[str(i/100)]) , (humanUNCERTAIN_max * human_FUZZY_UNCERTAIN_dict[str(i/100)]) , (humanBAD_max * human_FUZZY_BAD_dict[str(i/100)]) )
        human_final_list.append(humannn)
        NOhattt = max((NOhatGOOD_max * NOhat_FUZZY_GOOD_dict[str(i/100)]) , (NOhatUNCERTAIN_max * NOhat_FUZZY_UNCERTAIN_dict[str(i/100)]) , (NOhatBAD_max * NOhat_FUZZY_BAD_dict[str(i/100)]) )
        NOhat_final_list.append(NOhattt)
        NOvesttt = max((NOvestGOOD_max * NOvest_FUZZY_GOOD_dict[str(i/100)]) , (NOvestUNCERTAIN_max * NOvest_FUZZY_UNCERTAIN_dict[str(i/100)]) , (NOvestBAD_max * NOvest_FUZZY_BAD_dict[str(i/100)]) )
        NOvest_final_list.append(NOvesttt)
        NOglovesss = max((NOglovesGOOD_max * NOgloves_FUZZY_GOOD_dict[str(i/100)]) , (NOglovesUNCERTAIN_max * NOgloves_FUZZY_UNCERTAIN_dict[str(i/100)]) , (NOglovesBAD_max * NOgloves_FUZZY_BAD_dict[str(i/100)]) )
        NOgloves_final_list.append(NOglovesss)

    human_sum = 0
    for i in range(0, 101): 
        human_sum += human_final_list[i]   
    human_sum_middle = human_sum / 2
    human_sum_2 = 0    
    human_zarib = 0
    for i in range(0, 101): 
        human_sum_2 += human_final_list[i]
        if human_sum_2 > human_sum_middle and human_zarib == 0:
            human_result = str(i/100)
            human_zarib =1

    NOhat_sum = 0
    for i in range(0, 101): 
        NOhat_sum += NOhat_final_list[i]   
    NOhat_sum_middle = NOhat_sum / 2
    NOhat_sum_2 = 0    
    NOhat_zarib = 0
    for i in range(0, 101): 
        NOhat_sum_2 += NOhat_final_list[i]
        if NOhat_sum_2 > NOhat_sum_middle and NOhat_zarib == 0:
            NOhat_result = str(i/100)
            NOhat_zarib =1

    NOvest_sum = 0
    for i in range(0, 101): 
        NOvest_sum += NOvest_final_list[i]   
    NOvest_sum_middle = NOvest_sum / 2
    NOvest_sum_2 = 0    
    NOvest_zarib = 0
    for i in range(0, 101): 
        NOvest_sum_2 += NOvest_final_list[i]
        if NOvest_sum_2 > NOvest_sum_middle and NOvest_zarib == 0:
            NOvest_result = str(i/100)
            NOvest_zarib =1

    NOgloves_sum = 0
    for i in range(0, 101): 
        NOgloves_sum += NOgloves_final_list[i]   
    NOgloves_sum_middle = NOgloves_sum / 2
    NOgloves_sum_2 = 0    
    NOgloves_zarib = 0
    for i in range(0, 101): 
        NOgloves_sum_2 += NOgloves_final_list[i]
        if NOgloves_sum_2 > NOgloves_sum_middle and NOgloves_zarib == 0:
            NOgloves_result = str(i/100)
            NOgloves_zarib =1



    # print('')
    # print('human_result', human_result)
    # print('NOhat_result', NOhat_result)
    # print('NOvest_result', NOvest_result)
    # print('NOgloves_result', NOgloves_result)


    resultVerySafe_max = max( rule_final18(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final24(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final26(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final27(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final45(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final51(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final53(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final54(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final63(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final69(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final71(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final72(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final75(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final77(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final78(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final79(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final80(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final81(human_result, NOhat_result, NOvest_result, NOgloves_result)) 
    resultSafe_max = max( rule_final9(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final15(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final17(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final21(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final23(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final25(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final39(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final42(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final44(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final15(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final48(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final50(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final52(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final60(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final62(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final66(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final68(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final70(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final74(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final76(human_result, NOhat_result, NOvest_result, NOgloves_result))
    resultMediumDanger_max = max( rule_final6(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final8(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final12(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final14(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final16(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final20(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final22(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final33(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final35(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final39(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final41(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final43(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final47(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final49(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final57(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final59(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final61(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final65(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final67(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final73(human_result, NOhat_result, NOvest_result, NOgloves_result))
    resultDangerous_max = max( rule_final3(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final5(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final7(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final11(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final13(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final19(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final30(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final32(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final34(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final38(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final40(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final46(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final55(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final56(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final58(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final64(human_result, NOhat_result, NOvest_result, NOgloves_result))
    resultVeryDangerous_max = max( rule_final1(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final2(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final4(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final10(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final28(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final29(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final31(human_result, NOhat_result, NOvest_result, NOgloves_result) , rule_final37(human_result, NOhat_result, NOvest_result, NOgloves_result))


    # print('')
    # print('resultVerySafe_max', resultVerySafe_max)
    # print('resultSafe_max', resultSafe_max)
    # print('resultMediumDanger_max', resultMediumDanger_max)
    # print('resultDangerous_max', resultDangerous_max)
    # print('resultVeryDangerous_max', resultVeryDangerous_max)

    result_final_list = []

    for i in range(0,101):
        resultt = max((resultVerySafe_max * result_FUZZY_VerySafe_dict[str(i/100)]) , (resultSafe_max * result_FUZZY_safe_dict[str(i/100)]) , (resultMediumDanger_max * result_FUZZY_MediumDanger_dict[str(i/100)]) , (resultDangerous_max * result_FUZZY_Dangerous_dict[str(i/100)]) , (resultVeryDangerous_max * result_FUZZY_VeryDangerous_dict[str(i/100)]))
        result_final_list.append(resultt)


    result_sum = 0
    for i in range(0, 101): 
        result_sum += result_final_list[i]   
    result_sum_middle = result_sum / 2
    result_sum_2 = 0    
    result_zarib = 0
    for i in range(0, 101): 
        result_sum_2 += result_final_list[i]
        if result_sum_2 > result_sum_middle and result_zarib == 0:
            result_result = str(i/100)
            result_zarib =1


    # print('')
    # print('result_result', result_result)
    
    return result_result, NOhat_result, NOvest_result, NOgloves_result, human_result












load_classification_models_start = time.time()
# GlovesClassifierModel = keras.models.load_model('/content/drive/MyDrive/Classification/B3TVT/Gloves_EfficientnetB0_71_67_70.h5')
# HatClassifierModel = keras.models.load_model('/content/drive/MyDrive/Classification/B3TVT/Hat_EfficientnetB3TVT_94_93_84.h5')
# VestClassifierModel = keras.models.load_model('/content/drive/MyDrive/Classification/B3TVT/Vest_EfficientnetB3TVT_96_98_97.h5')

load_classification_models_finish = time.time()
load_classification_models_time = load_classification_models_finish - load_classification_models_start


def IOU(x1min, y1min, x1max, y1max, x2min, y2min, x2max, y2max):
    xmin_intersection = max(x1min, x2min)
    ymin_intersection = max(y1min, y2min)
    xmax_intersection = min(x1max, x2max)
    ymax_intersection = min(y1max, y2max)
    intersection = (ymax_intersection - ymin_intersection) * (xmax_intersection - xmin_intersection)
    union = ((y1max - y1min) * (x1max - x1min)) + ((y2max - y2min) * (x2max - x2min)) - intersection
    if union == 0:
        IOU = 0
    else:
        iou = intersection / union 
        if 0 <= iou <= 1:
            IOU = iou
        else:
            IOU = 0
    return IOU


def vis_result(img, results):

    list_human_conf_intbbox1bbox3bbox0bbox2 = []
    list_PPE_conf_intbbox1bbox3bbox0bbox2 = []

    for res_i, res in enumerate(results):

    ### compliance start


        if str(res[:3][0]) != 'human':

            int_PPE_conf_intbbox1bbox3bbox0bbox2 = [] 
            int_PPE_conf_intbbox1bbox3bbox0bbox2 = [str(res[:3][0]), res[:3][1], int(res[:3][2][1]), int(res[:3][2][3]), int(res[:3][2][0]), int(res[:3][2][2])]         
            list_PPE_conf_intbbox1bbox3bbox0bbox2.append(int_PPE_conf_intbbox1bbox3bbox0bbox2)



    ### compliance finish

        labelamir, confamir, bboxamir = res[:3]

        if str(labelamir) == 'human':

            label, conf, bbox = res[:3]

            bbox = [int(i) for i in bbox]
            if len(res) > 3:
                reid_feat = res[4]
                print("reid feat dim {}".format(len(reid_feat)))

            color = 0      #label_color[opt.label_name.index(label)]
            # show box
            # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # show label and conf

            int_human_conf_intbbox1bbox3bbox0bbox2 = [] 
            int_human_conf_intbbox1bbox3bbox0bbox2 = [conf, int(bbox[1]), int(bbox[3]), int(bbox[0]), int(bbox[2])]         
            list_human_conf_intbbox1bbox3bbox0bbox2.append(int_human_conf_intbbox1bbox3bbox0bbox2)


            txt = '{}:{:.2f}'.format(label, conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(txt, font, 0.4, 2)[0]
            # cv2.rectangle(img, (bbox[0], bbox[1] - txt_size[1] - 2), (bbox[0] + txt_size[0], bbox[1] - 2), color, -1)
            # cv2.putText(img, txt, (bbox[0], bbox[1] - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return img, list_human_conf_intbbox1bbox3bbox0bbox2, list_PPE_conf_intbbox1bbox3bbox0bbox2


# def detect_video():
#     detector = Detector(opt)
#     video_dir = opt.video_dir
#     save_folder = "output_video"

#     assert os.path.isfile(video_dir), "cannot find {}".format(video_dir)
#     cap = cv2.VideoCapture(video_dir)
#     # cap = cv2.VideoCapture(0)
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_num = int(cap.get(7))
#     mkdir(save_folder)
#     save_path = os.path.join(save_folder, os.path.basename(video_dir))
#     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
#     idx = 1
#     while True:
#         ret_val, frame = cap.read()
#         if ret_val:
#             print("detect frame {}/{}".format(idx, frame_num))
#             results = detector.run(frame, vis_thresh=opt.vis_thresh, show_time=True)
#             print(results)
#             frame = vis_result(frame, results)
#             vid_writer.write(frame)
#             idx += 1
#         else:
#             break
#     vid_writer.release()
#     print("save video to {}".format(save_path))


def detect():
    img_dir = opt.dataset_path + "/images/val2017" if "img_dir" not in opt else opt["img_dir"]
    output = "output"
    mkdir(output, rm=True)

    NOTCR = 0

    img_list = get_img_path(img_dir, extend=".jpg")
    assert len(img_list) != 0, "cannot find img in {}".format(img_dir)

    model = tf.keras.models.load_model('/content/drive/MyDrive/Classification/jadid_B0TVT_TFdata/B0_160_2ndPhaseAll3_ckpt')

    def PSNR(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
                      # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr  


#     generator = tf.keras.models.load_model('/content/drive/MyDrive/edsr/edsr_filter64_resblock8_valpsnr38.7.ckpt', custom_objects={'PSNR':PSNR})

    yolo_total_time = 0
    yolo_number = 0
    classification_total_time = 0
    classification_number = 0
    image_total_time = 0
    image_number = 0
    indexamirr = 0

    detector = Detector(opt)

    for index, image_path in enumerate(tqdm.tqdm(img_list)):
    
        indexamirr+=1
        if indexamirr >1:

            yolo_number+=1
            image_number+=1
            classification_number+=1


            yolo_total_time_start = time.time()
            image_total_time_start = time.time()

        print("------------------------------")
        print("{}/{}, {}".format(index, len(img_list), image_path))

        assert os.path.isfile(image_path), "cannot find {}".format(image_path)
        img = cv2.imread(image_path)
        s1 = time.time()
        results = detector.run(img, vis_thresh=opt.vis_thresh, show_time=True)
        # print("[pre_process + inference + post_process] time cost: {}s".format(time.time() - s1))
        # print('results', results)


        img, list_human_conf_intbbox1bbox3bbox0bbox2, list_PPE_conf_intbbox1bbox3bbox0bbox2 = vis_result(img, results)
        # print('list_human_conf_intbbox1bbox3bbox0bbox2', list_human_conf_intbbox1bbox3bbox0bbox2)
        # print('list_PPE_conf_intbbox1bbox3bbox0bbox2', list_PPE_conf_intbbox1bbox3bbox0bbox2)


        ### compliance start
        compliance_time_start = time.time()
        yolox_matches=[]
        for ppe in list_PPE_conf_intbbox1bbox3bbox0bbox2:
            corresponding_human = 0

            x1minamirr = ppe[4]
            y1minamirr = ppe[2]
            x1maxamirr = ppe[5]
            y1maxamirr = ppe[3]

            for human in list_human_conf_intbbox1bbox3bbox0bbox2:
                Npo = 0
                Nop = 0

                x2minamirr = human[3]
                y2minamirr = human[1]
                x2maxamirr = human[4]
                y2maxamirr = human[2]                

                iou = IOU(x1minamirr, y1minamirr, x1maxamirr, y1maxamirr, x2minamirr, y2minamirr, x2maxamirr, y2maxamirr)

                if iou > 0:

                    confppe = ppe[1]
                    confhuman = human[0]
                    ppetype = str(ppe[0])
                    for ppeee in list_PPE_conf_intbbox1bbox3bbox0bbox2:
                        if (str(ppeee[0]) == ppetype) or (str(ppeee[0]) == 'NO' + ppetype) or (str(ppeee[0]) == ppetype[2:]):

                            x1minppeee = ppeee[4]
                            y1minppeee = ppeee[2]
                            x1maxppeee = ppeee[5]
                            y1maxppeee = ppeee[3]

                            iou_po = IOU(x1minppeee, y1minppeee, x1maxppeee, y1maxppeee, x2minamirr, y2minamirr, x2maxamirr, y2maxamirr)
                            if iou_po > 0:
                                Npo+=1




                    for humannn in list_human_conf_intbbox1bbox3bbox0bbox2:

                        x2minhumannn = humannn[3]
                        y2minhumannn = humannn[1]
                        x2maxhumannn = humannn[4]
                        y2maxhumannn = humannn[2]

                        iou_op = IOU(x1minamirr, y1minamirr, x1maxamirr, y1maxamirr, x2minhumannn, y2minhumannn, x2maxhumannn, y2maxhumannn)
                        if iou_op > 0:
                            Nop+=1

                    MatchVal = 0.1*iou + 0.7*(2/(Npo+Nop)) + 0.2*((confppe+confhuman)/2)
                    candid_human = [MatchVal, confhuman, x2minamirr, y2minamirr, x2maxamirr, y2maxamirr]
                    if (corresponding_human == 0) or (candid_human[0] > corresponding_human[0]):
                        corresponding_human = candid_human

            if corresponding_human != 0:
                HUMANandPPE = corresponding_human + [ppetype, confppe, x1minamirr, y1minamirr, x1maxamirr, y1maxamirr]       
                yolox_matches.append(HUMANandPPE)         
                # print('HUMANandPPE', HUMANandPPE)
            else:
                # print('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
                NOTCR+=1


        list_morattab = []
        for i in range(len(yolox_matches)):
            if i == 0:
                list_morattab.append(yolox_matches[i])
                for j in range(i+1,len(yolox_matches)):
                    if yolox_matches[i][1] == yolox_matches[j][1]:
                        list_morattab.append(yolox_matches[j])

            else:
                if yolox_matches[i] not in list_morattab:
                    list_morattab.append(yolox_matches[i])
                    for j in range(i+1,len(yolox_matches)):
                        if yolox_matches[i][1] == yolox_matches[j][1]:
                            list_morattab.append(yolox_matches[j])


        # print('list_morattab: ', list_morattab)
        # for ijkl in list_morattab:
            # print('yolox_matches', ijkl)

        # avallll = 0
        # zarib = 0
        # for yolopee in list_morattab:
        #     zarib +=1 
        #     human_confidence = yolopee[1]
        #     if avallll == 0:
        #         textamir = yolopee[6] + ': ' + str(yolopee[7])[:4]
        #         if 'NO' in textamir:
        #             color = (0,0,255)
        #         else: 
        #             color = 255
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         txt_size = cv2.getTextSize(textamir, font, 0.4, 2)[0]
        #         cv2.rectangle(img, (yolopee[2], yolopee[3] ), (yolopee[2] + txt_size[0], yolopee[3] + txt_size[1]), color, -1)
        #         cv2.putText(img, textamir, (yolopee[2], yolopee[3] +8 ), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA) 
                
        #         avallll = 1
        #         conf_ghabli = human_confidence              
        #         nnn = 1

        #     elif human_confidence == conf_ghabli:
        #         textamir = yolopee[6] + ': ' + str(yolopee[7])[:4]
        #         if 'NO' in textamir:
        #             color = (0,0,255)
        #         else: 
        #             color = 255                
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         txt_size = cv2.getTextSize(textamir, font, 0.4, 2)[0]
        #         cv2.rectangle(img, (yolopee[2], yolopee[3] + nnn * txt_size[1]), (yolopee[2] + txt_size[0], yolopee[3] + txt_size[1] + nnn * txt_size[1]), color, -1)
        #         cv2.putText(img, textamir, (yolopee[2], yolopee[3] +8 + nnn * txt_size[1]), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA) 
        #         nnn+=1
                            
        #     else:

        #         nnn = 0
        #         textamir = yolopee[6] + ': ' + str(yolopee[7])[:4]
        #         if 'NO' in textamir:
        #             color = (0,0,255)
        #         else: 
        #             color = 255              
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         txt_size = cv2.getTextSize(textamir, font, 0.4, 2)[0]
        #         cv2.rectangle(img, (yolopee[2], yolopee[3] + nnn * txt_size[1]), (yolopee[2] + txt_size[0], yolopee[3] + txt_size[1] + nnn * txt_size[1]), color, -1)
        #         cv2.putText(img, textamir, (yolopee[2], yolopee[3] +8 + nnn * txt_size[1]), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA) 
        #         nnn = 1
        #         conf_ghabli = human_confidence              
                

        compliance_time_finish = time.time()
        # print('compliance_time: ', compliance_time_finish-compliance_time_start)



        ### compliance finish


        save_p = output + "/" + image_path.split("/")[-2]

        if indexamirr >1:
            yolo_total_time_finish = time.time()
            yolo_total_time += (yolo_total_time_finish - yolo_total_time_start)


        mkdir(save_p)
        save_img = save_p + "/" + os.path.basename(image_path)

        Classification_time_startt = time.time()

        if indexamirr >1:
            classification_total_time_start = time.time()


        imgtotal = image.load_img(image_path)
        imgtotal = image.img_to_array(imgtotal)

        human_list = []
        FUZZY_human_coordinates_list = []
        for i_human in list_human_conf_intbbox1bbox3bbox0bbox2:
            FUZZY_human_coordinates = [i_human[3], i_human[1], i_human[4], i_human[2]]
            humanimg = imgtotal[i_human[1]:i_human[2] , i_human[3]:i_human[4]]
            humanimg = image.array_to_img(humanimg)
            humanimg = humanimg.resize((80, 160))
            humanimg = image.img_to_array(humanimg)
            human_list.append(humanimg)
            FUZZY_human_coordinates_list.append(FUZZY_human_coordinates)

        if len(human_list) != 0 :
            all_human = human_list[0]
            all_human = np.expand_dims(all_human, axis=0)
            # all_human = generator.predict_step(all_human)

            for i in range(1, len(human_list)):
              
                # all_human = np.concatenate((all_human, generator.predict_step(np.expand_dims(human_list[i], axis=0))))

                all_human = np.concatenate((all_human, np.expand_dims(human_list[i], axis=0)))


            
            
            # all_human = generator.predict_step(all_human)

            
            # all_human = image.array_to_img(all_human.numpy())
            # all_human = all_human.resize((80, 160))
            # all_human = image.img_to_array(all_human)


            # Gloves_Classification_result = GlovesClassifierModel.predict(all_human)
            # Hat_Classification_result = HatClassifierModel.predict(all_human)
            # Vest_Classification_result = VestClassifierModel.predict(all_human)

            All3_result = model.predict(all_human)
            Hat_Classification_result = All3_result[0]
            Vest_Classification_result = All3_result[1]
            Gloves_Classification_result = All3_result[2]

            if indexamirr >1:            
                classification_total_time_finish = time.time()
                classification_total_time += (classification_total_time_finish - classification_total_time_start)

            Classification_time_finishh = time.time()
            Classification_time = Classification_time_finishh - Classification_time_startt

            Classification_result = []
            FUZZY_Classification_result = []
            for i in range(len(human_list)):
                Gloves_conf = Gloves_Classification_result[i][0]
                Hat_conf = Hat_Classification_result[i][0]
                Vest_conf = Vest_Classification_result[i][0]

                FUZZY_Classification_result.append([Hat_conf, Vest_conf, Gloves_conf])

                if Gloves_conf >= 0.5:
                    Gloves_label = 'Nogloves:' + str(Gloves_conf)[0:4]
                else:
                    Gloves_conf = 1 - Gloves_conf
                    Gloves_label = 'NSgloves:' + str(Gloves_conf)[0:4]

                if Hat_conf >= 0.5:
                    Hat_label = 'Nohat:' + str(Hat_conf)[0:4]
                else:
                    Hat_conf = 1 - Hat_conf
                    Hat_label = 'hat:' + str(Hat_conf)[0:4]

                if Vest_conf >= 0.5:
                    Vest_label = 'Novest:' + str(Vest_conf)[0:4]
                else:
                    Vest_conf = 1 - Vest_conf
                    Vest_label = 'Vest:' + str(Vest_conf)[0:4]                    
                # print('Vest_confAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA:', Vest_conf)
                Classification_result.append([Hat_label, Vest_label, Gloves_label])



            # for i in range(len(Classification_result)):
            #     ymin = list_human_conf_intbbox1bbox3bbox0bbox2[i][1]
            #     ymax = list_human_conf_intbbox1bbox3bbox0bbox2[i][2]
            #     xmin = list_human_conf_intbbox1bbox3bbox0bbox2[i][3]
            #     xmax = list_human_conf_intbbox1bbox3bbox0bbox2[i][4]

                # txt_hat = Classification_result[i][0]
                # txt_vest = Classification_result[i][1]
                # txt_gloves = Classification_result[i][2]

                # if 'No' in txt_hat:
                #     color = (0,0,255)
                # else: 
                #     color = 255
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # txt_size = cv2.getTextSize(txt_hat, font, 0.4, 2)[0]
                # cv2.rectangle(img, (xmin, ymax - txt_size[1] - txt_size[1] - txt_size[1] - 2), (xmin + txt_size[0], ymax - txt_size[1] - txt_size[1] - 2), color, -1)
                # cv2.putText(img, txt_hat, (xmin, ymax - txt_size[1] - txt_size[1] - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


                # if 'No' in txt_gloves:
                #     color = (0,0,255)
                # else: 
                #     color = 255
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # txt_size = cv2.getTextSize(txt_gloves, font, 0.4, 2)[0]
                # cv2.rectangle(img, (xmin, ymax - txt_size[1] - 2), (xmin + txt_size[0], ymax - 2), color, -1)
                # cv2.putText(img, txt_gloves, (xmin, ymax - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


                # if 'No' in txt_vest:
                #     color = (0,0,255)
                # else: 
                #     color = 255
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # txt_size = cv2.getTextSize(txt_vest, font, 0.4, 2)[0]
                # cv2.rectangle(img, (xmin, ymax - txt_size[1] - txt_size[1] - 2), (xmin + txt_size[0], ymax - txt_size[1] - 2), color, -1)
                # cv2.putText(img, txt_vest, (xmin, ymax - txt_size[1] - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)                



        ### FUZZY start

        human_shomare = 0
        for human_coordinates , Classification_confs in zip(FUZZY_human_coordinates_list, FUZZY_Classification_result):

            xminhuman = human_coordinates[0]
            yminhuman = human_coordinates[1]
            xmaxhuman = human_coordinates[2]
            ymaxhuman = human_coordinates[3]

                
            if Classification_confs[0] > 0.5:
                NOhat_efficientnet = str((Classification_confs[0] - 0.5) * 2)[:4]
                hat_efficientnet = str(0)
                if NOhat_efficientnet == '1':
                    NOhat_efficientnet = '1.0'
                if float(NOhat_efficientnet) > 1:
                    NOhat_efficientnet = str(0)

            else:
                NOhat_efficientnet = str(0)
                hat_efficientnet = str((0.5 - Classification_confs[0]) * 2)[:4]
                if hat_efficientnet == '1':
                    hat_efficientnet = '1.0'
                if float(hat_efficientnet) > 1:
                    hat_efficientnet = str(0)

            if Classification_confs[1] > 0.5:
                NOvest_efficientnet = str((Classification_confs[1] - 0.5) * 2)[:4]
                vest_efficientnet = str(0)
                if NOvest_efficientnet == '1':
                    NOvest_efficientnet = '1.0'
                if float(NOvest_efficientnet) > 1:
                    NOvest_efficientnet = str(0)

            else:
                NOvest_efficientnet = str(0)
                vest_efficientnet = str((0.5 - Classification_confs[1]) * 2)[:4]
                if vest_efficientnet == '1':
                    vest_efficientnet = '1.0'
                if float(vest_efficientnet) > 1:
                    vest_efficientnet = str(0)


            if Classification_confs[2] > 0.5:
                NOgloves_efficientnet = str((Classification_confs[2] - 0.5) * 2)[:4]
                gloves_efficientnet = str(0)
                if NOgloves_efficientnet == '1':
                    NOgloves_efficientnet = '1.0'
                if float(NOgloves_efficientnet) > 1:
                    NOgloves_efficientnet = str(0)

            else:
                NOgloves_efficientnet = str(0)
                gloves_efficientnet = str((0.5 - Classification_confs[2]) * 2)[:4]
                if gloves_efficientnet == '1':
                    gloves_efficientnet = '1.0'
                if float(gloves_efficientnet) > 1:
                    gloves_efficientnet = str(0)

            

            corresponding_hats = ['0.19']
            corresponding_NOhats = ['0.19']
            corresponding_vests = ['0.19']
            corresponding_NOvests = ['0.19']
            corresponding_gloveses = ['0.19']
            corresponding_NOgloveses = ['0.19']


            for humanppe in list_morattab:


                if [humanppe[2], humanppe[3], humanppe[4], humanppe[5]] == [xminhuman, yminhuman, xmaxhuman, ymaxhuman]:

                    human_yolo = str(humanppe[1])[:4]
                    if human_yolo == '1':
                        human_yolo = '1.0'
  
                    if humanppe[6] == 'hat':
                        neww = str(humanppe[7])[:4]
                        if neww == '1':
                            neww = '1.0'
                        corresponding_hats.append(neww)

                    if humanppe[6] == 'NOhat':
                        neww = str(humanppe[7])[:4]
                        if neww == '1':
                            neww = '1.0'
                        corresponding_NOhats.append(neww)

                    if humanppe[6] == 'vest':
                        neww = str(humanppe[7])[:4]
                        if neww == '1':
                            neww = '1.0'
                        corresponding_vests.append(neww)

                    if humanppe[6] == 'NOvest':
                        neww = str(humanppe[7])[:4]
                        if neww == '1':
                            neww = '1.0'
                        corresponding_NOvests.append(neww)

                    if humanppe[6] == 'gloves':
                        neww = str(humanppe[7])[:4]
                        if neww == '1':
                            neww = '1.0'
                        corresponding_gloveses.append(neww)

                    if humanppe[6] == 'NOgloves':
                        neww = str(humanppe[7])[:4]
                        if neww == '1':
                            neww = '1.0'
                        corresponding_NOgloveses.append(neww)
            
            
            hat_yolo = max(corresponding_hats)
            NOhat_yolo = max(corresponding_NOhats)
            vest_yolo = max(corresponding_vests)
            NOvest_yolo = max(corresponding_NOvests)
            gloves_yolo = max(corresponding_gloveses)
            NOgloves_yolo = max(corresponding_NOgloveses)



            # print('human_yolo', human_yolo)
            # print('hat_yolo', hat_yolo)
            # print('NOhat_yolo', NOhat_yolo)
            # print('vest_yolo', vest_yolo)
            # print('NOvest_yolo', NOvest_yolo)
            # print('gloves_yolo', gloves_yolo)
            # print('NOgloves_yolo', NOgloves_yolo)
            # print('hat_efficientnet', hat_efficientnet)
            # print('NOhat_efficientnet', NOhat_efficientnet)
            # print('vest_efficientnet', vest_efficientnet)
            # print('NOvest_efficientnet', NOvest_efficientnet)
            # print('gloves_efficientnet', gloves_efficientnet)
            # print('NOgloves_efficientnet', NOgloves_efficientnet)
            # print('')

            PPE_Danger_level, hat_Danger_level, vest_Danger_level, gloves_Danger_level, human_level = INFERENCE(human_yolo, hat_yolo, NOhat_yolo, vest_yolo, NOvest_yolo, gloves_yolo, NOgloves_yolo, hat_efficientnet, NOhat_efficientnet, vest_efficientnet, NOvest_efficientnet, gloves_efficientnet, NOgloves_efficientnet)

            # print('********************************************************************************************************************************************************')
            # print('human_level: ', human_level)            
            # print('PPE_Danger_level: ', PPE_Danger_level)
            # print('hat_Danger_level: ', hat_Danger_level)
            # print('vest_Danger_level: ', vest_Danger_level)
            # print('gloves_Danger_level: ', gloves_Danger_level)
            # print('********************************************************************************************************************************************************')



            if float(human_level) > 0.45:
                
                human_shomare+=1
                float_human_level = float(human_level)
                float_PPE_Danger_level = float(PPE_Danger_level)
                float_hat_Danger_level = float(hat_Danger_level)
                float_vest_Danger_level = float(vest_Danger_level)
                float_gloves_Danger_level = float(gloves_Danger_level)

                if float_human_level >=0.5:
                   human_level_modif = str(0.52 + (float_human_level-0.5)*1.51)
                else:
                   human_level_modif = str(0.5 - (0.5-float_human_level)*1.51)

                if float_PPE_Danger_level >=0.5:
                   PPE_Danger_level_modif = str(0.53 + (float_PPE_Danger_level-0.5)*1.22)
                else:
                   PPE_Danger_level_modif = str(0.48 - (0.5-float_PPE_Danger_level)*1.22)

                if float_hat_Danger_level >=0.5:
                   hat_Danger_level_modif = str(0.52 + (float_hat_Danger_level-0.5)*1.51)
                else:
                   hat_Danger_level_modif = str(0.49 - (0.5-float_hat_Danger_level)*1.51)

                if float_vest_Danger_level >=0.5:
                   vest_Danger_level_modif = str(0.52 + (float_vest_Danger_level-0.5)*1.51)
                else:
                   vest_Danger_level_modif = str(0.49 - (0.5-float_vest_Danger_level)*1.51)

                if float_gloves_Danger_level >=0.5:
                   gloves_Danger_level_modif = str(0.52 + (float_gloves_Danger_level-0.5)*1.51)
                else:
                   gloves_Danger_level_modif = str(0.49 - (0.5-float_gloves_Danger_level)*1.51)                                    



                color = 0      #label_color[opt.label_name.index(label)]
                # show box
                cv2.rectangle(img, (xminhuman, yminhuman), (xmaxhuman, ymaxhuman), color, 2)
                # show label and conf


                txt = 'W: ' + human_level_modif[0:4]
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt, font, 0.4, 2)[0]
                cv2.rectangle(img, (xminhuman, yminhuman - txt_size[1] - 2), (xminhuman + txt_size[0], yminhuman - 2), color, -1)
                cv2.putText(img, txt, (xminhuman, yminhuman - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)



                txt_PPE_Danger_level = 'PPE:' + PPE_Danger_level_modif[0:4]
                txt_hat_Danger_level = 'H:' + hat_Danger_level_modif[0:4]
                txt_vest_Danger_level = 'V:' + vest_Danger_level_modif[0:4]
                txt_gloves_Danger_level = 'G:' + gloves_Danger_level_modif[0:4]




                if float(PPE_Danger_level) > 0.5:
                    color = (0,0,255)
                else: 
                    color = 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt_PPE_Danger_level, font, 0.4, 2)[0]
                cv2.rectangle(img, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - txt_size[1] - txt_size[1] - 2), (xminhuman + txt_size[0], ymaxhuman - txt_size[1] - txt_size[1] - txt_size[1] - 2), color, -1)
                cv2.putText(img, txt_PPE_Danger_level, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - txt_size[1] - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


                if float(hat_Danger_level) > 0.5:
                    color = (0,0,255)
                else: 
                    color = 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt_hat_Danger_level, font, 0.4, 2)[0]
                cv2.rectangle(img, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - txt_size[1] - 2), (xminhuman + txt_size[0], ymaxhuman - txt_size[1] - txt_size[1] - 2), color, -1)
                cv2.putText(img, txt_hat_Danger_level, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


                if float(vest_Danger_level) > 0.5:
                    color = (0,0,255)
                else: 
                    color = 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt_vest_Danger_level, font, 0.4, 2)[0]
                cv2.rectangle(img, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - 2), (xminhuman + txt_size[0], ymaxhuman - txt_size[1] - 2), color, -1)
                cv2.putText(img, txt_vest_Danger_level, (xminhuman, ymaxhuman - txt_size[1] - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)     


                if float(gloves_Danger_level) > 0.5:
                    color = (0,0,255)
                else: 
                    color = 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt_gloves_Danger_level, font, 0.4, 2)[0]
                cv2.rectangle(img, (xminhuman, ymaxhuman - txt_size[1] - 2), (xminhuman + txt_size[0], ymaxhuman - 2), color, -1)
                cv2.putText(img, txt_gloves_Danger_level, (xminhuman, ymaxhuman - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)




             
             
             
                imgg = cv2.imread(image_path)


                color = 0      #label_color[opt.label_name.index(label)]
                # show box
                cv2.rectangle(imgg, (xminhuman, yminhuman), (xmaxhuman, ymaxhuman), color, 2)
                # show label and conf
                txt = 'W: ' + human_level_modif[0:4]
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt, font, 0.4, 2)[0]
                cv2.rectangle(imgg, (xminhuman, yminhuman - txt_size[1] - 2), (xminhuman + txt_size[0], yminhuman - 2), color, -1)
                cv2.putText(imgg, txt, (xminhuman, yminhuman - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


                if float(PPE_Danger_level) > 0.5:
                    color = (0,0,255)
                else: 
                    color = 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt_PPE_Danger_level, font, 0.4, 2)[0]
                cv2.rectangle(imgg, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - txt_size[1] - txt_size[1] - 2), (xminhuman + txt_size[0], ymaxhuman - txt_size[1] - txt_size[1] - txt_size[1] - 2), color, -1)
                cv2.putText(imgg, txt_PPE_Danger_level, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - txt_size[1] - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


                if float(hat_Danger_level) > 0.5:
                    color = (0,0,255)
                else: 
                    color = 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt_hat_Danger_level, font, 0.4, 2)[0]
                cv2.rectangle(imgg, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - txt_size[1] - 2), (xminhuman + txt_size[0], ymaxhuman - txt_size[1] - txt_size[1] - 2), color, -1)
                cv2.putText(imgg, txt_hat_Danger_level, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


                if float(vest_Danger_level) > 0.5:
                    color = (0,0,255)
                else: 
                    color = 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt_vest_Danger_level, font, 0.4, 2)[0]
                cv2.rectangle(imgg, (xminhuman, ymaxhuman - txt_size[1] - txt_size[1] - 2), (xminhuman + txt_size[0], ymaxhuman - txt_size[1] - 2), color, -1)
                cv2.putText(imgg, txt_vest_Danger_level, (xminhuman, ymaxhuman - txt_size[1] - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)     


                if float(gloves_Danger_level) > 0.5:
                    color = (0,0,255)
                else: 
                    color = 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt_gloves_Danger_level, font, 0.4, 2)[0]
                cv2.rectangle(imgg, (xminhuman, ymaxhuman - txt_size[1] - 2), (xminhuman + txt_size[0], ymaxhuman - 2), color, -1)
                cv2.putText(imgg, txt_gloves_Danger_level, (xminhuman, ymaxhuman - 2), font, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


                save_imgg = save_p + "/" + os.path.basename(image_path)[:-4] + '_' + str(human_shomare) + os.path.basename(image_path)[-4:]
                cv2.imwrite(save_imgg, imgg)





        ### FUZZY finish


        if indexamirr >1:
            image_total_time_finish = time.time()
            image_total_time += (image_total_time_finish - image_total_time_start)

        
        cv2.imwrite(save_img, img)
        print("save image to {}".format(save_img))

        # print('humanConcatenateAAAAAAmirshape : ' , all_human.shape)
        # print('list_human_conf_intbbox1bbox3bbox0bbox2AAAAAAMMMMMMMIIIIIIIIIIRRRRRRRRR : ' , list_human_conf_intbbox1bbox3bbox0bbox2)

        # print('Gloves_Classification_result : ' , Gloves_Classification_result)
        # print('Hat_Classification_result : ' , Hat_Classification_result)
        # print('Vest_Classification_result : ' , Vest_Classification_result)
        # print('Classification_time : ' , Classification_time)


    print(' ')
    
    akhar_akhar = time.time()
    time_kol = akhar_akhar - aval_aval
    print('Models Loading Time: ', time_kol - image_total_time)

    if image_number != 0:
        print('Image Average Time: ', str(image_total_time/image_number)[0:4], ' s')
        print('Average FPS: ', str(1 * image_number / image_total_time)[0:4])

    # print('yolo_total_time: ', yolo_total_time, '/', yolo_number, ': ', yolo_total_time/yolo_number)
    # print('classification_total_time: ', classification_total_time, '/', classification_number, ': ', classification_total_time/classification_number)

    # print(NOTCR)

    # print('load_classification_models_time: ', load_classification_models_time)


if __name__ == "__main__":
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")

    if 'video_dir' not in opt.keys():
        detect()
    else:
        detect_video()

