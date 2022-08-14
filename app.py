import streamlit as st
st.text('This is TEXT')

import os
import shutil
import gdown
import tensorflow as tf

st.text('All imported.')

url = 'https://drive.google.com/uc?export=download&id=1upo5sgFRlAZiPYXm7-nf-yv6ajqkINCz'
output = 'YOLOX.pth'
gdown.download(url, output, quiet=False)


model = tf.keras.models.load_model('EfficientNet')


########################################################################################################################

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

  
  
  
  
  ########################################################################################################
  
  
  
  
