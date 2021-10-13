import argparse
import os.path
import re
import sys
import tarfile
import cv2
import math
import numpy as np
from six.moves import urllib
import tensorflow as tf
import time
from gtts import gTTS
import pygame
import enum
import os, sys
# from threading import Thread
from sklearn.cluster import KMeans
import cv2

#===============================================================#
# color reference                                               #
#===============================================================#
class Colors_Name(enum.Enum):
	# 하양 베이지 라임 아이보리 개나리 노랑 살구 옥 은 귤
    white = 0
    UP_white = enum.auto()
    DOWN_white = enum.auto()
    # beige = enum.auto()
    # lime = enum.auto()
    # ivory = enum.auto()
    # forsythia = enum.auto()
    yellow = enum.auto()
    # apricot = enum.auto()
    # turquoise = enum.auto()
    # silver = enum.auto()
    # tangerine = enum.auto()
	# 연두 산호 하늘 주황 밝은파랑 시안 황토 자홍 분홍 담청 
    # chartreuse = enum.auto()
    # coral = enum.auto()
    sky_blue = enum.auto()
    UP_sky_blue = enum.auto()
    DOWN_sky_blue = enum.auto()
    # orange = enum.auto()
    # light_blue = enum.auto()
    # cyan = enum.auto()
    # ocher = enum.auto()
    # claret = enum.auto()
    # pink = enum.auto()
    # powder_blue = enum.auto()
	# 초록 바다 회색 밝은보라 빨강 올리브 에메랄드그린 카키 아쿠아마린 암청
    green = enum.auto()
    UP_green = enum.auto()
    # DOWN_green = enum.auto()
    # seagrass = enum.auto()
    # gray = enum.auto()
    # light_purple = enum.auto()
    red = enum.auto()
    UP_red = enum.auto()
    DOWN_red = enum.auto()
    # olive = enum.auto()
    # emerald_green = enum.auto()
    # khaki = enum.auto()
    # aqua_marine = enum.auto()
    # dark_blue = enum.auto()
	# 심홍 보라 파랑 갈색 청자 청록 군청 코발트블루 장미 자주
    # magenta = enum.auto()
    purple = enum.auto()
    UP_purple = enum.auto()
    DOWN_purple = enum.auto()
    blue = enum.auto()
    UP_blue = enum.auto()
    DOWN_blue = enum.auto()
    brown = enum.auto()
    UP_brown = enum.auto()
    DOWN_brown = enum.auto()
    # blue_purple = enum.auto()
    # blue_green = enum.auto()
    # ultramarine = enum.auto()
    # cobalt_blue = enum.auto()
    # rose = enum.auto()
    # amethyst = enum.auto()
	# 고동 남색 검정
    # anburn = enum.auto()
    # navy = enum.auto()
    UP_navy = enum.auto()
    DOWN_navy = enum.auto()
    black = enum.auto()
    UP_black = enum.auto()
    DOWN_black = enum.auto()

color_num = len(Colors_Name)
'''
# https://encycolorpedia.kr/named 참고
colors_rgb = np.zeros([color_num, 3], dtype = int)
colors_rgb[Colors_Name.white.value] = [233, 233, 233]
# colors_rgb[Colors_Name.beige.value] = [245, 245, 220]
# colors_rgb[Colors_Name.lime.value] = [191, 255, 0]
# colors_rgb[Colors_Name.ivory.value] = [236, 230, 204]
# colors_rgb[Colors_Name.forsythia.value] = [236, 230, 0]
# colors_rgb[Colors_Name.yellow.value] = [247, 212, 0]
# colors_rgb[Colors_Name.apricot.value] = [251, 206, 177]
# colors_rgb[Colors_Name.turquoise.value] = [131, 220, 183]
# colors_rgb[Colors_Name.silver.value] = [192, 192, 192]
# colors_rgb[Colors_Name.tangerine.value] = [248, 155, 0]
# colors_rgb[Colors_Name.chartreuse.value] = [129, 193, 71]
# colors_rgb[Colors_Name.coral.value] = [242, 152, 134]
# colors_rgb[Colors_Name.sky_blue.value] = [80, 188, 223]
# colors_rgb[Colors_Name.orange.value] = [255, 127, 0]
# colors_rgb[Colors_Name.light_blue.value] = [74, 168, 216]
# colors_rgb[Colors_Name.cyan.value] = [0, 163, 210]
# colors_rgb[Colors_Name.ocher.value] = [198, 138, 18]
# colors_rgb[Colors_Name.claret.value] = [255, 0, 255]
# colors_rgb[Colors_Name.pink.value] = [255, 51, 153]
# colors_rgb[Colors_Name.powder_blue.value] = [62, 145, 181]
colors_rgb[Colors_Name.green.value] = [76,94,82]
# colors_rgb[Colors_Name.seagrass.value] = [0, 128, 255]
# colors_rgb[Colors_Name.gray.value] = [128, 128, 128]
# colors_rgb[Colors_Name.light_purple.value] = [137, 119, 173]
colors_rgb[Colors_Name.red.value] = [168, 37, 53]
# colors_rgb[Colors_Name.olive.value] = [128, 128, 0]
# colors_rgb[Colors_Name.emerald_green.value] = [0, 141, 98]
# colors_rgb[Colors_Name.khaki.value] = [143, 120, 75]
# colors_rgb[Colors_Name.aqua_marine.value] = [94, 126, 155]
# colors_rgb[Colors_Name.dark_blue.value] = [0, 128, 128]
# colors_rgb[Colors_Name.magenta.value] = [220, 20, 60]
colors_rgb[Colors_Name.purple.value] = [104, 71, 176]
colors_rgb[Colors_Name.blue.value] = [31, 63, 162]
colors_rgb[Colors_Name.brown.value] = [134, 71, 53]
# colors_rgb[Colors_Name.blue_purple.value] = [105, 55, 161]
# colors_rgb[Colors_Name.blue_green.value] = [58, 77, 69]
# colors_rgb[Colors_Name.ultramarine.value] = [70, 73, 100]
# colors_rgb[Colors_Name.cobalt_blue.value] = [0, 73, 140]
# colors_rgb[Colors_Name.rose.value] = [141, 25, 43]
# colors_rgb[Colors_Name.amethyst.value] = [102, 0, 153]
# colors_rgb[Colors_Name.anburn.value] = [128, 0, 0]
# colors_rgb[Colors_Name.navy.value] = [0, 0, 128]
colors_rgb[Colors_Name.navy.value] = [63, 77, 112]
colors_rgb[Colors_Name.black.value] = [31,35,36]
#===============================================================#
'''
# https://encycolorpedia.kr/named 참고
colors_rgb = np.zeros([color_num, 3], dtype = int)
colors_rgb[Colors_Name.white.value] = [255, 255, 255]
colors_rgb[Colors_Name.UP_white.value] = [244, 238, 226] # 아이보리
colors_rgb[Colors_Name.DOWN_white.value] = [185, 173, 151] # 배이지
# colors_rgb[Colors_Name.beige.value] = [245, 245, 220]
# colors_rgb[Colors_Name.lime.value] = [191, 255, 0]
# colors_rgb[Colors_Name.ivory.value] = [236, 230, 204]
# colors_rgb[Colors_Name.forsythia.value] = [236, 230, 0]
colors_rgb[Colors_Name.yellow.value] = [255,232,12]
# colors_rgb[Colors_Name.apricot.value] = [251, 206, 177]
# colors_rgb[Colors_Name.turquoise.value] = [131, 220, 183]
# colors_rgb[Colors_Name.silver.value] = [192, 192, 192]
# colors_rgb[Colors_Name.tangerine.value] = [248, 155, 0]
# colors_rgb[Colors_Name.chartreuse.value] = [129, 193, 71]
# colors_rgb[Colors_Name.coral.value] = [242, 152, 134]
colors_rgb[Colors_Name.sky_blue.value] = [137,200,217]
colors_rgb[Colors_Name.UP_sky_blue.value] = [98,165,182]
colors_rgb[Colors_Name.DOWN_sky_blue.value] = [127,150,166]
# colors_rgb[Colors_Name.orange.value] = [255, 127, 0]
# colors_rgb[Colors_Name.light_blue.value] = [74, 168, 216]
# colors_rgb[Colors_Name.cyan.value] = [0, 163, 210]
# colors_rgb[Colors_Name.ocher.value] = [198, 138, 18]
# colors_rgb[Colors_Name.claret.value] = [255, 0, 255]
# colors_rgb[Colors_Name.pink.value] = [255, 51, 153]
# colors_rgb[Colors_Name.powder_blue.value] = [62, 145, 181]
colors_rgb[Colors_Name.green.value] = [138, 126, 93]
colors_rgb[Colors_Name.UP_green.value] = [112,103,83]
# colors_rgb[Colors_Name.DOWN_green.value] = [56,62, 62]
# colors_rgb[Colors_Name.seagrass.value] = [0, 128, 255]
# colors_rgb[Colors_Name.gray.value] = [128, 128, 128]
# colors_rgb[Colors_Name.light_purple.value] = [137, 119, 173]
colors_rgb[Colors_Name.red.value] = [194,41,59]
colors_rgb[Colors_Name.UP_red.value] = [162,31,47]
colors_rgb[Colors_Name.DOWN_red.value] = [102,13,27]
# colors_rgb[Colors_Name.olive.value] = [128, 128, 0]
# colors_rgb[Colors_Name.emerald_green.value] = [0, 141, 98]
# colors_rgb[Colors_Name.khaki.value] = [143, 120, 75]
# colors_rgb[Colors_Name.aqua_marine.value] = [94, 126, 155]
# colors_rgb[Colors_Name.dark_blue.value] = [0, 128, 128]
# colors_rgb[Colors_Name.magenta.value] = [220, 20, 60]
colors_rgb[Colors_Name.purple.value] = [88,61,164]
colors_rgb[Colors_Name.UP_purple.value] = [72,50,150]
colors_rgb[Colors_Name.DOWN_purple.value] = [50,32,114]
colors_rgb[Colors_Name.blue.value] = [47,90,193]
colors_rgb[Colors_Name.UP_blue.value] = [30,56,149]
colors_rgb[Colors_Name.DOWN_blue.value] = [14,29,96]
colors_rgb[Colors_Name.brown.value] = [155,98,77]
colors_rgb[Colors_Name.UP_brown.value] = [106,60,44]
colors_rgb[Colors_Name.DOWN_brown.value] = [67,21,8]
# colors_rgb[Colors_Name.blue_purple.value] = [105, 55, 161]
# colors_rgb[Colors_Name.blue_green.value] = [58, 77, 69]
# colors_rgb[Colors_Name.ultramarine.value] = [70, 73, 100]
# colors_rgb[Colors_Name.cobalt_blue.value] = [0, 73, 140]
# colors_rgb[Colors_Name.rose.value] = [141, 25, 43]
# colors_rgb[Colors_Name.amethyst.value] = [102, 0, 153]
# colors_rgb[Colors_Name.anburn.value] = [128, 0, 0]
# colors_rgb[Colors_Name.navy.value] = [0, 0, 128]
colors_rgb[Colors_Name.UP_navy.value] = [56,66,91]
colors_rgb[Colors_Name.DOWN_navy.value] = [43,50,68]
colors_rgb[Colors_Name.black.value] = [46,46,48]
colors_rgb[Colors_Name.UP_black.value] = [27,27,25]
colors_rgb[Colors_Name.DOWN_black.value] = [5,3,4]
#===============================================================#

class NodeLookup(object): # file
    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                model_dir, 'test.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                model_dir, 'test.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):

        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            
            uid = parsed_items[0]
            
            human_string = parsed_items[2]
           
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():

    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            model_dir, 'retrained_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def maybe_download_and_extract():
    # Download and extract model tar file
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename,
                 float(
                     count *
                     block_size) /
                    float(total_size) *
                    100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def youngJun(labels):
    kind = ['shirts','tshirts','hood','jacket','suit', 'dress']
    pattern = ['default','stripe','check','flower']
    returnList = []
    returnList2 = []
    for i in labels:
        if i in kind and len(returnList) == 0:
            returnList = i
        if i in pattern and len(returnList2) == 0:
            returnList2 = i
    return returnList, returnList2
    
def change_TrackbarValue(l_h, u_h, l_s, u_s, l_v, u_v):
    cv2.setTrackbarPos('lower_h','skin_hsv',l_h)
    cv2.setTrackbarPos('upper_h','skin_hsv',u_h)
    cv2.setTrackbarPos('lower_s','skin_hsv',l_s)
    cv2.setTrackbarPos('upper_s','skin_hsv',u_s)
    cv2.setTrackbarPos('lower_v','skin_hsv',l_v)
    cv2.setTrackbarPos('upper_v','skin_hsv',u_v)
# h-색,s-채,v-명 sv(클수록 진하고 밝음)

def nothing(x):
    pass


(l_skinh, u_skinh) = (0, 18)
(l_skins, u_skins) = (64, 255)
(l_skinv, u_skinv) = (26, 220)

cv2.namedWindow('skin_hsv')
cv2.resizeWindow('skin_hsv', 600,300)

cv2.createTrackbar('lower_h','skin_hsv',0,255,nothing)
cv2.createTrackbar('upper_h','skin_hsv',0,255,nothing)
cv2.createTrackbar('lower_s','skin_hsv',0,255,nothing)
cv2.createTrackbar('upper_s','skin_hsv',0,255,nothing)
cv2.createTrackbar('lower_v','skin_hsv',0,255,nothing)
cv2.createTrackbar('upper_v','skin_hsv',0,255,nothing)

change_TrackbarValue(l_skinh, u_skinh, l_skins, u_skins, l_skinv, u_skinv)

def mouse_callback(event, x, y, flags, param):
    global l_skinh, u_skinh, l_skins, u_skins, l_skinv, u_skinv
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        color = frame[y, x]
        one_pixel = np.uint8([[color]])
        hsv = cv2.cvtColor(one_pixel, cv2.COLOR_BGR2HSV)
        hsv = hsv[0][0]

    if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse left clicked: ", hsv)
        if u_skinh < hsv[0]:
            u_skinh = hsv[0]
        elif l_skinh > hsv[0]:
            l_skinh = hsv[0]
        if u_skins < hsv[1]:
            u_skins = hsv[1]
        elif l_skins > hsv[1]:
            l_skins = hsv[1]
        if u_skinv < hsv[2]:
            u_skinv = hsv[2]
        elif l_skinv > hsv[2]:
            l_skinv = hsv[2]
        change_TrackbarValue(l_skinh, u_skinh, l_skins, u_skins, l_skinv, u_skinv)

    if event == cv2.EVENT_RBUTTONDOWN:
        print("mouse write clicked: ", hsv)
        if u_skinh > hsv[0] and l_skinh < hsv[0]:
            if hsv[0] - l_skinh < u_skinh - hsv[0]:
                l_skinh = hsv[0]
            else:
                u_skinh = hsv[0]
        if u_skins > hsv[1] and l_skins < hsv[1]:
            if hsv[1] - l_skins < u_skins - hsv[1]:
                l_skins = hsv[1]
            else:
                u_skins = hsv[1]
        if u_skinv > hsv[2] and l_skinv < hsv[2]:
            if hsv[2] - l_skinv < u_skinv - hsv[2]:
                l_skinv = hsv[2]
            else:
                u_skinv = hsv[2]
        change_TrackbarValue(l_skinh, u_skinh, l_skins, u_skins, l_skinv, u_skinv)

cam_default = cam_num = 0
cap1 = cv2.VideoCapture(1 + cam_default)
cap2 = cv2.VideoCapture(1 + (cam_num^1))
def setup_change(x):
    brightness = cv2.getTrackbarPos('brightness', 'setup')
    contrast = cv2.getTrackbarPos('contrast', 'setup')
    saturation = cv2.getTrackbarPos('saturation', 'setup')
    exposure = -cv2.getTrackbarPos('exposure', 'setup')
    default_exposure = float(-cv2.getTrackbarPos('default_exposure', 'setup'))
    cap1.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap1.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap1.set(cv2.CAP_PROP_SATURATION, saturation)
    cap1.set(cv2.CAP_PROP_EXPOSURE, exposure)
    exposure = cap1.get(cv2.CAP_PROP_EXPOSURE)
    print('exposure', exposure)
    print('default_exposure', default_exposure)
    pass

def model():
    # frame3 = cv2.imread("/test.png")
    image_data = tf.gfile.FastGFile("./test.png", 'rb').read()
    predictions = sess.run(
        softmax_tensor, {
            'DecodeJpeg/contents:0': image_data})

    predictions = np.squeeze(predictions)
    node_lookup = NodeLookup()

    # change n_pred for more predictions
    n_pred = 8
    top_k = predictions.argsort()[-n_pred:][::-1]
    print(top_k)
    print("###")
    
    labels = []

    for node_id in top_k:
        human_string_n = node_lookup.id_to_string(node_id)
        
        print(human_string_n)
        print("여기다")
        score = predictions[node_id]
        print("score : {}".format(score))
        labels.append(human_string_n)
        
    '''
    if score > .3:
        # Some manual corrections
        # Kind of cheating
        # if human_string_n == "stethoscope":
        #     human_string_n = "Headphones"
        # if human_string_n == "spatula":
        #     human_string_n = "fork"
        # if human_string_n == "iPod":
        #     human_string_n = "iPhone"
        if human_string_n == "shirtsstripe":
            human_string_n = "shirtsstripe"
        if human_string_n == "hoodiecheck":
            human_string_n = "hoodiecheck"
        if human_string_n == "hoodiedefault":
            human_string_n = "hoodiedefault"
        if human_string_n == "tshirtsdefault":
            human_string_n = "tshirtsdefault"
        if human_string_n == "shirtsdefault":
            human_string_n = "shirtsdefault"
        if human_string_n == "jacketmilitary":
            human_string_n = "jacketmilitary"
        if human_string_n == "tshirtsstripe":
            human_string_n = "tshirtsstripe"
        if human_string_n == "hoodiestripe":
            human_string_n = "hoodiestripe"
        if human_string_n == "suitdefault":
            human_string_n = "suitdefault"
        human_string = human_string_n
        print(human_string)
        print("??????????")
        lst = human_string.split()
        print(labels)
        print("^^^^^^^^^^")
        human_string = " ".join(lst[0:2])
        #human_string_filename = str(lst[0])
    '''
    returnList1, returnList2 = youngJun(labels)
    
    print("라벨 : {}".format(labels))
    print("정답은 ?!!>?!?! {} 이거랑 {} 입니당".format(returnList1,returnList2))
    
    file_name = cloth_file_path + returnList1 + returnList2 + '.mp3'
    play = True
    
    '''
    current = time.time()
    fps = frame_count / (current - start)
    if last > 30 and pygame.mixer.music.get_busy(
    ) == False and human_string != human_string_n:
        pred += 1
        name = returnList1 + returnList2 + ".mp3"

        # Only get from google if we dont have it
        if not os.path.isfile(name):
            tts = gTTS(text="옷의 종류는" + returnList1 + "패턴은 " + returnList2 + "입니다.", lang='ko')
        #   tts.save(name)

        last = 0
        pygame.mixer.music.load(name)
        pygame.mixer.music.play()
    '''
    return play, file_name

brightness = cap1.get(cv2.CAP_PROP_BRIGHTNESS)
contrast = cap1.get(cv2.CAP_PROP_CONTRAST)
saturation = cap1.get(cv2.CAP_PROP_SATURATION)
exposure = cap1.get(cv2.CAP_PROP_EXPOSURE)
temperature = cap1.get(cv2.CAP_PROP_TEMPERATURE)
print('brightness', brightness)
print('contrast', contrast)
print('saturation', saturation)
print('exposure', exposure)
print('temperature', temperature)

exposure = -10.0
default_exposure = -6.0
_, _ = cap1.read()
cap1.set(cv2.CAP_PROP_EXPOSURE, exposure)
time.sleep(0.25)
exposure = cap1.get(cv2.CAP_PROP_EXPOSURE)
print('exposure', exposure)

cv2.namedWindow('setup')

cv2.createTrackbar('brightness','setup',0,255,setup_change)
cv2.createTrackbar('contrast','setup',0,255,setup_change)
cv2.createTrackbar('saturation','setup',0,255,setup_change)
cv2.createTrackbar('exposure','setup',0,14,setup_change)
cv2.createTrackbar('default_exposure','setup',0,14,setup_change)

cv2.setTrackbarPos('brightness','setup',int(brightness))
cv2.setTrackbarPos('contrast','setup',int(contrast))
cv2.setTrackbarPos('saturation','setup',int(saturation))
cv2.setTrackbarPos('exposure','setup',int(-exposure))
cv2.setTrackbarPos('default_exposure','setup',int(-default_exposure))

color_detect_rect = np.zeros(4)
color_diff = np.zeros([color_num], dtype = int)
prev_rect_x = 0
prev_notice = 0

frame_write_interval = 0
cam_change_interval = 0
cam_save_interval = 0
temperature_check_interval = 0

# exeCode = "python classify.py --model fashion.model --labelbin mlb.pickle --image test.png"

model_dir = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_callback)

# Download and create graph
maybe_download_and_extract()
create_graph()

cloth_mp3_play = False
cloth_file_path = 'voice/cloth/'

notice_color_mp3_play = False    
notices_file_path = 'voice/notices/'
colors_file_path = 'voice/colors/'

# Variables declarations
frame_count = 0
score = 0
start = time.time()
pred = 0
last = 0
human_string = None
returnList1 = []
returnList2 = []
pastValue1 = 0
pastValue2 = 0

pygame.mixer.init()
pygame.mixer.music.load("{}7.mp3".format(notices_file_path))
pygame.mixer.music.play()

with tf.Session() as sess:
    #softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    while True:
        try:  #an error comes if it does not find anything in window as it cannot find contour of max area
            #therefore this try error statement
            # cap1.set(cv2.CAP_PROP_TEMPERATURE, 6500)
            
            _, frame1 = cap1.read()
            # frame2 = frame1.copy()
            _, frame2 = cap2.read()
            
            temperature_check_interval += 1
            if temperature_check_interval >= 40:
                temperature_check_interval = 0
                temperature = cap1.get(cv2.CAP_PROP_TEMPERATURE)
                if temperature != 6500:
                    print('temperature', temperature)
                    cap1.set(cv2.CAP_PROP_TEMPERATURE, 6500)
            
            if cam_num == cam_default:
                frame = frame1
            else:
                frame = frame2
            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            
            if cam_num == cam_default:
                #손인식을 할 범위의 사이즈 (100,100),(400,400) 사각형의 모서리
                #roi = frame[100:400, 100:400]
                roi = frame
                
                #roi 범위 안의 색영역추출
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # 추출한 색영역과 비교할 범위 (살색) [색범위,채도,명암]
                l_skinh = cv2.getTrackbarPos('lower_h', 'skin_hsv')
                u_skinh = cv2.getTrackbarPos('upper_h', 'skin_hsv')
                l_skins = cv2.getTrackbarPos('lower_s', 'skin_hsv')
                u_skins = cv2.getTrackbarPos('upper_s', 'skin_hsv')
                l_skinv = cv2.getTrackbarPos('lower_v', 'skin_hsv')
                u_skinv = cv2.getTrackbarPos('upper_v', 'skin_hsv')

                lower_skin = np.array([l_skinh, l_skins, l_skinv], dtype = np.uint8)
                upper_skin = np.array([u_skinh, u_skins, u_skinv], dtype = np.uint8)

                # 추출한 색영역 hsv가 살색 범위만 남긴다. (0 or 255)
                mask = cv2.inRange(hsv, lower_skin, upper_skin)
                cv2.imshow("Skin color detection", mask)

                #cv2.GaussianBlur 중심에 있는 픽셀에 높은 가중치 = 노이즈제거 (0 ~ 255)
                mask = cv2.GaussianBlur(mask, (21,21), 0)
                mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]

                #외곽의 픽셀을 1(흰색)으로 채워 노이즈제거 interations -반복횟수
                kernel = np.ones((3, 3), np.uint8) 
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                #cv2.findContours 경계선 찾기 cv2.RETR_TREE 경계선 찾으며 계층관계 구성 cv2.CHAIN_APPROX_SIMPLE 경계선을 그릴 수 있는 point만 저장
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                #경계선 중 최대값 찾기
                cnt = max(contours, key = lambda x: cv2.contourArea(x))

                #엡실론 값에 따라 컨투어 포인트의 값을 줄인다. 각지게 만듬 Douglas-Peucker 알고리즘 이용
                epsilon = 0.0005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                M = cv2.moments(cnt)

                #중심점
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(roi, (cx, cy), 6, [255, 255, 0], -1)

                #외곽의 점을 잇는 컨벡스 홀
                hull = cv2.convexHull(cnt)
                cv2.drawContours(roi, [hull], 0, (0, 255, 0), 2)

                #컨벡스홀 면적과 외곽면적 정의
                areahull = cv2.contourArea(hull)
                areacnt = cv2.contourArea(cnt)

                #컨벡스홀-외곽면적의 비율
                arearatio = ((areahull - areacnt) / areacnt) * 100

                # 깊이의 개수
                l = 0
                notice = 0
                detect_cam_change_finger = False
                detect_pointing_finger = False
                
                if areacnt < 1500:
                    #"좀 더 안쪽을 가리켜주세요" 명령 추가
                    if areacnt > 500:
                        notice = 1
                    cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    #cv2.convexityDefects 컨벡스 결함
                    hull = cv2.convexHull(approx, returnPoints = False)

                    defects = cv2.convexityDefects(approx, hull)

                    #시작점, 끝점, 결점을 정한다
                    for i in range(defects.shape[0]): #defects 컨벡스 결함의 수 만큼 반복
                        s, e, f, d = defects[i, 0]
                        start = tuple(approx[s][0])
                        end = tuple(approx[e][0])
                        far = tuple(approx[f][0])

                        # end,for,start 점의 삼각형 길이
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        s = (a + b + c) / 2
                        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                        #컨벡스 결함으로 이루어진 삼각형에서의 깊이
                        d = (2 * ar) / a

                        # 코사인 법칙을 이용한 손가락 사이 각도 **2 = ^2
                        angle = math.acos((b**2 + c**2 - a**2)/(2 * b * c)) * 57
                    
                        # 각도와 깊이를 확인해 far end start 각점에 표시
                        if angle <= 90 and d > 50:
                            l += 1
                            cv2.circle(roi, far, 6, [255, 0, 0], -1)
                            cv2.circle(roi, end, 6, [255, 0, 0], 1)
                            cv2.circle(roi, start, 6, [0, 0, 255], 1)

                        #컨벡스홀 라인그리기 start-end로 각각 
                        #cv2.line(roi, start, end, [0, 255, 0], 2)

                    #display corresponding gestures which are in their ranges
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if l == 0:
                        if arearatio < 12:
                            notice = 2
                            cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        else:   
                            detect_pointing_finger = True
                            cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                            
                            length_from_center = np.zeros(len(approx), dtype=int)
                            for i in range(len(approx)):
                                length_from_center[i] = math.sqrt((approx[i][0][0] - cx)**2 + (approx[i][0][1] - cy)**2)
                            farthest_point = np.argmax(length_from_center)
                            
                            topmost = approx[farthest_point][0]
                            color_detect_rect = [topmost[0] - 20, topmost[1] - 75, topmost[0] + 20, topmost[1] - 5]
                            cv2.circle(roi, (topmost[0], topmost[1] - 2), 4, [0, 100, 100], -1)
                            cv2.rectangle(roi, (color_detect_rect[0], color_detect_rect[1]), (color_detect_rect[2], color_detect_rect[3]), (0, 255, 0), 0)
                    elif l == 1:  #손가락 2개일 때 캠 전환
                        detect_cam_change_finger = True
                        cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    else :
                        notice = 3
                        cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

                if detect_cam_change_finger == True:
                    cam_change_interval += 1
                    if cam_change_interval >= 20:
                        cam_change_interval = 0
                        cam_num ^= 1
                        print('cam_num = ',cam_num)
                        notice = 4
                        notice_interval = 20
                else:
                    cam_change_interval = 0
                
                if abs(prev_rect_x - color_detect_rect[2]) > 6:
                    print (abs(prev_rect_x - color_detect_rect[2]))
                    frame_write_interval = 0
                    detect_pointing_finger = False
                prev_rect_x = color_detect_rect[2]            
                if detect_pointing_finger == True:
                    frame_write_interval += 1
                    if frame_write_interval >= 10:
                        frame_write_interval = 0
                        
                        _, _ = cap1.read()
                        cap1.set(cv2.CAP_PROP_EXPOSURE, -cv2.getTrackbarPos('default_exposure', 'setup'))
                        print('default_exposure', cap1.get(cv2.CAP_PROP_EXPOSURE))
                        time.sleep(0.25)
                        _, color_frame = cap1.read()
                        color_frame = cv2.flip(color_frame, 1)
                        cap1.set(cv2.CAP_PROP_EXPOSURE, -cv2.getTrackbarPos('exposure', 'setup'))
                        print('exposure', cap1.get(cv2.CAP_PROP_EXPOSURE))
                        time.sleep(0.25)
                        
                        roi2 = color_frame[color_detect_rect[1]:color_detect_rect[3], color_detect_rect[0]:color_detect_rect[2]]
                        cv2.imshow('roi2', roi2)
                        # roi2 = color_frame
                        rgbroi = cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB)
                        rgbroi = rgbroi.reshape((-1, 3))
                        rgbroi = np.float32(rgbroi)
                        
                        #K-MEANS
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                        k = 2
                        ret,label,center = cv2.kmeans(rgbroi,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                        print('center', center)
                        
                        center = np.uint8(center)
                        res = center[label.flatten()]
                        res2 = res.reshape((roi2.shape))
                        px = center[0]
                        print('color', px)
                        
                        for i in Colors_Name:
                            color_diff[i.value] = abs(px[0] - colors_rgb[i.value][0])
                            color_diff[i.value] += abs(px[1] - colors_rgb[i.value][1])
                            color_diff[i.value] += abs(px[2] - colors_rgb[i.value][2])
                        color = color_diff.argmin()

                        if cloth_mp3_play == False:
                            mp3_file_name = colors_file_path + Colors_Name(color).name + '.mp3'
                            notice_color_mp3_play = True
                        cv2.imshow('color_frame', color_frame)
                else:
                    frame_write_interval = 0
            else:
                cam_change_interval = 0
                frame_write_interval = 0
                
                print(cam_save_interval)
                cam_save_interval += 1
                if cam_save_interval >= 50:
                    # 사진찍을게요
                    cam_num ^= 1
                    notice = 6
                    notice_interval = 20
                    cv2.imwrite('test.png', raw_frame)
                    cam_save_interval = 0
                    print('initial cam_save : ',cam_save_interval)
                    cloth_mp3_play, mp3_file_name = model()
                
            if notice != prev_notice and cloth_mp3_play == False:
                notice_interval += 1
            else:
                notice_interval = 0
            if notice_interval >= 20:
                notice_interval = 0
                prev_notice = notice
                if notice > 0:
                    mp3_file_name = notices_file_path + str(notice) + '.mp3'
                    notice_color_mp3_play = True
            '''
            if os.path.exists("test.png"):
                execute()
            '''
            if (notice_color_mp3_play == True or cloth_mp3_play == True) and pygame.mixer.music.get_busy() == False:
                notice_color_mp3_play = False
                cloth_mp3_play = False
                print(mp3_file_name)
                pygame.mixer.music.load(mp3_file_name)
                pygame.mixer.music.play()
            
            cv2.imshow('mask', mask)
                
        except:
            pass
                
        cv2.imshow('frame', frame)
            
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
sess.close()
cv2.destroyAllWindows()
cap1.release()
cap2.release()