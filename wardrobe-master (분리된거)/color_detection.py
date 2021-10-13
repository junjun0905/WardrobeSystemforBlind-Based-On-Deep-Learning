import os, sys
import decimal
import enum
import numpy as np
from playsound import playsound

#===============================================================#
# color reference                                               #
#===============================================================#
class Colors_Name(enum.Enum):
   # 하양 베이지 라임 아이보리 개나리 노랑 살구 옥 은 귤
    white = 0
    UP_white = enum.auto()
    DOWN_white = enum.auto()
    # beige = enum.auto()
    lime = enum.auto()
    # ivory = enum.auto()
    # forsythia = enum.auto()
    yellow = enum.auto()
    # apricot = enum.auto()
    # turquoise = enum.auto()
    # silver = enum.auto()
    # tangerine = enum.auto()
   # 연두 산호 하늘 주황 밝은파랑 시안 황토 자홍 분홍 담청 
    chartreuse = enum.auto()
    # coral = enum.auto()
    sky_blue = enum.auto()
    orange = enum.auto()
    # light_blue = enum.auto()
    cyan = enum.auto()
    ocher = enum.auto()
    # claret = enum.auto()
    pink = enum.auto()
    # powder_blue = enum.auto()
   # 초록 바다 회색 밝은보라 빨강 올리브 에메랄드그린 카키 아쿠아마린 암청
    green = enum.auto()
    UP_green = enum.auto()
    DOWN_green = enum.auto()
    seagrass = enum.auto()
    # gray = enum.auto()
    # light_purple = enum.auto()
    red = enum.auto()
    UP_red = enum.auto()
    DOWN_red = enum.auto()
    # olive = enum.auto()
    emerald_green = enum.auto()
    # khaki = enum.auto()
    # aqua_marine = enum.auto()
    # dark_blue = enum.auto()
   # 심홍 보라 파랑 갈색 청자 청록 군청 코발트블루 장미 자주
    magenta = enum.auto()
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
    blue_green = enum.auto()
    # ultramarine = enum.auto()
    # cobalt_blue = enum.auto()
    rose = enum.auto()
    amethyst = enum.auto()
   # 고동 남색 검정
    anburn = enum.auto()
    navy = enum.auto()
    UP_navy = enum.auto()
    DOWN_navy = enum.auto()
    black = enum.auto()
    UP_black = enum.auto()
    DOWN_black = enum.auto()

color_num = len(Colors_Name)

# https://encycolorpedia.kr/named 참고
colors_rgb = np.zeros([color_num, 3], dtype = int)
colors_rgb[Colors_Name.white.value] = [255, 255, 255]
colors_rgb[Colors_Name.white.value] = [244, 238, 226] # 아이보리
colors_rgb[Colors_Name.white.value] = [185, 173, 151] # 배이지
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
colors_rgb[Colors_Name.green.value] = [92, 126, 93]
colors_rgb[Colors_Name.green.value] = [74,103,83]
colors_rgb[Colors_Name.green.value] = [56,62, 62]
# colors_rgb[Colors_Name.seagrass.value] = [0, 128, 255]
# colors_rgb[Colors_Name.gray.value] = [128, 128, 128]
# colors_rgb[Colors_Name.light_purple.value] = [137, 119, 173]
colors_rgb[Colors_Name.red.value] = [194,41,59]
colors_rgb[Colors_Name.red.value] = [162,31,47]
colors_rgb[Colors_Name.red.value] = [102,13,27]
# colors_rgb[Colors_Name.olive.value] = [128, 128, 0]
# colors_rgb[Colors_Name.emerald_green.value] = [0, 141, 98]
# colors_rgb[Colors_Name.khaki.value] = [143, 120, 75]
# colors_rgb[Colors_Name.aqua_marine.value] = [94, 126, 155]
# colors_rgb[Colors_Name.dark_blue.value] = [0, 128, 128]
# colors_rgb[Colors_Name.magenta.value] = [220, 20, 60]
# colors_rgb[Colors_Name.purple.value] = [109,73,181]
colors_rgb[Colors_Name.purple.value] = [88,61,164]
colors_rgb[Colors_Name.UP_purple.value] = [72,50,150]
colors_rgb[Colors_Name.DOWN_purple.value] = [50,32,114]
colors_rgb[Colors_Name.blue.value] = [47,90,193]
colors_rgb[Colors_Name.blue.value] = [30,56,149]
colors_rgb[Colors_Name.blue.value] = [14,29,96]
colors_rgb[Colors_Name.brown.value] = [155,98,77]
colors_rgb[Colors_Name.brown.value] = [106,60,44]
colors_rgb[Colors_Name.brown.value] = [67,21,8]
# colors_rgb[Colors_Name.blue_purple.value] = [105, 55, 161]
# colors_rgb[Colors_Name.blue_green.value] = [58, 77, 69]
# colors_rgb[Colors_Name.ultramarine.value] = [70, 73, 100]
# colors_rgb[Colors_Name.cobalt_blue.value] = [0, 73, 140]
# colors_rgb[Colors_Name.rose.value] = [141, 25, 43]
# colors_rgb[Colors_Name.amethyst.value] = [102, 0, 153]
# colors_rgb[Colors_Name.anburn.value] = [128, 0, 0]
# colors_rgb[Colors_Name.navy.value] = [0, 0, 128]
# colors_rgb[Colors_Name.navy.value] = [64,78,113]
colors_rgb[Colors_Name.UP_navy.value] = [56,66,91]
colors_rgb[Colors_Name.DOWN_navy.value] = [43,50,68]
colors_rgb[Colors_Name.black.value] = [46,46,48]
colors_rgb[Colors_Name.black.value] = [27,27,25]
colors_rgb[Colors_Name.black.value] = [5,3,4]
#===============================================================#

os.chdir('voice')
APP_FOLDER = os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir('..')

colors_file_path = os.path.join(APP_FOLDER, 'colors')
notices_file_path = os.path.join(APP_FOLDER, 'notices')

filename = 'color_detection_play.txt'
color_diff = np.zeros([color_num], dtype = int)

while True:
    file = open(filename, 'r', encoding="utf8")
    text_rgb = file.read()
    file.close()
    mp3_play = False
    
    split_data = text_rgb.split(',')
	# 유효 데이터인지 확인하고 동작하기
    if len(split_data) >= 5 and split_data[4].isdigit() == 1 and decimal.Decimal(split_data[4]) == 1:
        if split_data[3].isdigit() == 1 and decimal.Decimal(split_data[3]) > 0:
            m_mpfile = split_data[3] + ".mp3"
            full_path = os.path.join(notices_file_path, m_mpfile)
            print(m_mpfile)
            mp3_play = True
        elif split_data[0].isdigit() == 1 and split_data[1].isdigit() == 1 and split_data[2].isdigit() == 1 :
            rgb = list(map(int, split_data[0:3]))
            if (rgb[0] >= 0 and rgb[0] <= 255 and
                rgb[1] >= 0 and rgb[1] <= 255 and
                rgb[2] >= 0 and rgb[2] <= 255):
                print(rgb)

                for i in Colors_Name:
                    color_diff[i.value] = abs(rgb[0] - colors_rgb[i.value][0])
                    color_diff[i.value] += abs(rgb[1] - colors_rgb[i.value][1])
                    color_diff[i.value] += abs(rgb[2] - colors_rgb[i.value][2])
                color = color_diff.argmin()

                m_mpfile = Colors_Name(color).name + ".mp3"
                full_path = os.path.join(colors_file_path, m_mpfile)
                print(m_mpfile)
                mp3_play = True
            else :
                print('wrong data')
        else :
            print('wrong data')
        
    if mp3_play == True:
        file = open(filename, 'w', encoding="utf8")
        file.write('0,0,0,0,0,\n')
        file.close()
    
        playsound(full_path)
