from gtts import gTTS
import os

try:
    sound_player_name = "wmplayer.exe"
    os.system("taskkill /f /im %s" % sound_player_name)
except:
    pass
os.chdir('voice')
os.chdir('cloth')

# file_name = "7"
# text = "검지만 펴서 옷을 가르키면 색상을, 손을 브이자를 만들어 보여주면 패턴과 종류를 알려드립니다."
returnList1 = "dress"
returnList2 = "flower"
file_name = returnList1 + returnList2
# text = "옷의 종류는" + returnList1 + ", 패턴은 " + "기본" + "입니다."
# text = "옷의 종류는" + returnList1 + ", 패턴은 " + "줄무늬" + "입니다."
# text = "옷의 종류는" + returnList1 + ", 패턴은 " + returnList2 + "입니다."
text = "옷의 종류는" + returnList1 + ", 패턴은 " + "꽃무늬" + "입니다."

tts = gTTS(text=text, lang='ko')
m_mpfile = file_name + ".mp3"
tts.save(m_mpfile)
os.system("start %s" % m_mpfile)
os.chdir('..')
os.chdir('..')