from pydub import AudioSegment
from pydub.utils import make_chunks
import os

audio_path = r'D:\dooraudio'
# 待处理文件上级目录

def mkdir(dir_name):
    main_path = r"D:\dooraudiosplit\\"
    # r 不能阻止\\的转义
    # 这里设置分割后的文件去的目录
    path = main_path+dir_name
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print
        "---  new folder...  ---"
        print
        "---  OK  ---"

    else:
        print
        "---  There is this folder!  ---"


def create_dir_from_audio_to_split(dirname):

    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):
        # subdir 该方法会继续进去子文件重复操作，直到没有子文件

        for filename in file_name_list:
            dir = filename.replace(".wav", "")
            print(dir)
            mkdir(dir)
            # 这里mkdir里面把要存储到的上级目录写了
            print("create successfully")
            result.append(dir)

    return result


def audio_split(sound_name):
    ## blues文件30s
    path = r"D:\dooraudio\\"+sound_name+".wav"
    # 这里设置待处理文件上级目录
    audio = AudioSegment.from_file(path, "wav")

    size = 5000  ## 切割的毫秒数

    chunks = make_chunks(audio, size)  ## 将文件切割为10s一块

    for i, chunk in enumerate(chunks):
        ## 枚举，i是索引，chunk是切割好的文件
        chunk_name = "{0}{1}.wav".format(sound_name, i)
        print(chunk_name)
        ## 保存文件
        to_path = r"D:\dooraudiosplit\\"+sound_name+r"\\"+"{0}".format(chunk_name)
        # 这里设置分割后的文件去的目录
        chunk.export(to_path, format="wav")


result = create_dir_from_audio_to_split(audio_path)
# "D:\audiosplit\\"下创造对应子文件 如"D:\audio\"下有待处理的"fans.wav" 我们能在"D:\audiosplit"下创造子文件夹"fans"

for sound_name in result:
    audio_split(sound_name)
# create_dir_from_audio_to_split反返回一个列表["buzzer_high", "buzzer_low", "fans", "pace"] 遍历"D:\audio\"下的".wav"文件分割到"D:\audiosplit"下对应的子文件夹中
