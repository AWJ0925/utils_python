# -----------------------------------------
# 视频的拼接：左右、上下
# Time: 2024/03/04
# -----------------------------------------
import os
from moviepy.editor import *

# video_path1 = "a-duibi_yinyong/23-10-23/orign/1-3_2023-10-30-14-08-21_re-crop.avi"  # L
# video_path2 = "a-duibi_yinyong/23-10-23/orign/1-3_192_0.5.avi"  # R

# OUTPUT_SRC = "a-duibi_yinyong/23-10-23/pt_onnx_1-3.mp4" 

# clip1 = VideoFileClip(video_path1)  # 读入视频
# clip2 = VideoFileClip(video_path2)
# final_clip = clips_array([[clip1, clip2]])  # 左右拼接
# # final_clip = clips_array([[clip1],[clip2]])  # 上下拼接

# final_clip.write_videofile(OUTPUT_SRC)

path_videos_left = os.path.join(r'G:\pythonProject\videos', '241105_fpn')  # 输出版本
path_videos_right = os.path.join(r'G:\pythonProject\videos', '241105_ppam')
path_res = os.path.join(r'G:\pythonProject\videos', 'fpn_ppam')

if not os.path.exists(path_res):
    os.mkdir(path_res)

# 循环拼接
for filename in os.listdir(path_videos_left):
    if filename.endswith('.avi'):
        
        left_src = os.path.join(path_videos_left, filename)  # 4 月份
        right_src = os.path.join(path_videos_right, filename)  # 当前月份

        OUTPUT_SRC = os.path.join(path_res, filename.replace('.avi', '.mp4'))

        clips = [VideoFileClip(left_src), VideoFileClip(right_src)]
        video = clips_array([clips])

        video.write_videofile(OUTPUT_SRC)

print('拼接视频完成！')

