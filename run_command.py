import os
for fname in ['0000.mp4','0005.mp4','0010.mp4','0011.mp4','0014.mp4','0026.mp4']:
    os.system(f'python main.py --device cpu --source ./video_demo/{fname} --draw --save-vid --hide-conf')