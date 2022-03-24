import detect_y5 as detect
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import json
from pathlib import Path
from utils.general import print_args

FILE = Path(__file__).resolve()

#테스트할 이미지의 파일 경로 설정
filepath = './test_input/B030811XX_00404.jpg'
with open(filepath, 'rb') as img:
    base64_string = base64.b64encode(img.read()) # 백엔드에서는 b64로 encode된 이미지를 준다고 가정

#Model Detect.py RUN
opt = detect.parse_opt()
opt.data = './data/kfood.yaml'
#weights 설정 (default = yolov5s.pt)
opt.weights = './runs/train/batch_16_0.004_epoch_50_v5x6/weights/best.pt'
#cpu 설정
opt.device = 'cpu'
detect.check_requirements(exclude=('tensorboard', 'thop'))
print_args(FILE.stem, opt)
output_dict = detect.run(**vars(opt), source=base64_string)
#image, bbox, class 확인 (image는 일부만 확인)
print(f"image : {output_dict['image'][0]} bbox : {output_dict['bbox']}, class : {output_dict['class']}")

#Save output image (bbox 그려져있는 image)
output_image = output_dict['image']
output_img_name = filepath.split('/')[-1][:-4]
weights_name = opt.weights.split('/')[3]
cv2.imwrite(f'./test_result/{weights_name}_{output_img_name}.jpg', output_image)

print('-------------end-------------')