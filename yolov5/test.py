from flask import Flask, request, render_template, jsonify, Response
from werkzeug.utils import secure_filename #보안
from socket import *
from image_sizeto640 import resize_image #이미지 크기 640으로 전환
from PIL import Image #이미지 크기 
import sys, os
import boto3, json
from botocore.client import Config
from http import HTTPStatus
import argparse
import io
import torch
import PIL
import shutil
PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
models = {}

#sys.path.append('D:\test\flask\yolov5')
DETECTION_URL = "/v1/object-detection/<model>"

app = Flask(__name__)

# input = { "file_name" : "{id}_{YYYYMMDD}"}
# ex {"file_name" : "1_20240405"}
# output = { "detect" : "dectect 갯수" }
# ex {"detect" : "3"}

@app.route('/receive_img', methods=['POST'])
def receive_img():
    if request.method !="POST":
        return
    if request.files.get("image"):

        # 값이 image인 파일 읽어오기 
        file = request.files["image"]
        
        # filename 추출하기
        filename = secure_filename(file.filename)
        
        # file 로컬에 저장하기
        file.save(os.path.join("./source/", filename))

        # 이미지 경로 저장
        image =f"./source/{filename}"
        
        #이미지 사이즈 640으로 변경
        resize_image(image)
    
        weight_path = "./runs/train/before_addplastic/weights/best.pt"
        terminal_command2 = f"python3 detect.py --weights {weight_path} --img 640 --conf 0.3 --source {image}"
        os.system(terminal_command2)

        os.remove(image)

        # ./runs/detect/exp/ 안의 file_name.txt 파일을 불러오기
        # 파일 안 txt 파일을 확인하여 각 줄의 숫자를 카운트하여 count 변수에 저장
        detection_file_path = f"./runs/detect/exp/{filename[:-4]}.txt"
        if os.path.exists(detection_file_path):
             with open(detection_file_path, 'r') as f:

                # 파일의 줄 수를 세어 검출된 물체의 수 계산
                num_detected_objects = sum(1 for line in f)
                print(num_detected_objects)
                
                f.close()

                # 텍스트 파일 삭제
                shutil.rmtree(detection_file_path[:14])
                print(detection_file_path[:14])

                return jsonify({"count": num_detected_objects, "status": HTTPStatus.OK})
        return jsonify({"count": 0, "status" : HTTPStatus.BAD_REQUEST})
    return jsonify({"Message": "이미지를 첨부하세요", "status" : HTTPStatus.BAD_REQUEST})

@app.route('/test', methods=['GET'])
def test():
    return 'test'

@app.route('/send_file', methods=['POST'])
def send_file():
    if request.method !="POST":
        return
    if request.files.get("image"):

        # 값이 image인 파일 읽어오기 
        file = request.files["image"]
        
        # filename 추출하기
        filename = secure_filename(file.filename)
        
        # file 로컬에 저장하기
        file.save(os.path.join("./source/", filename))

        # 이미지 경로 저장
        image =f"./source/{filename}"
        
        #이미지 사이즈 640으로 변경
        resize_image(image)

        weight_path = "./runs/train/before_addplastic/weights/best.pt"

        terminal_command2 = f"python3 detect.py --weights {weight_path} --img 640 --conf 0.3 --source {image}"
        os.system(terminal_command2)
        os.remove(image)

        image_path = f"./runs/detect/exp/{filename}"

        # 이미지를 바이트 스트림으로 변환하여 클라이언트에게 반환
        with Image.open(image_path) as img:
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format=img.format)
            img_byte_array.seek(0)
        
        # 이미지 파일 및 디렉토리 삭제
        os.remove(image_path)  # 이미지 파일 삭제
        shutil.rmtree(os.path.dirname(image_path[:14]))  # 이미지가 있는 디렉토리 삭제
        
        # 바이트 스트림을 Response 객체에 담아 클라이언트에게 반환합니다.
        return Response(img_byte_array, mimetype='image/' + img.format.lower())
    return jsonify({"data" : "No Image", "status" : HTTPStatus.BAD_REQUEST})


# 0.0.0.0 으로 모든 IP에 대한 연결을 허용해놓고 포트는 8082로 설정
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port",default=5000, type=int)
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load("ultralytics/yolov5", 'custom', './runs/train/before_addplastic/weights/best.pt', force_reload=True, skip_validation=True)


    app.run(host="0.0.0.0", port=opt.port)
