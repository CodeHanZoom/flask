import os
from PIL import Image

def resize_image(input_imagefile_path):
    # 이미지 열기
    img = Image.open(input_imagefile_path)
    
    # 변환할 크기 설정
    target_size = (640, 640)
    
    # 이미지를 지정된 크기로 변환
    resized_img = img.resize(target_size, Image.ANTIALIAS)
    
    # 파일 이름에서 확장자 추출
    file_name, file_extension = os.path.splitext(input_imagefile_path)
    
    # 변환된 이미지를 저장
    output_file_path = f"{file_name}{file_extension}"
    resized_img.save(output_file_path)
    
    print(f"이미지를 {target_size} 크기로 변환하여 {output_file_path}에 저장하였습니다.")
