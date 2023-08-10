import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import operator
import skimage
import json

from PIL import Image
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from skimage import color
from matplotlib import font_manager


detector = MTCNN()

def crop_face(image_path):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    result = detector.detect_faces(image_np)
    if len(result) == 0:
        return None
    x1, y1, width, height = result[0]['box']
    x2, y2 = x1 + width, y1 + height
    cropped_face = image_np[y1:y2, x1:x2]
    resized_face = cv2.resize(cropped_face, (224, 224))
    return resized_face

# 코사인 유사도 함수 / 픽셀당 계산
def compute_cosine_similarity(face1, face2):
    return cosine_similarity(face1.reshape(1, -1), face2.reshape(1, -1))

#json저장 파일 load
def json_images(json_filepath):
    with open(json_filepath, 'r') as f:
        preprocessed_images = json.load(f)
        preprocessed_images = {k: np.array(v) for k, v in preprocessed_images.items()}
    return preprocessed_images

def find_most_similar_images(input_image_path, preprocessed_images):
    input_face = crop_face(input_image_path)
    if input_face is None:
        return []
    input_face = input_face / 255.0
    input_face = color.rgb2lab(input_face)

    similarity_scores = {}
    for filename, img in preprocessed_images.items():
        if img.shape != input_face.shape:
            continue
        similarity_score = compute_cosine_similarity(input_face.flatten(), img.flatten())
        similarity_scores[filename] = similarity_score[0][0]

    sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_images = [(filename, score) for filename, score in sorted_scores[:3]]
    
    # for filename in preprocessed_images:
    #     preprocessed_images[filename] = np.array(preprocessed_images[filename])

    return top_images
    
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=16)

def visualize_top_images(top_images, image_directory, inputname):
    plt.figure(figsize=(10, 6))
    sorted_top_images = sorted(top_images, key=lambda x: x[1], reverse=True)
    for i, (filename, _) in enumerate(sorted_top_images):
        plt.subplot(1, len(sorted_top_images), i+1)
        img = Image.open(os.path.join(image_directory, filename))
        plt.imshow(img)
        plt.title(f"{filename[:-4]}", fontproperties=font_prop)
        plt.axis("off")
    plt.tight_layout()
    save_path = '/home/jiyezzangg/project/static/' + inputname[:-4] + '_zz.png'
    # plt.show()
    plt.savefig(save_path)

    
def fashion_recommendation(top_images, target_directory, inputname):
    plt.figure(figsize=(10, 6))
    
    for i, (filename, _) in enumerate(top_images):
        target_path = os.path.join(target_directory, filename)
        
        if os.path.exists(target_path):
            img = Image.open(target_path)
            plt.subplot(1, len(top_images), i+1)
            plt.imshow(img)
            plt.title(f"{filename[:-4]}", fontproperties=font_prop)
            plt.axis("off")    
    plt.tight_layout()
    save_path = '/home/jiyezzangg/project/static/' + inputname[:-4] + '_kk.png'
    # plt.show()
    plt.savefig(save_path)
