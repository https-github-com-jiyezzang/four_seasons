#!/usr/bin/env python
# coding: utf-8

# # (1) 라이브러리 설치 및 함수 정의

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm
import os
from skimage import color


# In[3]:


# LAB 기준값 -> 웜/쿨 분류
criteria_lab = {
    'warm': np.array([63.45, 11.80, 17.11]),
    'cool': np.array([64.80, 10.87, 15.04])
}

## RGB 기준값 -> 봄 여름이 많이 나옴(가을1, 겨울0)
criteria_rgb = {
    'spring': np.array([252.75, 203.5, 152]),
    'summer': np.array([254, 229.25, 164.5]),
    'fall': np.array([240.75, 199.25, 132.75]),
    'winter': np.array([240, 201.5, 129.5])
}


# In[ ]:


# 코사인 유사도 함수
def compute_cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 퍼스널 컬러 분류 함수
def classify_skin_type(pixel):
    pixel_lab = color.rgb2lab(np.array([pixel])/255.0)
    pixel_rgb = pixel
    similarities_lab = np.array([compute_cosine_sim(pixel_lab[0], tone) for tone in criteria_lab.values()])
    primary_class = ['warm', 'cool'][np.argmax(similarities_lab)]

    if primary_class == 'warm':
        secondary_class = ['spring', 'fall'][np.argmax([compute_cosine_sim(pixel_rgb, criteria_rgb['spring']),
                                                        compute_cosine_sim(pixel_rgb, criteria_rgb['fall'])])]
    else:
        secondary_class = ['summer', 'winter'][np.argmax([compute_cosine_sim(pixel_rgb, criteria_rgb['summer']),
                                                          compute_cosine_sim(pixel_rgb, criteria_rgb['winter'])])]
    return primary_class, secondary_class


# In[10]:


# color map 정의
color_map_primary = {
    "warm": [255, 0, 0], ## 빨간색
    "cool": [0, 0, 255]  ## 파란색
}

color_map_secondary = {
    "spring": [255, 255, 0], ## 노란색
    "summer": [0, 255, 0],  ## 연두색
    "fall": [255, 127, 0],  ## 주황색
    "winter": [0, 255, 255] ## 하늘색
}


# In[ ]:


# LAB + RGB 4계절 분류 함수 정의

def skin_classification(directory, file_select):
  files = [f for f in os.listdir(directory) if f.endswith(file_select)]
  # files = files[:10]

  for file in files:
      file_path = os.path.join(directory, file)

      # Load and resize the image
      image = Image.open(file_path).resize((224, 224))
      image_np = np.array(image)

      height, width, _ = image_np.shape
      results_primary = np.zeros_like(image_np, dtype='uint8')
      results_secondary = np.zeros_like(image_np, dtype='uint8')

      for j in tqdm(range(height)):
          for i in range(width):
              current_pixel = image_np[j, i]
              primary_class, secondary_class = classify_skin_type(current_pixel)
              results_primary[j, i] = color_map_primary[primary_class]
              results_secondary[j, i] = color_map_secondary[secondary_class]

      # Create a new figure
      fig, ax = plt.subplots(1, 3, figsize=(15, 5))

      # Add the original image to the first subplot
      ax[0].imshow(image_np)
      ax[0].set_title('Original')

      # Add the primary classified image to the second subplot
      ax[1].imshow(results_primary)
      ax[1].set_title('Primary Classification')

      # Add the secondary classified image to the third subplot
      ax[2].imshow(results_secondary)
      ax[2].set_title('Secondary Classification')

      # Display the plot
      plt.show()


# # (2) 얼굴 영역의 LAB + RGB 값을 이용한 레이블링
# 
# - 얼굴만 인식해서, 얼굴 전체의 픽셀 별 클래스 분류 최빈값으로 대표 색상 정하기

# ## MTCNN을 이용한 레이블 분류
# 
# - MTCNN은 다른 얼굴 감지 방법보다 약간 느릴 수 있지만 정확도 측면에서 매우 잘 수행되는 경향

# In[6]:


get_ipython().system('pip install mtcnn')


# In[7]:


from mtcnn import MTCNN
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import collections
import cv2
import os
import shutil
from collections import Counter


# In[ ]:


detector = MTCNN()

def classify_images_file(directory, file_select):
    result_labels = []

    files = [f for f in os.listdir(directory) if f.endswith(file_select)]
    for file in files:
        file_path = os.path.join(directory, file)

        # Load and resize the image
        image = Image.open(file_path).resize((224, 224))
        image_np = np.array(image)

        # Detect faces in the image
        faces = detector.detect_faces(image_np)

        # Keep track of the labels
        labels_primary = []

        for face in faces:
            # Get the bounding box of the face
            x, y, width, height = face['box']

            # Loop over the pixels in the face region
            for j in range(y, y + height):
                for i in range(x, x + width):
                    # Ensure the pixel coordinates are within the bounds of the image
                    if j < 0 or i < 0 or j >= image_np.shape[0] or i >= image_np.shape[1]:
                        continue

                    current_pixel = image_np[j, i]
                    primary_class, _ = classify_skin_type(current_pixel)

                    # Record the labels
                    labels_primary.append(primary_class)

        # Get the most common label
        primary_label = collections.Counter(labels_primary).most_common(1)[0][0]

        result_labels.append(primary_label)

        # Create a new subfolder if it doesn't exist
        new_folder_path = os.path.join(directory, primary_label)
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)

        # Move the file to the new subfolder
        new_file_path = os.path.join(new_folder_path, file)
        shutil.move(file_path, new_file_path)

    return result_labels


# In[ ]:


directory = '/content/drive/MyDrive/Colab Notebooks/AIFFEL/Four_seasons/사계절_연예인 이미지 데이터셋/train data/'
result = classify_images(directory, '05.JPG')


# In[ ]:


print(len(result))


# In[ ]:


counter = Counter(result)

print(counter['warm'])
print(counter['cool'])

