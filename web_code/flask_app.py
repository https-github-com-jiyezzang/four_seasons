from celebrity import find_most_similar_images, visualize_top_images, fashion_recommendation
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from matplotlib import font_manager
import os
import model
import json
import numpy as np

app = Flask(__name__)

json_path = '/home/jiyezzangg/project/recommendation/preprocessed_images.json'
if os.path.exists(json_path):
    with open(json_path, 'r') as json_file:
        preprocessed_images = json.load(json_file)
        preprocessed_images = {k: np.array(v) for k, v in preprocessed_images.items()}
        
@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join('/home/jiyezzangg/project/uploads', filename)
        file.save(file_path)
        image_rec = filename[:-4] + '_zz.png'
        image_rec_fas = filename[:-4] + '_kk.png'
        
        warm, cool = model.predict(file_path)
        
        if warm > cool:
            rec_color = 0
        else:
            rec_color = 1

        # top_images 리턴 받고 [(연예인 filename1, 거리값),(연예인 filename2, 거리값),(연예인 filename3, 거리값)]
        similar_images = find_most_similar_images(file_path, preprocessed_images)
        # ['/home/jiyezzangg/project/recommendation/celebrity/연예인 filename1', '/home/jiyezzangg/project/recommendation/celebrity/연예인 filename2', '/home/jiyezzangg/project/recommendation/celebrity/연예인 filename3']
        # top_image_paths = [os.path.join('/home/jiyezzangg/project/recommendation/celebrity', filename) for filename, _ in similar_images]
        visualize_top_images(similar_images, '/home/jiyezzangg/project/recommendation/celebrity', filename)
        fashion_recommendation(similar_images, '/home/jiyezzangg/project/recommendation/celebrity_fashion', filename)

        return render_template('after.html', warm=warm, cool=cool, warm_ratio=warm, cool_ratio=cool, rec_color=rec_color, image_rec=image_rec, image_rec_fas=image_rec_fas)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)