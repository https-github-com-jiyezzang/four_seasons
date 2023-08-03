from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import model

app = Flask(__name__)

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
        
        # Use the model to predict the warm and cool values
        warm, cool = model.predict(file_path)
        
        # Calculate warm and cool ratios
        # warm_ratio = warm / max_ratio *100
        # cool_ratio = cool / max_ratio *100
        
        # Determine recommended color
        if warm > cool:
            rec_color = 0
        else:
            rec_color = 1
    
        return render_template('after.html', warm=warm, cool=cool, warm_ratio=warm, cool_ratio=cool, rec_color=rec_color)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
