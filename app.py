from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

labels = ['Cat', 'Dog']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        file = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        result = model.predict(img)
        prediction = labels[np.argmax(result)]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
