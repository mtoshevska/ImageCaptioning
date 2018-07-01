import os
import glob
from deeprnn.deeprnn import generate_deeprnn_captions
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

app.config['SECRET_KEY'] = '?\xbf,\xb4\x8d\xa3"<\x9c\xb0@\x0f5\xab,w\xee\x8d$0\x13\x8b83'


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        for filename in glob.glob(f"{app.config['UPLOADED_PHOTOS_DEST']}/*"):
            os.remove(filename)
        filename = photos.save(request.files['photo'])
        session['filename'] = filename
        file_path = '/' + app.config['UPLOADED_PHOTOS_DEST'] + '/' + filename
        session['file_path'] = file_path
        session['deeprnn_caption'] = 'caption ......'  # generate_deeprnn_captions('..' + file_path)
        return redirect('/results', code=302)
    return render_template('upload.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    return render_template('results.html')


@app.route('/return_image_path', methods=['GET'])
def return_image_path(filename):
    return app.config['UPLOADED_PHOTOS_DEST'] + filename


if __name__ == '__main__':
    app.run(debug=True)
