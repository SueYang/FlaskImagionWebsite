import sys
import os

sys.path.insert(1, os.path.join(os.path.abspath("."), './venv/Lib/site-packages'))

from flask import Flask, render_template, request, url_for, send_from_directory, json
from werkzeug.utils import secure_filename

import keras


# for local server testing
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploaded')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# To show directory folder, enable below two lines. For debugging
# from flask_autoindex import AutoIndex
#AutoIndex(app, browse_root=os.path.curdir)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/picture_single')
def picture_single():
    return render_template('portfolio-single.html')

@app.route('/upload_picture', methods=['GET', 'POST'])
def upload_picture():
    if request.method == 'POST':
        file = request.files['uploadimgs']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # print(os.path.join(app.config['UPLOAD_FOLDER'], filename))   #debug in console
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
    return render_template('upload-picture.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/blog_single')
def blog_single():
    return render_template('blog-single.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

if __name__ == '__main__':
    app.run()

