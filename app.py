from flask import Flask, render_template, request, redirect,url_for,jsonify
import pickle
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


""" @app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = LinearRegressionModel.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='House price should be $ {}'.format(output))
 """



def model_predict(img_path, model):
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    size = (8, 8)
    test_img = cv2.resize(src, size)

    test_img = test_img / test_img.max()

    test_img = test_img.reshape(1,-1)
    preds = model.predict(test_img)[0]    
    print("preds *************************",preds)
    preds = "The Number in the Image is : "+str(preds)
    return preds

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        print("result ********************",result)
        return result
    return None



@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('page404.html'), 404


@app.route('/predict',methods=['GET'])
def pred():
    return redirect(url_for('Home'))

if __name__=="__main__":
    app.run(debug=True)
