# Flask application???? ## Livia
import numpy as np
import pickle
from flask import Flask, request, render_template

app_final = Flask(__name__, template_folder='template')
final_model = pickle.load(open('final_model.pickle', 'rb'))


@app_final.route('/', methods=['GET'])
def home():
    return render_template('project.html')


@app_final.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = np.array([float(x) for x in request.form.values()])
    final_features = [np.array(int_features)]
    prediction = round(final_model.predict(final_features)[0])

    print(prediction)

    if prediction >= 0.5:
        return render_template("project.html", prediction_text='Yes, the drug is persistent'.format(prediction))
    else:
        return render_template("project.html", prediction_text='No, the drug is not persistent'.format(prediction))


if __name__ == "__main__":
    app_final.run(port=5000, debug=True)
