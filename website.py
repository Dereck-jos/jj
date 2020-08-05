from flask import Flask, request, jsonify
import traceback
import pandas as pd
import pickle
import flask

#app definition
app = Flask(__name__)

#loading the model
with open('finalized_model.sav', 'rb') as f:
    model = pickle.load(f)

#loading one hot encoder
with open('ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

#loading the column of the data
with open('le.pkl', 'rb') as f:
    le = pickle.load(f)    

@app.route('/', methods=['POST','GET'])
def predict():
    if flask.request.method == 'GET':
        return "Welcome to prediction page"

    if flask.request.method == 'POST':
        try:
            json_ = request.json
            print(json_)
            query_ = ohe.transform(pd.DataFrame(json_))
            # query = query_.reindex(columns = model_columns, fill_value= 0)
            prediction = list(le.inverse_transform(model.predict(query_)))
 
            return jsonify({
                "prediction":str(prediction)
            })

        except:
            return jsonify({
                "trace" : request.json
            })

if __name__ == "__main__":
   app.run(debug=True)
