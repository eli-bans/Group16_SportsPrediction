# import necessary dependencies
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
scaler = pickle.load(open('../new_scaler_parameters.pkl', 'rb'))
model = pickle.load(open('../new_player_ratings_model.pkl', 'rb'))


# Define a route for the homepage
@app.route('/')
def home():
    # Get the prediction result and confidence from the query parameters
    team = request.args.get('team', '')
    prediction_result = request.args.get('prediction_result', '')
    actual_confidence = request.args.get('actual_confidence', '')

    # Render the home page with the prediction result and confidence
    return render_template('index.html', team=team, prediction_result=prediction_result,
                           actual_confidence=actual_confidence)



# Define a route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get user input from the form
        potential = float(request.form['potential'])
        value_eur = float(request.form['value_eur'])
        wage_eur = float(request.form['wage_eur'])
        release_clause_eur = float(request.form['release_clause_eur'])
        passing = float(request.form['passing'])
        dribbling = float(request.form['dribbling'])
        attacking_short_passing = float(request.form['attacking_short_passing'])
        movement_reactions = float(request.form['movement_reactions'])
        power_shot_power = float(request.form['power_shot_power'])
        mentality_vision = float(request.form['mentality_vision'])
        mentality_composure = float(request.form['mentality_composure'])

        # Make predictions using the loaded model
        input_data = pd.DataFrame(data=[[potential, value_eur, wage_eur, release_clause_eur, passing, dribbling,
                                         attacking_short_passing, movement_reactions, power_shot_power,
                                         mentality_vision,
                                         mentality_composure]],
                                  columns=['potential', 'value_eur', 'wage_eur', 'release_clause_eur', 'passing',
                                           'dribbling', 'attacking_short_passing', 'movement_reactions',
                                           'power_shot_power',
                                           'mentality_vision', 'mentality_composure'])


        # Calculate prediction intervals using bootstrapping
        num_samples = 5
        predictions = []
        for i in range(num_samples):
            scaled_input = scaler.transform(input_data)
            result = model.predict(scaled_input)
            predictions.append(result[0])

        # The lower and upper bounds represent a range within which you
        # can be confident that the true prediction falls.
        lower_bound = np.percentile(predictions, 2.5)
        upper_bound = np.percentile(predictions, 97.5)


        response = {
            'prediction': float(result[0]),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
        alpha = response['upper_bound'] - response['lower_bound']
        actual_confidence = (1 - alpha)*100


        return redirect(url_for('home', prediction_result=result[0], actual_confidence=actual_confidence))

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
