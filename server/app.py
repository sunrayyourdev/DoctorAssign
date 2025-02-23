from flask import Flask, jsonify, request
from flask_cors import CORS  # To handle CORS (Cross-Origin Resource Sharing)

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Example data
data = [
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"},
]

# Define a route to fetch data
@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(data)

# Define a route to add data
@app.route('/api/data', methods=['POST'])
def add_data():
    new_item = request.json
    data.append(new_item)
    return jsonify(new_item), 201

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)