import json
import numpy as np
from flask import Flask, request, jsonify
import testmodel as mt

app = Flask(__name__)


@app.route("/demand-prediction", methods=['POST'])
def predict_demand():
	if request.is_json:
	
		model = None
		result = []
	
		print(request.get_json())
	
		request_data = json.loads(request.get_json())
		model_action = request_data['model-action']
		
		if model_action == 'create':
		
			model = mt.create_new_model(np.array(request_data['train-data']),
				request_data['history-size'],
				request_data['answer-size'],
				request_data['min-value'],
				request_data['max-value'])
				
			if request_data['save']:
				mt.save_model(model, request_data['model-name'])
				
		elif model_action == 'load':
			
			model = mt.load_model(request_data['model-name'])
			
		if model != None:

			history = np.array(request_data['history'])
			min = request_data['min-value']
			max = request_data['max-value']
			result = mt.predict_result(model, history, min, max)
			result = result[0].tolist()
			
	return jsonify(result)
	
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)
