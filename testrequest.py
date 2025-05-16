import requests
import json

train_data = []
arr = list(range(10))

for i in range(10):
	train_data += arr

history = train_data[5:15]

history_size = 10
answer_size = 2
min_value = min(arr)
max_value = max(arr)

variant = int(input('Вариант: '))

model_action = ''
model_name = ''
save = False

if(variant) == 0:
	model_action = 'create'
elif(variant) == 1:
	model_action = 'create'
	model_name = 'test.h5'
	save = True
elif(variant) == 2:
	model_action = 'load'
	model_name = 'test.h5'
	
data = {'model-action' : model_action,
		'model-name' : model_name,
		'train-data' : train_data,
		'history' : history,
		'history-size' : history_size,
		'answer-size' : answer_size,
		'min-value' : min_value,
		'max-value' : max_value,
		'save' : save
		}

jsondata = json.dumps(data)

response = requests.post('http://localhost:5000/demand-prediction', json=jsondata)

print(response.status_code)
print(response.text)
