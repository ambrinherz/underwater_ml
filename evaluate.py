import pickle
import json


def evaluate(csv_filename, model_filename, scalar_filename, results_filename):
	model = pickle.load(open(model_filename, 'rb'))
	scalar = pickle.load(open(scalar_filename, 'rb'))
	rows = get_rows(csv_filename)

	print('evaluating rows for {}'.format(model_filename))

	results = evaluate_rows(
		rows, 
		lambda row: model_evaluation(model, scalar, row),
		mackenzie_evaluation)

	with open(results_filename, 'w') as f:
		f.write(json.dumps(results))


	print('calculating distances')
	avg_distances = get_avg_dist(results)

	print(avg_distances)


def get_avg_dist(results):
	eval1_dist = 0
	eval2_dist = 0

	for row in results:
		actual = float(row[3])

		dist1 = abs(actual - float(row[4]))
		dist2 = abs(actual - float(row[5]))
		print('{} {}'.format(dist1, dist2))

		eval1_dist += dist1
		eval2_dist += dist2

	eval1_avg = eval1_dist / len(results)
	eval2_avg = eval2_dist / len(results)

	return eval1_avg, eval2_avg


# def leroy_evaluation(row):
# 	depth = float(row[0])
# 	temp = float(row[1])
# 	sal = float(row[2])

# 	results = 1402.5 + 5 * temp - .0544 * temp ** 2 + .00021 * temp ** 3 + 1.33 * sal - .0123 * sal * temp + .000087 * sal * temp ** 2 + .0156 * depth + .000000255 * depth ** 2 - .0000000000073 * depth ** 2 + .0000012 * depth * (lat - 45) - .00000000000095 * temp * depth ** 3 + .0000003 * temp ** 2 * depth + .0000143 * sal * depth
# 	print(result)
# 	return result


def mackenzie_evaluation(row):
	depth = float(row[0])
	temp = float(row[1])
	sal = float(row[2])

	result = 1448.96 + 4.591 * temp - .05304 * temp ** 2 + .0002374 * temp ** 3 + 1.34 * (sal - 35) + .0163 * depth + .0000001675 * depth ** 2 - .01025 * temp * (sal - 35) - .0000000000007139 * temp * depth ** 3	
	return result;


def evaluate_rows(rows, eval1, eval2):
	for row in rows:
		row.append(eval1(row))
		row.append(eval2(row))
	return rows


def model_evaluation(model, scalar, row):
	scaled_inputs = scalar.transform([row[:-1]])
	return model.predict(scaled_inputs)[0]


def get_rows(filename):
	rows = []

	with open(filename, 'r') as f:
		lines = f.readlines()[1:]

	for line in lines:
		line = line.replace('\n', '')
		rows.append(line.split(','))

	return rows
	

if __name__ == '__main__':
	evaluate(
		'arctic/arctic_combined.csv',
		'test/arctic-BaggingwithTree-300-1623809061.235102.model',
		'test/arctic-BaggingwithTree-300-1623808984.788518.scalar',
		'test/results/results-arctic-BaggingwithTree300')
