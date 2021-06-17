import pickle
import json


def evaluate(csv_filename, model_filename, scalar_filename, results_filename):
	model = pickle.load(open(model_filename, 'rb'))
	scalar = pickle.load(open(scalar_filename, 'rb'))
	rows = get_rows(csv_filename)

	print('evaluating rows for {}'.format(model_filename))

	results = evaluate_rows(
		rows, 
		lambda row: model_evaluation(model, scalar, row))

	with open(results_filename, 'w') as f:
		f.write(json.dumps(results))


	print('calculating distance')
	avg_distance = get_avg_dist(results)

	print('Average distance: {}'.format(avg_distance))


def get_avg_dist(results):
	eval_dist = 0

	for row in results:
		actual = float(row[3])

		dist = abs(actual - float(row[4]))
		print(dist)

		eval_dist += dist

	return eval_dist / len(results)


def evaluate_rows(rows, eval1):
	for row in rows:
		row.append(eval1(row))
	return rows



def model_evaluation(model, scalar, row):
	scaled_inputs = scalar.transform([[row[2]]])
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
		'world_wide/ww_no_ll.csv',
		'world_wide/ww-sal-bagging-1598554805.250578.model',
		'world_wide/ww-sal-bagging-1598554803.692277.scalar',
		'world_wide/results-sal-bagging')
