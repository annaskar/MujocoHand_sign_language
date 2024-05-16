import csv
import numpy as np

ONE_HOT_SIGNS = {
    'rest': np.float32([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    'one': np.float32([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    'two': np.float32([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    'three': np.float32([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    'four': np.float32([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    'five': np.float32([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    'six': np.float32([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    'seven': np.float32([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    'eight': np.float32([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    'nine': np.float32([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
}

ONE_HOT_ORDERS = {
    0: np.float32([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    1: np.float32([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    2: np.float32([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    3: np.float32([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    4: np.float32([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    5: np.float32([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    6: np.float32([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    7: np.float32([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    8: np.float32([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    9: np.float32([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
}

NUM_ACTUATORS = 20 #total of hand controlers



# Reads dataset x, y with x --> (sign, order) y --> (ctrl)
def read_dataset(dataset_filepath: str, one_hot: bool) -> (dict[str, np.ndarray], np.ndarray):
    signs = []
    orders = []
    ctrls = []

    with open(dataset_filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)

        for row in reader:
            sign = row[0]
            order = int(row[1])

            if one_hot:
                sign = ONE_HOT_SIGNS[sign]
                order = ONE_HOT_ORDERS[order]

            ctrl = [float(pos) for pos in row[2:]]

            signs.append(sign)
            orders.append(order)
            ctrls.append(ctrl)
    x = {'sign': np.array(signs), 'order': np.float32(orders)}
    y = np.float32(ctrls)
    return x, y
