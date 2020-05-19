import pandas as pd

from patterns_detection_module.path import PathModel

if __name__ == '__main__':
    FILE_PATH = 'model.pkl'
    TEST_BATCH = 100

    # building nums
    building_num = 13
    # gates code table
    gates_code_table = pd.read_csv('test_data/gate_code.csv').set_index('gate').to_dict()['id']
    # read access control data
    raw_data = pd.read_csv('test_data/test_data1.csv', dtype={'building': str, 'floor': str, 'IO': str})
    raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
    # create path model
    path_model = PathModel()
    x, y = path_model.data_encoding(raw_data=raw_data, building_num=building_num, gates_code_table=gates_code_table)
    # split training and test data
    x_raw_data, y_raw_data = raw_data.iloc[:-TEST_BATCH], raw_data.iloc[-TEST_BATCH:]
    x_train, x_test = x[:-TEST_BATCH], x[-TEST_BATCH:]
    y_train, y_test = y[:-TEST_BATCH], y[-TEST_BATCH:]
    # train model
    path_model.fit(X=x_train, y=y_train)
    # save model
    path_model.save_model(filename=FILE_PATH)
    # load pre-trained model
    path_model.load_model(filename=FILE_PATH)
    # predict gate code
    y_pred = path_model.predict(X=x_test)
    # plot predict and truth gates
    path_model.plot_output(y_test, y_pred)
    # output wrong predict cases
    path_model.to_csv('output.csv', y_raw_data, y_test, y_pred)
    print('Accuracy = ', path_model.score(X=x_test[:-1], y=y_test[:-1]))
