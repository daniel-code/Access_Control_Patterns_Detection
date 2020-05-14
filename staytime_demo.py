import pandas as pd

from patterns_detection_module.staytime import StayTimeModel

if __name__ == '__main__':
    FILE_PATH = 'model.h5'
    ENCODE_TYPE = 2

    # read access control data
    raw_data = pd.read_csv('test_data/test_data1.csv')
    raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
    # create staytime model
    staytime_model = StayTimeModel(encode_type=ENCODE_TYPE)
    # encode raw data into [previous staytime, staytime, time of day]
    encode_data, data = staytime_model.data_encoding(raw_data)
    # using modified K-mean clustering
    staytime_model.fit(data=encode_data)
    # save each group info.
    staytime_model.save_model(FILE_PATH)
    # load pre-trained group info.
    staytime_model.load_model(FILE_PATH)
    # predicted access control is out of patterns
    y_pred, labels = staytime_model.predict(data=encode_data)
    # access control pattern visualisation
    staytime_model.plot_output(encode_data, center_verbose=False)
    # output abnormal data to csv file
    staytime_model.to_csv('output.csv', data, encode_data, y_pred)
