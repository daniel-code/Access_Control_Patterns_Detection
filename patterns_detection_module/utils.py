import os


def check_path(filename):
    """
    Check file path, if file path not exist, create file path recurrently

    :param filename:  file path
    :return: None
    """
    index = filename.rfind('/')
    if index != -1 and not os.path.exists(filename[:index]):
        os.makedirs(filename[:index])


def convert_floor(floor: str) -> int:
    """
    Convert floor into int

    :param floor: floor string from access record
    :return: floor int code

    """
    floor_table = {
        'B2': -2,
        'B1': -1,
        'BF': -1,
        '07A': 7,
        '07B': 7,
        'RF': 100
    }
    if floor in floor_table:
        return floor_table[floor]
    else:
        try:
            outputcode = int(floor)
            return outputcode
        except ValueError as e:
            print(e)
        return 0


def convert_IOcode(IOcode: str) -> int:
    """
    Convert IO code into int

    :param IOcode: IO string code
    :return: IO int code
    """
    IO = 0
    if IOcode[0] == 'I':
        IO = 1
    return IO
