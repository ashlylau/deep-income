import os


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Constants():
    def __init__(self):
        pass