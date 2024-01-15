import os

def rm_under_dir(path):
    for file_name in os.listdir(path):
        # construct full file path
        file = path + file_name
        if os.path.isfile(file):
            # print('Deleting file:', file)
            os.remove(file)