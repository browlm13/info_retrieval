import json
import os
import logging
import sys

# external
import numpy as np
import pickle

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))

# read in config/directory_structure.json
# handle reading and writing of data

DIRECTORY_STRUCTURE_FILE_PATH = "config/directory_structure.json"   #from root


def load_directory_structure():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    head = os.path.commonprefix([script_dir, DIRECTORY_STRUCTURE_FILE_PATH])
    file_path = os.path.join(head, DIRECTORY_STRUCTURE_FILE_PATH)
    with open(file_path) as json_data:
        directory_structure_dict = json.load(json_data)

    return directory_structure_dict


def save(type, data, parameters_list, output_type='json'):
    directory_structure_dict = load_directory_structure()
    file_path = directory_structure_dict['path_templates'][type]
    if parameters_list is not None:
        file_path = file_path % tuple(parameters_list)
    directory_path = os.path.split(file_path)[0]

    # if directory does not exist, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    #
    # write response summary file
    #

    if output_type == 'json':
        with open(file_path, 'w') as file:
            file.write(json.dumps(data))

    if output_type == 'numpy_array':
        np.save(file_path, data)

    if output_type == 'pickle_dict':
        with open(file_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_path(type, parameters_list, force=False):
    directory_structure_dict = load_directory_structure()
    file_path = directory_structure_dict['path_templates'][type]
    if parameters_list is not None:
        file_path = file_path % tuple(parameters_list)

    # check if file exists
    if not os.path.isfile(file_path) and not os.path.isdir(file_path):
        logger.warning("file does not exit")
        if force:
            return file_path
        return None

    return file_path


def get_template(type):
    directory_structure_dict = load_directory_structure()
    return directory_structure_dict['path_templates'][type]