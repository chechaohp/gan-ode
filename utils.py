

def load_value_file(file_path):
    """ Load value from a file
    """
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value