import os
import torch

def load_value_file(file_path):
    """ Load value from a file
    """
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def set_device(config):

    if config.gpus == "": # cpu
        return 'cpu', False, ""
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(config.gpus)

        if torch.cuda.is_available() is False: # cpu
            return 'cpu', False, ""
        else:
            # gpus = config.gpus.split(',') # if config.gpus is a list
            # gpus = (',').join(list(map(str, range(0, len(gpus))))) # generate a list of string number from 0 to len(config.gpus)
            gpus = list(range(len(config.gpus)))
            if config.parallel is True and len(gpus) > 1: # multi gpus
                return 'cuda:0', True, gpus
            else: # single gpu
                return 'cuda:'+ str(gpus[0]), False, gpus
