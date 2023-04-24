import yaml
from pdb import set_trace
import yb_utils as yb
import numpy as np

with open('../COMPUTE/cantor.yml', 'r') as stream:
    par = yaml.safe_load(stream)

par['Lost']['Recover'] = True
print(par)

def dict2att(dictIn, outloc, container='Header', pre='',
             bool_as_int=True):
    """
    Write all elements of a dictionary as attributes to an HDF5 file.

    Typically, this is used to write the paramter structure of a 
    program to its output. If keys are themselves dictionaries, these 
    are recursively output with an underscore between them 
    (e.g. dictIn['Sim']['Input']['Flag'] --> 'Sim_Input_Flag').

    Parameters:
    -----------
    dictIn : dict
        The dictionary to output.
    outloc : string
        The HDF5 file name to write the dictionary to.
    container : string, optional
        The container to which the dict's elements will be written as 
        attributes (can be group or dataset). The default is a group 
        'Header'. If the container does not exist, a group with the specified
        name will be implicitly created.
    pre : string, optional:
        A prefix given to all keys from this dictionary. This is mostly
        used to output nested dictionaries (see description at top), but 
        may also be used to append a 'global' prefix to all keys.
    bool_as_int : bool, optional:
        If True (default), boolean keys will be written as 0 or 1, instead
        of as True and False.
    """

    if len(pre):
        preOut = pre + '_'
    else:
        preOut = pre

    for key in dictIn.keys():

        value = dictIn[key]
        if isinstance(value, dict):
            # Nested dict: call function again to iterate
            dict2att(value, outloc, container = container, 
                     pre = preOut + key)
        else:
            # Single value: write to HDF5 file

            if value is None:
                value = 0

            if value is True:
                value = 1
            if value is False:
                value = 0

            if isinstance(value, str):
                value = np.string_(value)

            yb.write_hdf5_attribute(outloc, container, preOut + key, 
                                    value)

dict2att(par, '/virgo/scratch/ybahe/TESTS/AttTest.hdf5', 'Header')

set_trace()

