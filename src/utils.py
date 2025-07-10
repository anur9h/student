import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException

import dill


def save_object(file_path, obj):
    """
    Save an object to a file using dill.
    
    Parameters:
    file_path (str): The path where the object will be saved.
    obj: The object to be saved.
    
    Returns:
    None
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        
    except Exception as e:
        raise CustomException(e, sys)