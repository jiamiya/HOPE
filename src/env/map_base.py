import numpy as np
from shapely.geometry.base import BaseGeometry

class Area(object):
    def __init__(
        self,
        shape: BaseGeometry = None, 
        subtype: str = None,
        color: float = None,
    ):
        self.shape = shape
        self.subtype = subtype
        self.color = color

    def get_shape(self):
        return np.array(self.shape.coords)