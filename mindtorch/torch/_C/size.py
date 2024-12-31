import operator
from functools import reduce


def _get_tuple_numel(input):
    return reduce(operator.mul, list(input))


class Size(tuple):
    def __new__(cls, shape):
        _shape = shape
        if not isinstance(_shape, (tuple, list)):
            raise TypeError("{} object is not supportted.".format(type(shape)))
        return tuple.__new__(Size, _shape)

    def numel(self):
        return _get_tuple_numel(self)

    def __repr__(self):
        return "torch.Size(" + str(list(self)) + ")"
