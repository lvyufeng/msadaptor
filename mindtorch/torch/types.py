class device():
    def __init__(self, type=None, index=None):
        if type is not None:
            if isinstance(type, str):
                if ':' in type:
                    if index is not None:
                        raise ValueError("`type` must not include an index because index was "
                                         f"passed explicitly: {type}")
                    _target, _id = type.split(':')
                    _id = int(_id)
                else:
                    _target = type
                    _id = index
            elif isinstance(type, device):
                if index is not None:
                    raise ValueError("torch.device(): When input is torch.device, `index` can not be set.")
                _target = type.type
                _id = type.index
            else:
                raise TypeError("torch.device(): `type` must be type of 'str' or 'torch.device'.")
        else:
            raise ValueError("torch.device(): `type` can not be None")

        self.type = _target
        self.index = _id

    def __repr__(self):
        if self.index is None:
            return f"device(type={self.type})"
        return f"device(type={self.type}, index={self.index})"

    def __eq__(self, __value):
        if not isinstance(__value, device):
            return False
        return hash(self) == hash(__value)

    def __hash__(self):
        return hash(str(self))
