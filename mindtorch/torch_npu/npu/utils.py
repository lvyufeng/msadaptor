class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.prev_idx = -1

    def __enter__(self):
        return

    def __exit__(self, *args):
        return