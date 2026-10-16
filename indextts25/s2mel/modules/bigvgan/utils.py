# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
