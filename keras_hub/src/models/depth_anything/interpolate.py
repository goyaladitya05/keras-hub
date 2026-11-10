from keras import backend
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


def interpolate(x, size, data_format=None):
    """Performs a backend-agnostic version of Torch's `F.interpolate`.

    Args:
        x: A 4D image tensor.
        size: A tuple of 2 integers, `(height, width)`.
        data_format: One of `channels_last` or `channels_first`.
    """
    data_format = standardize_data_format(data_format)
    if backend.backend() == "jax":
        import jax

        if data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))
        scale = ops.convert_to_tensor(
            [
                (size[0] - 1.0) / (x.shape[1] - 1.0),
                (size[1] - 1.0) / (x.shape[2] - 1.0),
            ]
        )
        translation = -(scale / 2.0 - 0.5)
        x = jax.image.scale_and_translate(
            x,
            (x.shape[0], *size, x.shape[-1]),
            method="bilinear",
            scale=scale,
            spatial_dims=(1, 2),
            translation=translation,
            antialias=False,
        )
        if data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))
    elif backend.backend() == "tensorflow":
        import tensorflow as tf

        if data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))
        x = tf.compat.v1.image.resize(
            x,
            size=size,
            method="bilinear",
            align_corners=True,
        )
        if data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))
    elif backend.backend() == "torch":
        import torch.nn.functional as F

        if data_format == "channels_last":
            x = ops.transpose(x, (0, 3, 1, 2))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        if data_format == "channels_last":
            x = ops.transpose(x, (0, 2, 3, 1))
    else:
        if data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))
        x = _bilinear_align_corners(x, size)
        if data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))
    return x


def _bilinear_align_corners(x, size):
    """Bilinear resize with `align_corners=True` using only `keras.ops`.

    Args:
        x: A 4D `channels_last` image tensor with static spatial dims.
        size: A tuple of 2 integers, `(height, width)`.
    """

    def _coords(in_len, out_len):
        if in_len == 1 or out_len == 1:
            pos = ops.zeros((out_len,), dtype="float32")
        else:
            scale = (in_len - 1) / (out_len - 1)
            pos = ops.arange(out_len, dtype="float32") * scale
        lo = ops.floor(pos)
        hi = ops.minimum(lo + 1, in_len - 1)
        return ops.cast(lo, "int32"), ops.cast(hi, "int32"), pos - lo

    out_h, out_w = size
    y0, y1, wy = _coords(x.shape[1], out_h)
    x0, x1, wx = _coords(x.shape[2], out_w)
    wy = ops.cast(ops.reshape(wy, (1, out_h, 1, 1)), x.dtype)
    wx = ops.cast(ops.reshape(wx, (1, 1, out_w, 1)), x.dtype)
    top = ops.take(x, y0, axis=1)
    bottom = ops.take(x, y1, axis=1)
    rows = top + (bottom - top) * wy
    left = ops.take(rows, x0, axis=2)
    right = ops.take(rows, x1, axis=2)
    return left + (right - left) * wx
