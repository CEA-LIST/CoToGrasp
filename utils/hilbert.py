"""
Hilbert Order
Modified from https://github.com/PrincetonLIPS/numpy-hilbert-curve

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Kaixin Xu
Please cite our work if the code is helpful to you.
"""
import torch

def right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     k: The number of bits to shift. Default 1.

     axis: The axis along which to shift.  Default -1.

    Returns:
    --------
     Returns an ndarray with zero prepended and the ends truncated, along
     whatever axis was specified."""

    # If we're shifting the whole thing, just return zeros.
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    # Determine the padding pattern.
    # padding = [(0,0)] * len(binary.shape)
    # padding[axis] = (k,0)

    # Determine the slicing pattern to eliminate just the last one.
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted

def gray2binary(gray, axis=-1):
    """Convert an array of Gray codes back into binary values.

    Parameters:
    -----------
     gray: An ndarray of gray codes.

     axis: The axis along which to perform Gray decoding. Default=-1.

    Returns:
    --------
     Returns an ndarray of binary values.
    """

    # Loop the log2(bits) number of times necessary, with shift and xor.
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def encode(locs, num_dims, num_bits):
    """Decode an array of locations in a hypercube into a Hilbert integer.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     locs - An ndarray of locations in a hypercube of num_dims dimensions, in
            which each dimension runs from 0 to 2**num_bits-1.  The shape can
            be arbitrary, as long as the last dimension of the same has size
            num_dims.

     num_dims - The dimensionality of the hypercube. Integer.

     num_bits - The number of bits for each dimension. Integer.

    Returns:
    --------
     The output is an ndarray of uint64 integers with the same shape as the
     input, excluding the last dimension, which needs to be num_dims.
    """

    # Keep around the original shape for later.
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            """
      The shape of locs was surprising in that the last dimension was of size
      %d, but num_dims=%d.  These need to be equal.
      """
            % (orig_shape[-1], num_dims)
        )

    if num_dims * num_bits > 63:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a int64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits, num_dims * num_bits)
        )

    # Treat the location integers as 64-bit unsigned and then split them up into
    # a sequence of uint8s.  Preserve the association by dimension.
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

    # Now turn these into bits and truncate to num_bits.
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    # Run the decoding process the other way.
    # Iterate forwards through the bits.
    for bit in range(0, num_bits):
        # Iterate forwards through the dimensions.
        for dim in range(0, num_dims):
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Now flatten out.
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

    # Convert Gray back to binary.
    hh_bin = gray2binary(gray)

    # Pad back out to 64 bits.
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)

    # Convert binary values into uint8s.
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )

    # Convert uint8s into uint64s.
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()

    return hh_uint64
