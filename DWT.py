import torch.nn as nn
import pywt
import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import pdb


def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def prep_filt_afb3d(h0, h1, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()

    h0_T = torch.nn.Parameter(data=torch.tensor(
        h0, device=device, dtype=t).reshape((1, 1, -1, 1, 1)), requires_grad=False)
    h1_T = torch.nn.Parameter(data=torch.tensor(
        h1, device=device, dtype=t).reshape((1, 1, -1, 1, 1)), requires_grad=False)
    h0_H = torch.nn.Parameter(data=torch.tensor(
        h0, device=device, dtype=t).reshape((1, 1, 1, -1, 1)), requires_grad=False)
    h1_H = torch.nn.Parameter(data=torch.tensor(
        h1, device=device, dtype=t).reshape((1, 1, 1, -1, 1)), requires_grad=False)
    h0_W = torch.nn.Parameter(data=torch.tensor(
        h0, device=device, dtype=t).reshape((1, 1, 1, 1, -1)), requires_grad=False)
    h1_W = torch.nn.Parameter(data=torch.tensor(
        h1, device=device, dtype=t).reshape((1, 1, 1, 1, -1)), requires_grad=False)

    return h0_T, h1_T, h0_H, h1_H, h0_W, h1_W


def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).

    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 5

    s = [1, 1, 1]
    s[d-2] = 2

    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order

    L = h0.numel()
    L2 = L // 2
    shape = [1, 1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)
    p = 2 * (outsize - 1) - N + L
    if mode == 'zero':
        # Sadly, pytorch only allows for same padding before and after, if
        # we need to do more padding after for odd length signals, have to
        # prepad
        if p % 2 == 1:
            pad = [0, 0, 0]
            pad[d-2] = 1
            x = F.pad(x, pad)

        pad = [0, 0, 0]
        pad[d-2] = p//2

        # Calculate the high and lowpass
        lohi = F.conv3d(x, h, padding=pad, stride=s, groups=C)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))
    return lohi


def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 5
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    L = g0.numel()
    shape = [1, 1, 1, 1, 1]
    shape[d] = L
    N = 2*lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = [1, 1, 1]
    s[d-2] = 2
    g0 = torch.cat([g0]*C, dim=0)
    g1 = torch.cat([g1]*C, dim=0)

    if mode == 'zero':
        pad = [0, 0, 0]
        pad[d-2] = L-2
        y = F.conv_transpose3d(lo, g0, stride=s, padding=pad, groups=C) + \
            F.conv_transpose3d(hi, g1, stride=s, padding=pad, groups=C)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    return y


class AFB3D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    @staticmethod
    def forward(ctx, x, lo_T, hi_T, lo_H, hi_H, lo_W, hi_W, mode):
        ctx.save_for_backward(lo_T, hi_T, lo_H, hi_H, lo_W, hi_W)
        ctx.shape = x.shape[-3:]
        mode = int_to_mode(mode)
        ctx.mode = mode

        if lo_H is None and lo_W is None:
            y = afb1d(x, lo_T, hi_T, dim=-3)
        elif lo_T is None:
            x_H = afb1d(x, lo_H, hi_H, mode=mode, dim=-2)
            y = afb1d(x_H, lo_W, hi_W, mode=mode, dim=-1)
        else:
            x_T = afb1d(x, lo_T, hi_T, mode=mode, dim=-3)
            x_H = afb1d(x_T, lo_H, hi_H, mode=mode, dim=-2)
            y = afb1d(x_H, lo_W, hi_W, mode=mode, dim=-1)
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            lo_T, hi_T, lo_H, hi_H, lo_W, hi_W = ctx.saved_tensors

            if lo_H is None and lo_W is None:
                s = dy.shape
                dy = dy.reshape(s[0], -1, 2, s[2], s[3], s[4])
                l, h = torch.unbind(dy, dim=2)
                dx = sfb1d(l, h, lo_T, hi_T, mode=mode, dim=-3)

            elif lo_T is None:
                s = dy.shape
                dy = dy.reshape(s[0], -1, 4, s[2], s[3], s[4])
                ll, lh, hl, hh = torch.unbind(dy, dim=2)
                l = sfb1d(ll, lh, lo_W, hi_W, mode=mode, dim=-1)
                h = sfb1d(hl, hh, lo_W, hi_W, mode=mode, dim=-1)
                dx = sfb1d(l, h, lo_H, hi_H, mode=mode, dim=-2)

            else:
                s = dy.shape
                dy = dy.reshape(s[0], -1, 8, s[2], s[3], s[4])
                lll, llh, lhl, lhh, hll, hlh, hhl, hhh = torch.unbind(
                    dy, dim=2)
                ll = sfb1d(lll, llh, lo_W, hi_W, mode=mode, dim=-1)
                lh = sfb1d(lhl, lhh, lo_W, hi_W, mode=mode, dim=-1)
                hl = sfb1d(hll, hlh, lo_W, hi_W, mode=mode, dim=-1)
                hh = sfb1d(hhl, hhh, lo_W, hi_W, mode=mode, dim=-1)
                l = sfb1d(ll, lh, lo_H, hi_H, mode=mode, dim=-2)
                h = sfb1d(hl, hh, lo_H, hi_H, mode=mode, dim=-2)
                dx = sfb1d(l, h, lo_T, hi_T, mode=mode, dim=-3)

            return dx, None, None, None, None, None, None, None


class DWT3D(nn.Module):
    """ Performs a 3d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        separable (bool): whether to do the filtering separably or not (the
            naive implementation can be faster on a gpu).
        only_hw (bool): set True while temporal pooling is not needed
        """

    def __init__(self, J=1, wave='db1', mode='zero', dim='thw'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        self.lo_T, self.hi_T, self.lo_H, self.hi_H, self.lo_W, self.hi_W = prep_filt_afb3d(
            h0_col, h1_col, device=None)
        self.mode = mode
        self.dim = dim
        assert(self.dim in ('t', 'hw', 'thw'))

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh)
                coefficients. yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """

        mode = mode_to_int(self.mode)

        # Do a multilevel transform
        if self.dim == 't':
            result = AFB3D.apply(x,
                                 self.lo_T, self.hi_T, None, None,
                                 None, None, mode)

        elif self.dim == 'hw':
            result = AFB3D.apply(x,
                                 None, None, self.lo_H, self.hi_H,
                                 self.lo_W, self.hi_W, mode)
        else:
            result = AFB3D.apply(x,
                                 self.lo_T, self.hi_T, self.lo_H, self.hi_H,
                                 self.lo_W, self.hi_W, mode)

        return result
