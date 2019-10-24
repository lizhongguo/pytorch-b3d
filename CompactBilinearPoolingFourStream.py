import types
import torch
import torch.nn as nn
from torch.autograd import Function


def CountSketchFn_forward(h, s, output_size, x, force_cpu_scatter_add=False):
    x_size = tuple(x.size())

    s_view = (1,) * (len(x_size)-1) + (x_size[-1],)

    out_size = x_size[:-1] + (output_size,)

    # Broadcast s and compute x * s
    s = s.view(s_view)
    xs = x * s

    # Broadcast h then compute h:
    # out[h_i] += x_i * s_i
    h = h.view(s_view).expand(x_size)

    if force_cpu_scatter_add:
        out = x.new(*out_size).zero_().cpu()
        return out.scatter_add_(-1, h.cpu(), xs.cpu()).cuda()
    else:
        out = x.new(*out_size).zero_()
        return out.scatter_add_(-1, h, xs)


def CountSketchFn_backward(h, s, x_size, grad_output):
    s_view = (1,) * (len(x_size)-1) + (x_size[-1],)

    s = s.view(s_view)
    h = h.view(s_view).expand(x_size)

    grad_x = grad_output.gather(-1, h)
    grad_x = grad_x * s
    return grad_x


class CountSketchFn(Function):

    @staticmethod
    def forward(ctx, h, s, output_size, x, force_cpu_scatter_add=False):
        x_size = tuple(x.size())

        ctx.save_for_backward(h, s)
        ctx.x_size = tuple(x.size())

        return CountSketchFn_forward(h, s, output_size, x, force_cpu_scatter_add)

    @staticmethod
    def backward(ctx, grad_output):
        h, s = ctx.saved_variables

        grad_x = CountSketchFn_backward(h, s, ctx.x_size, grad_output)
        return None, None, None, grad_x


class CountSketch(nn.Module):
    r"""Compute the count sketch over an input signal.

    .. math::

        out_j = \sum_{i : j = h_i} s_i x_i

    Args:
        input_size (int): Number of channels in the input array
        output_size (int): Number of channels in the output sketch
        h (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s (array, optional): Optional array of size input_size of -1 and 1.

    .. note::

        If h and s are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input: (...,input_size)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input_size, output_size, h=None, s=None):
        super(CountSketch, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        if h is None:
            h = torch.LongTensor(input_size).random_(0, output_size)
        if s is None:
            s = 2 * torch.Tensor(input_size).random_(0, 2) - 1

        # The Variable h being a list of indices,
        # If the type of this module is changed (e.g. float to double),
        # the variable h should remain a LongTensor
        # therefore we force float() and double() to be no-ops on the variable h.
        def identity(self):
            return self

        h.float = types.MethodType(identity, h)
        h.double = types.MethodType(identity, h)

        self.register_buffer('h', h)
        self.register_buffer('s', s)

    def forward(self, x):
        x_size = list(x.size())

        assert(x_size[-1] == self.input_size)

        return CountSketchFn.apply(self.h, self.s, self.output_size, x)


def ComplexMultiply_forward(X_re, X_im, Y_re, Y_im):
    Z_re = torch.addcmul(X_re*Y_re, -1, X_im, Y_im)
    Z_im = torch.addcmul(X_re*Y_im,  1, X_im, Y_re)
    return Z_re, Z_im


def ComplexMultiply_backward(X_re, X_im, Y_re, Y_im, grad_Z_re, grad_Z_im):
    grad_X_re = torch.addcmul(grad_Z_re * Y_re,  1, grad_Z_im, Y_im)
    grad_X_im = torch.addcmul(grad_Z_im * Y_re, -1, grad_Z_re, Y_im)
    grad_Y_re = torch.addcmul(grad_Z_re * X_re,  1, grad_Z_im, X_im)
    grad_Y_im = torch.addcmul(grad_Z_im * X_re, -1, grad_Z_re, X_im)
    return grad_X_re, grad_X_im, grad_Y_re, grad_Y_im


def ComplexMultiplyFourStream_forward(A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im):
    X_re, X_im = ComplexMultiply_forward(A_re, A_im, B_re, B_im)
    Y_re, Y_im = ComplexMultiply_forward(C_re, C_im, D_re, D_im)
    Z_re, Z_im = ComplexMultiply_forward(X_re, X_im, Y_re, Y_im)
    return Z_re, Z_im


def ComplexMultiplyFourStream_backward(A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im, grad_Z_re, grad_Z_im):
    X_re, X_im = ComplexMultiply_forward(A_re, A_im, B_re, B_im)
    Y_re, Y_im = ComplexMultiply_forward(C_re, C_im, D_re, D_im)
    grad_X_re, grad_X_im, grad_Y_re, grad_Y_im = ComplexMultiply_backward(
        X_re, X_im, Y_re, Y_im, grad_Z_re, grad_Z_im)
    grad_A_re, grad_A_im, grad_B_re, grad_B_im = ComplexMultiply_backward(
        A_re, A_im, B_re, B_im, grad_X_re, grad_X_im)
    grad_C_re, grad_C_im, grad_D_re, grad_D_im = ComplexMultiply_backward(
        C_re, C_im, D_re, D_im, grad_Y_re, grad_Y_im)

    return grad_A_re, grad_A_im, grad_B_re, grad_B_im, grad_C_re, grad_C_im, grad_D_re, grad_D_im


class ComplexMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_re, X_im, Y_re, Y_im):
        ctx.save_for_backward(X_re, X_im, Y_re, Y_im)
        return ComplexMultiply_forward(X_re, X_im, Y_re, Y_im)

    @staticmethod
    def backward(ctx, grad_Z_re, grad_Z_im):
        X_re, X_im, Y_re, Y_im = ctx.saved_tensors
        return ComplexMultiply_backward(X_re, X_im, Y_re, Y_im, grad_Z_re, grad_Z_im)


class ComplexMultiplyFourStream(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im):
        ctx.save_for_backward(A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im)
        return ComplexMultiplyFourStream_forward(A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im)

    @staticmethod
    def backward(ctx, grad_Z_re, grad_Z_im):
        A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im = ctx.saved_tensors
        return ComplexMultiplyFourStream_backward(A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im, grad_Z_re, grad_Z_im)


class CompactBilinearPoolingFn(Function):

    @staticmethod
    def forward(ctx, h1, s1, h2, s2, output_size, x, y, force_cpu_scatter_add=False):
        ctx.save_for_backward(h1, s1, h2, s2, x, y)
        ctx.x_size = tuple(x.size())
        ctx.y_size = tuple(y.size())
        ctx.force_cpu_scatter_add = force_cpu_scatter_add
        ctx.output_size = output_size

        # Compute the count sketch of each input
        px = CountSketchFn_forward(
            h1, s1, output_size, x, force_cpu_scatter_add)
        fx = torch.rfft(px, 1)
        re_fx = fx.select(-1, 0)
        im_fx = fx.select(-1, 1)
        del px
        py = CountSketchFn_forward(
            h2, s2, output_size, y, force_cpu_scatter_add)
        fy = torch.rfft(py, 1)
        re_fy = fy.select(-1, 0)
        im_fy = fy.select(-1, 1)
        del py

        # Convolution of the two sketch using an FFT.
        # Compute the FFT of each sketch

        # Complex multiplication
        re_prod, im_prod = ComplexMultiply_forward(re_fx, im_fx, re_fy, im_fy)

        # Back to real domain
        # The imaginary part should be zero's
        re = torch.irfft(torch.stack((re_prod, im_prod),
                                     re_prod.dim()), 1, signal_sizes=(output_size,))

        return re

    @staticmethod
    def backward(ctx, grad_output):
        h1, s1, h2, s2, x, y = ctx.saved_tensors

        # Recompute part of the forward pass to get the input to the complex product
        # Compute the count sketch of each input
        px = CountSketchFn_forward(
            h1, s1, ctx.output_size, x, ctx.force_cpu_scatter_add)
        py = CountSketchFn_forward(
            h2, s2, ctx.output_size, y, ctx.force_cpu_scatter_add)

        # Then convert the output to Fourier domain
        grad_output = grad_output.contiguous()
        grad_prod = torch.rfft(grad_output, 1)
        grad_re_prod = grad_prod.select(-1, 0)
        grad_im_prod = grad_prod.select(-1, 1)

        # Compute the gradient of x first then y

        # Gradient of x
        # Recompute fy
        fy = torch.rfft(py, 1)
        re_fy = fy.select(-1, 0)
        im_fy = fy.select(-1, 1)
        del py
        # Compute the gradient of fx, then back to temporal space
        grad_re_fx = torch.addcmul(
            grad_re_prod * re_fy,  1, grad_im_prod, im_fy)
        grad_im_fx = torch.addcmul(
            grad_im_prod * re_fy, -1, grad_re_prod, im_fy)
        grad_fx = torch.irfft(torch.stack(
            (grad_re_fx, grad_im_fx), grad_re_fx.dim()), 1, signal_sizes=(ctx.output_size,))
        # Finally compute the gradient of x
        grad_x = CountSketchFn_backward(h1, s1, ctx.x_size, grad_fx)
        del re_fy, im_fy, grad_re_fx, grad_im_fx, grad_fx

        # Gradient of y
        # Recompute fx
        fx = torch.rfft(px, 1)
        re_fx = fx.select(-1, 0)
        im_fx = fx.select(-1, 1)
        del px
        # Compute the gradient of fy, then back to temporal space
        grad_re_fy = torch.addcmul(
            grad_re_prod * re_fx,  1, grad_im_prod, im_fx)
        grad_im_fy = torch.addcmul(
            grad_im_prod * re_fx, -1, grad_re_prod, im_fx)
        grad_fy = torch.irfft(torch.stack(
            (grad_re_fy, grad_im_fy), grad_re_fy.dim()), 1, signal_sizes=(ctx.output_size,))
        # Finally compute the gradient of y
        grad_y = CountSketchFn_backward(h2, s2, ctx.y_size, grad_fy)
        del re_fx, im_fx, grad_re_fy, grad_im_fy, grad_fy

        return None, None, None, None, None, grad_x, grad_y, None


class CompactBilinearPoolingFnFourStream(Function):

    @staticmethod
    def forward(ctx, h1, s1, h2, s2, h3, s3, h4, s4, output_size, a, b, c, d, force_cpu_scatter_add=False):
        ctx.save_for_backward(h1, s1, h2, s2, h3, s3, h4, s4, a, b, c, d)
        ctx.a_size = tuple(a.size())
        ctx.b_size = tuple(b.size())
        ctx.c_size = tuple(c.size())
        ctx.d_size = tuple(d.size())

        ctx.force_cpu_scatter_add = force_cpu_scatter_add
        ctx.output_size = output_size

        # Compute the count sketch of each input
        pa = CountSketchFn_forward(
            h1, s1, output_size, a, force_cpu_scatter_add)
        fa = torch.rfft(pa, 1)
        re_fa = fa.select(-1, 0)
        im_fa = fa.select(-1, 1)
        del pa

        pb = CountSketchFn_forward(
            h2, s2, output_size, b, force_cpu_scatter_add)
        fb = torch.rfft(pb, 1)
        re_fb = fb.select(-1, 0)
        im_fb = fb.select(-1, 1)
        del pb

        pc = CountSketchFn_forward(
            h3, s3, output_size, c, force_cpu_scatter_add)
        fc = torch.rfft(pc, 1)
        re_fc = fc.select(-1, 0)
        im_fc = fc.select(-1, 1)
        del pc

        pd = CountSketchFn_forward(
            h4, s4, output_size, d, force_cpu_scatter_add)
        fd = torch.rfft(pd, 1)
        re_fd = fd.select(-1, 0)
        im_fd = fd.select(-1, 1)
        del pd

        # Convolution of the two sketch using an FFT.
        # Compute the FFT of each sketch

        # Complex multiplication
        re_prod, im_prod = ComplexMultiplyFourStream_forward(
            re_fa, im_fa, re_fb, im_fb, re_fc, im_fc, re_fd, im_fd)

        # Back to real domain
        # The imaginary part should be zero's
        re = torch.irfft(torch.stack((re_prod, im_prod),
                                     re_prod.dim()), 1, signal_sizes=(output_size,))

        return re

    @staticmethod
    def backward(ctx, grad_output):
        h1, s1, h2, s2, h3, s3, h4, s4, a, b, c, d = ctx.saved_tensors

        # Recompute part of the forward pass to get the input to the complex product
        # Compute the count sketch of each input
        pa = CountSketchFn_forward(
            h1, s1, ctx.output_size, a, ctx.force_cpu_scatter_add)
        pb = CountSketchFn_forward(
            h2, s2, ctx.output_size, b, ctx.force_cpu_scatter_add)
        pc = CountSketchFn_forward(
            h2, s2, ctx.output_size, c, ctx.force_cpu_scatter_add)
        pd = CountSketchFn_forward(
            h2, s2, ctx.output_size, d, ctx.force_cpu_scatter_add)

        # Then convert the output to Fourier domain
        grad_output = grad_output.contiguous()
        grad_prod = torch.rfft(grad_output, 1)
        grad_re_prod = grad_prod.select(-1, 0)
        grad_im_prod = grad_prod.select(-1, 1)

        # Compute the gradient of x first then y

        # Gradient of x
        # Recompute fy
        fa = torch.rfft(pa, 1)
        re_fa = fa.select(-1, 0)
        im_fa = fa.select(-1, 1)
        del pa
        fb = torch.rfft(pb, 1)
        re_fb = fb.select(-1, 0)
        im_fb = fb.select(-1, 1)
        del pb
        fc = torch.rfft(pc, 1)
        re_fc = fc.select(-1, 0)
        im_fc = fc.select(-1, 1)
        del pc
        fd = torch.rfft(pd, 1)
        re_fd = fd.select(-1, 0)
        im_fd = fd.select(-1, 1)
        del pd

        grad_re_fa, grad_im_fa, grad_re_fb, grad_im_fb, grad_re_fc, grad_im_fc, grad_re_fd, grad_im_fd = \
            ComplexMultiplyFourStream_backward(
                re_fa, im_fa, re_fb, im_fb, re_fc, im_fc, re_fd, im_fd, grad_re_prod, grad_im_prod)

        # Compute the gradient of fx, then back to temporal space
        grad_fa = torch.irfft(torch.stack(
            (grad_re_fa, grad_im_fa), grad_re_fa.dim()), 1, signal_sizes=(ctx.output_size,))
        grad_fb = torch.irfft(torch.stack(
            (grad_re_fb, grad_im_fb), grad_re_fb.dim()), 1, signal_sizes=(ctx.output_size,))
        grad_fc = torch.irfft(torch.stack(
            (grad_re_fc, grad_im_fc), grad_re_fc.dim()), 1, signal_sizes=(ctx.output_size,))
        grad_fd = torch.irfft(torch.stack(
            (grad_re_fd, grad_im_fd), grad_re_fd.dim()), 1, signal_sizes=(ctx.output_size,))

        # Finally compute the gradient of x
        grad_a = CountSketchFn_backward(h1, s1, ctx.a_size, grad_fa)
        grad_b = CountSketchFn_backward(h2, s2, ctx.b_size, grad_fb)
        grad_c = CountSketchFn_backward(h3, s3, ctx.c_size, grad_fc)
        grad_d = CountSketchFn_backward(h4, s4, ctx.d_size, grad_fd)

        del re_fa, im_fa, grad_re_fa, grad_im_fa, grad_fa, \
            re_fb, im_fb, grad_re_fb, grad_im_fb, grad_fb, \
            re_fc, im_fc, grad_re_fc, grad_im_fc, grad_fc, \
            re_fd, im_fd, grad_re_fd, grad_im_fd, grad_fd

        return None, None, None, None, None, None, None, None, None, grad_a, grad_b, grad_c, grad_d, None


class CompactBilinearPooling(nn.Module):
    r"""Compute the compact bilinear pooling between two input array x and y

    .. math::

        out = \Psi (x,h_1,s_1) \ast \Psi (y,h_2,s_2)

    Args:
        input_size1 (int): Number of channels in the first input array
        input_size2 (int): Number of channels in the second input array
        output_size (int): Number of channels in the output array
        h1 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s1 (array, optional): Optional array of size input_size of -1 and 1.
        h2 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s2 (array, optional): Optional array of size input_size of -1 and 1.
        force_cpu_scatter_add (boolean, optional): Force the scatter_add operation to run on CPU for testing purposes

    .. note::

        If h1, s1, s2, h2 are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input 1: (...,input_size1)
        - Input 2: (...,input_size2)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input1_size, input2_size, output_size, h1=None, s1=None, h2=None, s2=None, force_cpu_scatter_add=False):
        super(CompactBilinearPooling, self).__init__()
        self.add_module('sketch1', CountSketch(
            input1_size, output_size, h1, s1))
        self.add_module('sketch2', CountSketch(
            input2_size, output_size, h2, s2))
        self.output_size = output_size
        self.force_cpu_scatter_add = force_cpu_scatter_add

    def forward(self, x, y=None):
        if y is None:
            y = x

        return CompactBilinearPoolingFn.apply(self.sketch1.h, self.sketch1.s, self.sketch2.h, self.sketch2.s, self.output_size, x, y, self.force_cpu_scatter_add)


class CompactBilinearPoolingFourStream(nn.Module):
    r"""Compute the compact bilinear pooling between two input array x and y

    .. math::

        out = \Psi (x,h_1,s_1) \ast \Psi (y,h_2,s_2)

    Args:
        input_size1 (int): Number of channels in the first input array
        input_size2 (int): Number of channels in the second input array
        output_size (int): Number of channels in the output array
        h1 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s1 (array, optional): Optional array of size input_size of -1 and 1.
        h2 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s2 (array, optional): Optional array of size input_size of -1 and 1.
        force_cpu_scatter_add (boolean, optional): Force the scatter_add operation to run on CPU for testing purposes

    .. note::

        If h1, s1, s2, h2 are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input 1: (...,input_size1)
        - Input 2: (...,input_size2)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input1_size, input2_size, input3_size, input4_size, output_size,
                 h1=None, s1=None, h2=None, s2=None, h3=None, s3=None, h4=None, s4=None, force_cpu_scatter_add=False):
        super(CompactBilinearPoolingFourStream, self).__init__()
        self.add_module('sketch1', CountSketch(
            input1_size, output_size, h1, s1))
        self.add_module('sketch2', CountSketch(
            input2_size, output_size, h2, s2))
        self.add_module('sketch3', CountSketch(
            input3_size, output_size, h3, s3))
        self.add_module('sketch4', CountSketch(
            input4_size, output_size, h4, s4))

        self.output_size = output_size
        self.force_cpu_scatter_add = force_cpu_scatter_add

    def forward(self, a, b, c, d):
        return CompactBilinearPoolingFnFourStream.apply(self.sketch1.h, self.sketch1.s, self.sketch2.h, self.sketch2.s,
                                              self.sketch3.h, self.sketch3.s, self.sketch4.h, self.sketch4.s,
                                              self.output_size, a, b, c, d, self.force_cpu_scatter_add)
