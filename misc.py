# -*- coding: utf-8 -*-

import sys
from numbers import Integral
from math import *
from collections.abc import *
from types import FunctionType
import inspect as ins
import re


def copy_func(f, name=None):
    """
    Return a function with same code, globals, defaults, closure, and
    clsname (or provide a new clsname).
    """
    return FunctionType(
        f.__code__,
        f.__globals__,
        name or f.__name__,
        f.__defaults__,
        f.__closure__
    )


def subdic(dic, keys=None, inplace=False):
    """
    Return a subset of the input mapping whose keys match the ``keys``
    iterable.

    :param dic: The input mapping (aka dictionary).
    :param keys: An iterable of valid (== hashable) keys.
    :param inplace: If True, perform "in place" operation, meaning that each
        matching item is first popped from the original before it is added to
        the so that at at any point the total amount of memory consumed during
        the operation is constant (or almost constant, as it is not a true
        operation; a new dictionary is created rather than non-matching items
        being deleted from the original mapping.
    :return:
    """

    if keys is None:
        return dic if inplace else dic.copy()

    f = dic.pop if inplace else dic.__getitem__
    # If keys is a Collection (and hence not an Iterator), use only
    # compositions of (lazy) built-ins for efficiency (both time and space).
    if isinstance(keys, Collection):
        keys = *filter(dic.__contains__, keys),
        return type(dic)(zip(keys, map(f, keys)))

    # If ``keys`` is not a ``Collection``, it should be an ``Iterator``. This
    # is not checked however -- we'll let the interpreter raise an error in the
    # generator expression if ``keys`` is neither. The genexpr method is as
    # efficient memory-wise as the ``zip()`` variant but somewhat slower.
    # It is however necessary as ``keys`` would otherwise be consumed twice
    # per ``zip()`` call, resulting in erroneous output.
    return type(dic)((k, f(k)) for k in keys if k in dic)


def eye(x, *args, **kwargs):
    """
    The identity function.

    :param x:
    :param args:
    :param kwargs:
    :return: The input argument, if there is only one, otherwise a tuple of all
        arguments and values of all keyword arguments.
    """
    if args or kwargs:
        return (x, *args, *kwargs.values())
    return x


def norm(x, p=2):
    """
    p-norm of a vector. See https://en.wikipedia.org/wiki/Lp_space

    :param x: Input vector, in the form of an iterable (or a matrix, if numpy).
    :param p: The exponent. A value of 2 (the default) corresponds to the
        Euclidean norm. It can be inf, in which case ``norm(x, p) == max(x)``.
    :return: The norm.
    """
    p = min(max(1, p), sys.float_info.max_10_exp)
    return fsum(map(p.__rpow__, x))**(1/p)


def indices(seq, x):
    """
    Indices (positions) of element ``x`` in a sequence.

    :param seq: Input Sequence.
    :param x: The element to be searched for.
    :return: A list of integers corresponding to the indices in ``seq`` at
        which ``x`` occurs. Comparison is performed by content (not by id),
        meaning that if n is in the returned list, then it is the case that
        ``seq[n] == x`` but not necessarily that ``seq[n] is x``
    """
    idx, idxs = -1, []
    app = idxs.append
    try:
        while True:
            idx = seq.index(x, idx + 1)
            app(idx)
    except ValueError:
        return idxs


def get_caller(n=None):
    frame = ins.currentframe()
    if not isinstance(n, Integral):
        n = sys.maxsize
    names = []

    while True:  # keep moving
        try:
            # to next outer frame
            frame = frame.f_back
            name = frame.f_code.co_name
            print('f_locals', frame.f_locals) # , 'clsname:', clsname)
            caller_class = frame.f_locals['self'].__class__.__name__
            names.append((caller_class, name))
            if name[0] == '!<':
                break
        except KeyError as exc:
            # err(exc)  # prints 'self' to stderr
            break
    return names


def getsignature(routine, *implementors, default=None):
    """
    Retrieves the signature of a callable (method/function/etc). Utility
    function that slightly extends the ``signature`` method of the ``inspect``
    module and tries some alternatives when the latter fails (which it does
    with certain ``dict`` methods, for example). Apart from the signature, the
    built-in/stdlib class that contained an implementation of ``routine`` and,
    when possible, the docstring are also retrieved.

    :param routine: Callable whose signature is to be determined.

    :param implementors: Tuple of classes that contain methods of the same
        name as the callable. These serve as a fallback in case the callable
        does not contain enough metadata to accurately retrieve its signature.

    :param default: Default signature, which is ``('self', '*args', '**kw``),
        unless ``routine`` is a ``classmethod`` or ``staticmethod``, in which
        case the first element, ``'self'``, is omitted.

    :return: A 3-tuple. The first element is the signature, which is itself
        a list of strings containing the names (and, when
        present, the default values) of all arguments. The second
        element is the implementation (if additional ones are supplied,
        otherwise ``None``) from which the signature was retrieved. The last
        element is the ``routine``'s docstring (when available).
    """
    if not callable(routine):
        raise TypeError('First argument must be a callable.')
    name, sig = routine.__name__, None

    if hasattr(routine, '__objclass__'):
        implementors = routine.__objclass__, *implementors

    for c in implementors:
        try:
            f = getattr(c, name)
            doc = f.__doc__

            if sig:
                return sig, c, doc
            sigobj = ins.signature(f)
            sig = sigobj.parameters.values()
            sig = [re.sub(r'<.+?>', 'None', str(s)) for s in sig]
            return sig, c, doc
        except (AttributeError, ValueError):
            pass

    if default is None:
        default = (*{
            classmethod: ('cls',),
            staticmethod: ()
        }.get(type(routine), ('self',)), '*args', '**kwargs')
    return default, None, None


def iskeyword(k):
    """
    Determines whether a string is a Python keyword.

    :param k: The string to be tested.
    """
    return k in {
        'False',
        'None',
        'True',
        'and',
        'as',
        'assert',
        'async',
        'await',
        'break',
        'class',
        'continue',
        'def',
        'del',
        'elif',
        'else',
        'except',
        'finally',
        'for',
        'from',
        'global',
        'if',
        'import',
        'in',
        'is',
        'lambda',
        'nonlocal',
        'not',
        'or',
        'pass',
        'raise',
        'return',
        'try',
        'while',
        'with',
        'yield'
    }


def qualname(obj):
    """
    A more robust version of __qualname__ that tries harder to get the object's
    fully qualified name, in the form of <module_name>.<class_name>.<obj_name>

    :param obj: Input object.
    :return: The fully qualified name of the object.
    """
    if isinstance(obj, type):
        suffix = ''
    else:
        obj, suffix = obj.__class__, '()'

    module = getattr(obj, '__module__', '')
    qname = getattr(obj, '__qualname__', '')
    s = qname or getattr(obj, '__name__', '')
    return f'{module}.{s}{suffix}'


def ordinal(n):
    """
    Ordinal of an integer, in string format. Stolen from StackOverflow.
    :param n: The input integer.
    :return: The ordinal of the input, e.g. ``ordinal(3) == '3rd'``
    """
    return f'{n}' + "tsnrhtdd"[(n//10 % 10 != 1) * (n % 10 < 4) * n % 10::4]


def format_bytes(n, precision=2):
    prefixes, m = ' KMGTPEZY', floor(log2(n)/10)
    mm = min(m, len(prefixes)-1)
    # default precision: bytes -> 0.0f, K-Y -> 0.2f, beyond Y -> 0.2e
    prec_str = f'0.{precision*(m > 0)}{"fe"[m > mm]}'
    return f'{{:{prec_str}}} {prefixes[mm].lstrip()}B'.format(n / (1 << 10*mm))


def err(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def prl(*lines, **kwargs):
    """
    Prints each element in the ``lines`` Iterable in its own line.
    :param lines: Iterable to be printed line-by-line
    """
    kwargs['sep'] = '\n'
    print(*lines, **kwargs)


if __name__ == '__main__':
    from mytools.testing import str2dic
    err(f'sys.argv = {sys.argv}, kwargs = {dict(a=1, b=2)}')
    sd = subdic(str2dic(), (*'aczd',))
    print(sd)
