import sys, json, importlib, builtins, base64, struct
class _ABI(Exception):
    pass  # signals a malformed payload -> port-protocol, not port-call

# --- PORTS.md 3.2 copy mode ---------------------------------------------
_TAG = 'dmc_tensor'
_VER = 1
_LAYOUT = 'row_major'
_WIDTH = {'i64': 8, 'i32': 4, 'f64': 8, 'f32': 4, 'bf16': 2, 'f16': 2, 'bool': 1}
_STRUCT = {'i64': '<q', 'i32': '<i', 'f64': '<d', 'f32': '<f', 'bool': '?'}
_NP_WIRE = {'int64': 'i64', 'int32': 'i32', 'float64': 'f64',
            'float32': 'f32', 'float16': 'f16', 'bool': 'bool'}
_NP_READ = {'i64': '<i8', 'i32': '<i4', 'f64': '<f8', 'f32': '<f4',
            'f16': '<f2', 'bool': '|b1'}
try:
    import numpy as _np
except Exception:
    _np = None

# id(ndarray) -> (the array, its arrival dtype), for the arrays this request
# rehydrated from an envelope whose dtype numpy cannot hold natively (bf16).
# The dtype belongs to the tensor, not to the storage numpy needed to compute
# with it, so an array the callee hands straight back crosses back as it came;
# an array numpy newly allocated is not in here and crosses as numpy's own.
_ARRIVED = {}

class _Tensor(object):
    # A copy-mode tensor without numpy: the metadata and the raw payload,
    # kept verbatim so an unmodified round trip is byte-identical.
    def __init__(self, dtype, shape, data):
        self.dtype, self.shape, self.data = dtype, shape, data
    def __repr__(self):
        return 'Tensor(%s, %r)' % (self.dtype, self.shape)
    def flat(self):
        # Elements in row-major order, as Python floats -- a convenience for a
        # callee with no numpy, not part of the round trip, which re-encodes
        # `self.data` verbatim. A Python float is a C double, so a signaling
        # NaN quiets on the way through here; that is unavoidable without a
        # real f32 type and costs nothing, because nothing re-encodes from
        # this. bf16 is widened by hand: it is a truncation of f32's top 16
        # bits, which is the whole conversion.
        if self.dtype == 'bf16':
            return [struct.unpack('<f', struct.pack('<I', h << 16))[0]
                    for (h,) in struct.iter_unpack('<H', self.data)]
        if self.dtype == 'f16':
            return [struct.unpack('<e', self.data[i:i+2])[0]
                    for i in range(0, len(self.data), 2)]
        return [v for (v,) in struct.iter_unpack(_STRUCT[self.dtype], self.data)]

def _envelope(dtype, shape, data):
    return {'data': base64.b64encode(data).decode('ascii'), _TAG: _VER,
            'dtype': dtype, 'layout': _LAYOUT, 'shape': [int(d) for d in shape]}

def _int(x):
    # A JSON integer, the way the Rust reader means it. Python's `bool` is a
    # subclass of `int` and `1.0 == 1`, so a bare `== 1` accepts `true` and
    # `1.0` -- and then this reader and `unpack_raw` disagree about which
    # documents are envelopes, which PORTS.md 3.2 says they never do.
    return isinstance(x, int) and not isinstance(x, bool)

def _tensor_in(v):
    if not _int(v.get(_TAG)) or v.get(_TAG) != _VER:
        raise _ABI('tensor envelope version %r is not %d' % (v.get(_TAG), _VER))
    extra = set(v) - {_TAG, 'data', 'dtype', 'layout', 'shape'}
    if extra:
        # The manifest rule (PORTS.md 3.2) covers the fields, not just the
        # version. This side is the mirror case the section names: an envelope
        # *sent* to a runtime, refused before any foreign code runs.
        raise _ABI('tensor envelope has unknown field(s) %s'
                   % ', '.join(repr(k) for k in sorted(extra)))
    if v.get('layout') != _LAYOUT:
        raise _ABI('tensor layout %r is not %r' % (v.get('layout'), _LAYOUT))
    dt = v.get('dtype')
    if dt not in _WIDTH:
        raise _ABI('unknown tensor dtype %r' % (dt,))
    shape = v.get('shape')
    if (not isinstance(shape, list) or not shape or len(shape) > 8
            or not all(_int(d) and d > 0 for d in shape)):
        # `_int`, not `isinstance(d, int)`: `[true]` would otherwise pass here
        # and die in `reshape` instead, surfacing as `port-call` where 3.2
        # promises `port-protocol`.
        raise _ABI('tensor shape must be 1 to 8 positive integers')
    try:
        data = base64.b64decode(v.get('data') or '', validate=True)
    except Exception:
        raise _ABI('tensor data is not valid base64')
    n = 1
    for d in shape:
        n *= d
    if len(data) != n * _WIDTH[dt]:
        raise _ABI('tensor payload is %d bytes but %s%r needs %d'
                   % (len(data), dt, shape, n * _WIDTH[dt]))
    t = _Tensor(dt, shape, data)
    if _np is None:
        return t
    if dt == 'bf16':
        # numpy has no bfloat16. Widen to f32 so the callee has something it
        # can compute with, and remember that this array arrived as bf16 so
        # _dehydrate can undo the widening rather than publish it.
        #
        # The widening is a bit move, done in numpy: shift the pattern into
        # f32's high half and reinterpret. Routing it through a Python float
        # would not be the same function -- a Python float is a C double, and
        # f32 -> f64 -> f32 quiets a signaling NaN, so 126 of the 65536 bf16
        # patterns (the two sNaN bands) would come back changed. Going through
        # the buffer is also ~300x faster, which copy mode needs: this is the
        # path weight-shaped tensors cross on.
        a = (_np.frombuffer(data, dtype='<u2').astype(_np.uint32) << 16) \
            .view(_np.float32).reshape(shape)
        _ARRIVED[id(a)] = (a, dt)
        return a
    return _np.frombuffer(data, dtype=_np.dtype(_NP_READ[dt])).copy().reshape(shape)

def _bf16_bytes(a):
    # f32 -> bf16 by truncating the low 16 mantissa bits, the same narrowing
    # ports.rs `write_elem` and the JIT's `dmc_f32_to_bf16` do. It is the exact
    # inverse of the widening above, so an untouched round trip returns the
    # sender's bytes -- all 65536 of them, sNaN included.
    u = _np.ascontiguousarray(a, dtype=_np.float32).view(_np.uint32)
    return (u >> 16).astype('<u2').tobytes()

def _rehydrate(v):
    if isinstance(v, dict):
        if _TAG in v:
            return _tensor_in(v)
        return dict((k, _rehydrate(x)) for k, x in v.items())
    if isinstance(v, list):
        return [_rehydrate(x) for x in v]
    return v

def _dehydrate(v):
    if isinstance(v, _Tensor):
        return _envelope(v.dtype, v.shape, v.data)
    if _np is not None:
        if isinstance(v, _np.ndarray):
            if v.ndim == 0:
                return v.item()   # a 0-d array is a scalar, not a tensor
            came = _ARRIVED.get(id(v))
            if came is not None and came[0] is v:
                return _envelope(came[1], list(v.shape), _bf16_bytes(v))
            name = v.dtype.name
            if name not in _NP_WIRE:
                raise _ABI('numpy dtype %r has no copy-mode wire dtype' % name)
            v = _np.ascontiguousarray(v)
            return _envelope(_NP_WIRE[name], list(v.shape), v.tobytes())
        if isinstance(v, _np.generic):
            return v.item()
    if isinstance(v, dict):
        return dict((k, _dehydrate(x)) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return [_dehydrate(x) for x in v]
    return v

class _Dmc(object):
    # The `dmc.*` namespace the harness itself serves (PORTS.md 3.2). The
    # name is reserved: it is resolved here before any importable module of
    # the same name, so a copy-mode round trip needs no third-party runtime.
    @staticmethod
    def echo(x=None):
        return x
    @staticmethod
    def shape(x):
        return list(getattr(x, 'shape', []))
    @staticmethod
    def dtype(x):
        if isinstance(x, _Tensor):
            return x.dtype
        if _np is not None and isinstance(x, _np.ndarray):
            came = _ARRIVED.get(id(x))
            if came is not None and came[0] is x:
                return came[1]
            return _NP_WIRE.get(x.dtype.name, x.dtype.name)
        return type(x).__name__

def _resolve(name):
    parts = name.split('.')
    if parts[0] == 'dmc':
        obj = _Dmc
        for p in parts[1:]:
            obj = getattr(obj, p)
        return obj
    if len(parts) == 1 and hasattr(builtins, parts[0]):
        return getattr(builtins, parts[0])
    obj = importlib.import_module(parts[0])
    for p in parts[1:]:
        obj = getattr(obj, p)
    return obj
def _unpack(payload):
    # (args, kwargs) from the JSON payload per PORTS.md §2. null/empty -> no
    # args; array -> positional; object -> {args, kwargs} envelope; a bare
    # scalar is not an argument vector and is an ABI error. Tensor envelopes
    # anywhere inside are rehydrated (§3.2) before the call sees them.
    if payload in (None, ''):
        return (), {}
    try:
        p = json.loads(payload)
    except Exception as e:
        raise _ABI('payload is not valid JSON: %s' % e)
    if isinstance(p, list):
        return tuple(_rehydrate(p)), {}
    if isinstance(p, dict) and _TAG in p:
        raise _ABI('a tensor is a value, not an argument vector: '
                   'wrap the envelope in a JSON array')
    if isinstance(p, dict):
        extra = set(p) - {'args', 'kwargs'}
        if extra:
            raise _ABI('payload object must contain only "args"/"kwargs", got %s'
                       % ', '.join(sorted(extra)))
        a = p.get('args', [])
        k = p.get('kwargs', {})
        if not isinstance(a, list):
            raise _ABI('"args" must be a JSON array')
        if not isinstance(k, dict):
            raise _ABI('"kwargs" must be a JSON object')
        return tuple(_rehydrate(a)), _rehydrate(k)
    raise _ABI('payload must be a JSON array, an {args, kwargs} object, or null')
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    _ARRIVED.clear()   # arrival dtypes live for exactly one request
    try:
        req = json.loads(line)
        args, kwargs = _unpack(req.get('payload'))
        fn = _resolve(req['name'])
        out = json.dumps({'ok': _dehydrate(fn(*args, **kwargs))})
    except _ABI as e:
        out = json.dumps({'perr': str(e)})
    except Exception as e:
        out = json.dumps({'err': str(e)})
    sys.stdout.write(out + '\n')
    sys.stdout.flush()
