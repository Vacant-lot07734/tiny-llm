import mlx.core as mx

a = mx.array([1, 2, 3])
print(5 / 2)
print(5 // 2)
print(a)
print(mx.rsqrt(a))
a = mx.arange(0, 12).reshape(2, 3, 2)
print(a, a.shape)
b = a[..., 1]
print(b, b.shape)
a = mx.array([[0, 0, 0], [0, 0, 0]])
b = mx.array([[1, 1, 1], [1, 1, 1]])
# 会增加一个维度，沿着新维度堆叠
c = mx.stack([a, b], axis=-1)
# 沿着指定维度拼接
d = mx.concat([a, b], axis=-1)
print(c, c.shape)
print(d, d.shape)
