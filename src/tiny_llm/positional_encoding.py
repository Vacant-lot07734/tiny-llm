import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len
        half_dims = dims // 2
        inner = mx.arange(0, half_dims, dtype=mx.float32) / half_dims
        freqs = mx.power(base, -inner)
        t = mx.arange(seq_len)
        # 计算两个一维tensor的外积
        # 相当于n×1的矩阵和1×m的矩阵相乘
        # 结果的每一行是 对应token位置的 分量 cos 或 sin 的参数
        freqs = mx.outer(t, freqs)
        # (S, half_dims)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)
        self.base = base
        self.half_dims = half_dims
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape
        if offset is not None:
            if isinstance(offset, slice):
                assert offset.stop - offset.start == S, f"offset must be of length {S}"
            elif isinstance(offset, list):
                assert (
                    len(offset) == N
                ), f"offsets must have the same length as batch size {N}"
                for o in offset:
                    assert o.stop - o.start == S, f"offset must be of length {S}"
                offset = mx.array([list(range(i.start, i.stop)) for i in offset])
        cos_basis = (
            self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset, :]
        )
        sin_basis = (
            self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset, :]
        )
        if self.traditional:
            # reshape x : (b, s, n_heads, head_dim // 2, 2)
            x = x.reshape(N, S, H, self.half_dims, 2)
            # shape (b, s, n_heads, head_dim // 2)
            x1 = x[..., 0]  # 分量偶数位置
            x2 = x[..., 1]  # 奇数位置
        else:
            # shape (b, s, n_heads, head_dim // 2)
            x1 = x[..., 0:self.half_dims]
            x2 = x[..., self.half_dims:]
        # (s, dims) reshape basis : (1, s, 1, dims // 2)
        cos_basis = cos_basis.reshape(-1, S, 1, self.half_dims)
        sin_basis = sin_basis.reshape(-1, S, 1, self.half_dims)
        # manually doing complex number multiplication (broadcast element-wise multiply)
        real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
        imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)
        if self.traditional:
            # real.shape: (N, S, H, self.half_dims)
            # imag.shape: (N, S, H, self.half_dims)
            # mx.stack 在新的最后一维 (-1) 上堆叠 real 和 imag
            # y_stacked.shape: (N, S, H, self.half_dims, 2)
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        else:
            y = mx.concat([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        return y.astype(x.dtype)
