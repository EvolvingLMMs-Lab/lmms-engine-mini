import ctypes

import bitsandbytes as bnb

# torch.cuda.amp.custom_fwd is deprecated >= 2.4
import torch
import triton

# tl.math.tanh now is libdevice.tanh
from packaging.version import Version

MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2


if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")
pass


if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice

    triton_tanh = libdevice.tanh
else:
    import triton.language as tl

    triton_tanh = tl.math.tanh
pass


def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


pass


get_ptr = bnb.functional.get_ptr

cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4 = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4 = bnb.functional.lib.cdequantize_blockwise_bf16_nf4
cgemm_4bit_inference_naive_fp16 = bnb.functional.lib.cgemm_4bit_inference_naive_fp16
cgemm_4bit_inference_naive_bf16 = bnb.functional.lib.cgemm_4bit_inference_naive_bf16


def QUANT_STATE(W):
    return getattr(W, "quant_state", None)


pass


def get_lora_parameters(proj):
    # For DPO or disabled adapters
    base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
    W = base_layer.weight

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None
    pass

    active_adapter = (
        proj.active_adapters[0]
        if hasattr(proj, "active_adapters")
        else proj.active_adapter
    )
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s


pass


def get_lora_parameters_bias(proj):
    # For DPO or disabled adapters
    base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
    W = base_layer.weight
    bias = base_layer.bias

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None, bias
    pass

    active_adapter = (
        proj.active_adapters[0]
        if hasattr(proj, "active_adapters")
        else proj.active_adapter
    )
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s, bias


pass


def fast_dequantize(W, quant_state=None, out=None):
    if quant_state is None:
        return W
    if type(quant_state) is not list:
        # New quant_state as a class
        # https://github.com/TimDettmers/bitsandbytes/pull/763/files
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    else:
        # Old quant_state as a list of lists
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2
    pass

    # Create weight matrix
    if out is None:
        out = torch.empty(shape, dtype=dtype, device="cuda:0")
    else:
        assert out.shape == shape
        assert out.dtype == dtype

    # NF4 dequantization of statistics
    n_elements_absmax = absmax.numel()
    out_absmax = torch.empty(n_elements_absmax, dtype=torch.float32, device="cuda:0")

    # Do dequantization
    ptr_out_absmax = get_ptr(out_absmax)
    cdequantize_blockwise_fp32(
        get_ptr(code2),
        get_ptr(absmax),
        get_ptr(absmax2),
        ptr_out_absmax,
        ctypes.c_int(blocksize2),
        ctypes.c_int(n_elements_absmax),
    )
    out_absmax += offset

    fx = (
        cdequantize_blockwise_fp16_nf4
        if dtype == torch.float16
        else cdequantize_blockwise_bf16_nf4
    )
    fx(
        get_ptr(None),
        get_ptr(W),
        ptr_out_absmax,
        get_ptr(out),
        ctypes.c_int(blocksize),
        ctypes.c_int(out.numel()),
    )

    # Careful returning transposed data
    is_transposed = True if W.shape[0] == 1 else False
    return out.t() if is_transposed else out


pass


def fast_gemv(X, W, quant_state, out=None):
    if quant_state is None:
        return torch.matmul(X, W, out=out)
    # For fast X @ W where seq_len == 1
    # From https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L1469
    _, q_len, hd = X.shape
    # assert(q_len == 1)

    if type(quant_state) is not list:
        # https://github.com/TimDettmers/bitsandbytes/pull/763/files
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        stats = quant_state.code
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    else:
        (
            absmax,
            shape,
            dtype,
            blocksize,
            compressed_stats,
            quant_type,
            stats,
        ) = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2
    pass
    # assert(dtype == X.dtype)
    bout = shape[0]

    if out is None:
        out = torch.empty(
            (
                1,
                1,
                bout,
            ),
            dtype=dtype,
            device="cuda:0",
        )
    # else:
    #     assert(out.shape == (1, 1, bout,))
    # pass

    n = 1
    m = shape[0]
    k = shape[1]
    lda = shape[0]
    ldc = shape[0]
    ldb = (hd + 1) // 2
    m = ctypes.c_int32(m)
    n = ctypes.c_int32(n)
    k = ctypes.c_int32(k)
    lda = ctypes.c_int32(lda)
    ldb = ctypes.c_int32(ldb)
    ldc = ctypes.c_int32(ldc)

    df = torch.empty(absmax.shape, dtype=torch.float32, device="cuda:0")
    cdequantize_blockwise_fp32(
        get_ptr(code2),
        get_ptr(absmax),
        get_ptr(absmax2),
        get_ptr(df),
        ctypes.c_int(blocksize2),
        ctypes.c_int(df.numel()),
    )
    df += offset
    absmax = df

    fx = (
        cgemm_4bit_inference_naive_fp16
        if dtype == torch.float16
        else cgemm_4bit_inference_naive_bf16
    )

    blocksize = ctypes.c_int32(blocksize)
    fx(
        m,
        n,
        k,
        get_ptr(X),
        get_ptr(W),
        get_ptr(absmax),
        get_ptr(stats),
        get_ptr(out),
        lda,
        ldb,
        ldc,
        blocksize,
    )

    return out


pass


def fast_linear_forward(proj, X, temp_lora=None, out=None):
    W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
    bsz, q_len, in_dim = X.shape
    if q_len != 1:
        return matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)

    if W_quant is None:
        out = torch.matmul(X, W.t(), out=out)
    elif bsz == 1 and q_len == 1:
        out = fast_gemv(X, W, W_quant, out=out)
    else:
        W = fast_dequantize(W.t(), W_quant)
        out = torch.matmul(X, W, out=out)
    pass

    # Add in LoRA weights
    if lora_A is not None:
        out_dim = out.shape[2]
        dtype = X.dtype

        if not hasattr(lora_A, "_fast_lora"):
            lora_A._fast_lora = lora_A.to(dtype)
            lora_B._fast_lora = lora_B.to(dtype)
        pass

        if bsz == 1:
            out = out.view(out_dim)
            temp_lora = torch.mv(lora_A._fast_lora, X.ravel(), out=temp_lora)
            out.addmv_(lora_B._fast_lora, temp_lora, alpha=lora_S)
        else:
            out = out.view(bsz, out_dim)
            temp_lora = torch.mm(
                X.view(bsz, in_dim), lora_A._fast_lora.t(), out=temp_lora
            )
            out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha=lora_S)
        pass
        out = out.view(bsz, 1, out_dim)
    pass

    if bias is not None:
        out += bias

    return out


pass


def matmul_lora(X, W, W_quant, A, B, s, out=None):
    dtype = X.dtype
    W = fast_dequantize(W.t(), W_quant)

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False
    pass

    out = torch.matmul(X, W, out=out)
    if W_quant is not None:
        del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        out += (X @ A.to(dtype)) @ (s * B.to(dtype))
    pass

    return out.view(batch, seq_len, -1) if reshape else out


def flops_per_token(config):
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    intermediate_size = config.intermediate_size
    kv_heads = config.num_key_value_heads
    attn_heads = config.num_attention_heads
    total_params = 0.0  # initilize total_params as float, to avoid overflow of 'flops' when using 512 gpus
    # head, mebedding not considered
    total_params += vocab_size * hidden_size
    # transformers
    params_per_block = 2 * hidden_size * hidden_size
    params_per_block += 4 * hidden_size * hidden_size * kv_heads // attn_heads
    params_per_block += 3 * hidden_size * intermediate_size
    total_params += params_per_block * num_hidden_layers

    flops = 6 * total_params
    return flops


def calc_gpt_flops(attention_mask, config):
    tokens_count = torch.sum(attention_mask != 0).item()
    flops = flops_per_token(config) * tokens_count
    token_count_list = torch.sum(attention_mask != 0, dim=1).tolist()
    for seq_len in token_count_list:
        flops += 12 * seq_len * seq_len * config.num_hidden_layers * config.hidden_size
    return flops
