import math

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.utils import prepare_chunk_indices


@triton.jit
def causal_mask_fn(q_start, kv_start, Q_BLOCK, KV_BLOCK):
    # pure arithmetic causal mask: kv_pos <= q_pos (no aux_mask lookup table)
    q_pos = q_start + tl.arange(0, Q_BLOCK)[:, None]
    kv_pos = kv_start + tl.arange(0, KV_BLOCK)[None, :]
    mask_causal = kv_pos <= q_pos

    return mask_causal


@triton.jit
def _sdpa_infer_single_block(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_T_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    qk_scale,
    mask,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    # -- Compute qk ----

    # Load (transposed) K block
    k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.dot(q, k_T)
    # tl.compile_hint(qk, "tile_cube_loop")

    qk = qk * qk_scale
    if mask is not None:
        qk = tl.where(mask, qk, float("-inf"))  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
    qk = qk - m_ij[:, None]  # Stabilize

    # Softmax weights p = exp(qk)
    p = tl.math.exp(qk)

    p_cast = p.to(k_T.dtype)

    # Load corresponding V block
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Softmax denominator (sum of each row)
    l_ij = tl.sum(p, 1)
    # -- Update m_i and l_i
    alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha[:, None]
    acc_ptr = tl.dot(p_cast, v, acc_ptr)
    # tl.compile_hint(acc_ptr, "tile_cube_loop")

    # Update current block max
    m_i = m_ij

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


@triton.jit
def paged_prefill_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    batch_size,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    softmax_scale,
    PAGE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")

    prev_q_chunks = 0

    for b_id in range(batch_size):
        q_start_loc = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end_loc = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end_loc - q_start_loc

        if seqlens_kv_ptr is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = tl.load(seqlens_kv_ptr + b_id)
        kv_cache_len = kv_seq_len - q_seq_len

        cur_q_chunks = tl.cdiv(q_seq_len, BLOCK_SIZE_M)
        cur_q_tasks = cur_q_chunks * NUM_Q_HEADS
        prev_q_tasks = prev_q_chunks * NUM_Q_HEADS
        prev_q_chunks += cur_q_chunks
        for q_task_id in range((prev_q_tasks + pid) % n_progs, cur_q_tasks, n_progs):
            q_block_id = q_task_id // NUM_Q_HEADS
            q_head_id = q_task_id % NUM_Q_HEADS

            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_block_start_in_seq = q_block_id * BLOCK_SIZE_M
            q_block_end_in_seq = min(q_block_start_in_seq + BLOCK_SIZE_M, q_seq_len)
            q_block_len = q_block_end_in_seq - q_block_start_in_seq

            Q_block_ptr = tl.make_block_ptr(
                base=q_ptr + (q_start_loc + q_block_start_in_seq) * stride_qt + q_head_id * stride_qh,
                shape=(q_block_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
                order=(1, 0),
            )
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr + (q_start_loc + q_block_start_in_seq) * stride_ot + q_head_id * stride_oh,
                shape=(q_block_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
                order=(1, 0),
            )

            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

            num_kv_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, BLOCK_SIZE_N)

            for kv_block_id in range(0, num_kv_blocks):
                kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
                kv_block_end_in_seq = min(kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

                logical_page_id = kv_block_start_in_seq // PAGE_SIZE
                kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
                physical_page_id = tl.load(
                    block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                )

                K_T_block_ptr = tl.make_block_ptr(
                    base=key_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_k_dim, stride_k_blksz),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                    order=(0, 1),
                )
                V_block_ptr = tl.make_block_ptr(
                    base=value_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )

                mask = causal_mask_fn(
                    kv_cache_len + q_block_start_in_seq,
                    kv_block_start_in_seq,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                )

                acc, l_i, m_i = _sdpa_infer_single_block(
                    acc,
                    l_i,
                    m_i,
                    q,
                    K_T_block_ptr,
                    V_block_ptr,
                    softmax_scale,
                    mask,
                    HEAD_DIM,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_D,
                    value_cache_ptr.dtype.element_ty == tl.float8e5,
                )

            tl.store(O_block_ptr, (acc / l_i[:, None]).to(o_ptr.type.element_ty), boundary_check=(0, 1))


def paged_attention_prefill_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    seqlens_kv: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
    max_q_len: Optional[int] = None,
    max_total_seq_len: Optional[int] = None,
) -> torch.Tensor:
    _, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = key_cache.shape
    batch_size = cu_q_lens.shape[0] - 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Note(chenyifan):
    #   In general, this paged attention kernel works in a `split-q` style.
    #   "bsz * query * q_head" is splited into tasks of shape [BLOCK_SIZE_M, HEAD_DIM]
    #   and then attributed to one program.
    #
    #   Currently, we chunk the queries manually according to a magic CHUNK_SIZE to split queries
    #   It should be better with a autotuned BLOCK_SIZE_M and a pre-configured max_seq_len

    o = torch.empty_like(q)

    # Safe shape-adaptive BLOCK_SIZE_M: use 64 for very short Q sequences to
    # reduce padding waste; keep 128 (proven safe) for all other shapes.
    # BLOCK_SIZE_M=256 triggered MLIR compilation failure, so we cap at 128.
    # causal mask is computed arithmetically in-kernel, so no aux_mask tensor.
    if max_q_len is not None and max_q_len > 0:
        chunk_size = 64 if max_q_len <= 256 else 128
    else:
        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        max_q = q_lens.max().item() if q_lens.numel() > 0 else 0
        chunk_size = 64 if max_q <= 256 else 128
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))
    cube_num = get_num_cores("cube")
    grid = (cube_num,)

    paged_prefill_kernel[grid](
        q,
        key_cache,
        value_cache,
        o,
        batch_size,
        cu_q_lens,
        seqlens_kv,
        block_tables.to(torch.int32),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        softmax_scale,
        page_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_SIZE_M=chunk_size,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=head_dim,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        set_workspace_multibuffer=4,
    )
    return o


# ============================================================================
# Paged Decode (Flash Attention - Decode path)
#
# Two-stage flash-decoding split / merge:
# - Stage1 partial: grid (num_tasks, num_splits), each program computes a
#   segment of KV's partial (m_i, l_i, acc) without final normalization; empty
#   splits keep the identity element (m=-inf, l=0, acc=0).
# - Stage2 merge: each task merges num_splits partials using the associativity
#   of online-softmax; for all-empty sequences (m_global == -inf) write 0
#   directly, avoiding exp(-inf - -inf) = NaN and 0/0 NaN.
# Full fp32 accumulation.
# ============================================================================


@triton.jit
def paged_decode_partial_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    partial_ptr,  # [num_tasks, num_splits, HEAD_DIM + 2] fp32  (or o_ptr when WRITE_OUTPUT)
    seqlens_ptr,
    block_tables_ptr,
    BATCH_SIZE,
    NUM_SPLITS,
    MAX_NUM_BLOCKS_PER_SEQ,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_pt_task,
    stride_pt_split,
    stride_pt_entry,
    stride_bt_batch,
    stride_bt_block,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # q-heads per kv-head group (Hq // Hkv)
    GROUP_M: tl.constexpr,     # padded power-of-two >= GROUP_SIZE, M dim of tl.dot
    GQA_INTERLEAVE: tl.constexpr,  # ABAB if True, AABB otherwise
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr,  # write acc/l_i straight to o, skip merge
    stride_ob,
    stride_oh,
    stride_od,
):
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be less than BLOCK_SIZE_D")
    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")

    # Group one batch's q-heads into a (GROUP_M, D) block and use tl.dot (cube
    # engine) for QK and PV. Each program = (batch, split); it loops the Hkv
    # groups. The whole acc tile is stored with a row mask.
    b_id = tl.program_id(0)
    split_id = tl.program_id(1)

    kv_seq_len = tl.load(seqlens_ptr + b_id)

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_m = tl.arange(0, GROUP_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    q_row_valid = offs_m < GROUP_SIZE

    num_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_SIZE_N)
    blocks_per_split = tl.cdiv(num_kv_blocks, NUM_SPLITS)
    start_block = split_id * blocks_per_split
    end_block = tl.minimum((split_id + 1) * blocks_per_split, num_kv_blocks)

    for kv_head_id in range(0, NUM_KV_HEADS):
        # q-head ids covered by this kv-head group, indexed by offs_m. Layout:
        #   ABAB (GQA_INTERLEAVE): kv_head_id + g * NUM_KV_HEADS
        #   AABB (contiguous)   : kv_head_id * GROUP_SIZE + g
        # padded rows (offs_m >= GROUP_SIZE) produce out-of-range ids but are
        # masked away by q_row_valid, so they never read/write real memory.
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + offs_m * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + offs_m
        task_ids = b_id * NUM_Q_HEADS + q_head_ids

        # load Q block (GROUP_M, HEAD_DIM); padded rows = 0
        q_ptrs = q_ptr + b_id * stride_qb + q_head_ids[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=q_row_valid[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0)

        m_i = tl.full((GROUP_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((GROUP_M,), dtype=tl.float32)
        acc = tl.zeros((GROUP_M, BLOCK_SIZE_D), dtype=tl.float32)

        # software-pipelining hint so the next K/V load overlaps the current
        # softmax+PV (compiler-issued prefetch), hiding DRAM latency on the
        # long-context cases. num_stages=2 double-buffers K/V.
        for kv_block_id in tl.range(start_block, end_block, num_stages=2):
            kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
            kv_block_end_in_seq = tl.minimum(kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

            logical_page_id = kv_block_start_in_seq // PAGE_SIZE
            kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
            physical_page_id = tl.load(block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block)

            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask_n = offs_n < kv_block_len

            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # QK = Q (GROUP_M, D) @ K^T (D, N) -> (GROUP_M, N) via cube tl.dot.
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
            qk = tl.where(mask_n[None, :], qk, float("-inf"))

            m_j = tl.max(qk, axis=1)
            m_ij = tl.maximum(m_i, m_j)
            qk = qk - m_ij[:, None]

            p = tl.math.exp(qk)

            v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

            l_ij = tl.sum(p, axis=1)

            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij

            acc = acc * alpha[:, None]

            # PV = P (GROUP_M, N) @ V (N, D) -> (GROUP_M, D) via cube tl.dot,
            # accumulated into acc in fp32.
            acc = tl.dot(p.to(v.dtype), v, acc=acc)

            m_i = m_ij

        # when WRITE_OUTPUT, this program is the only split, so the partial IS
        # the final answer up to the divide — write acc/l_i straight to o and
        # skip the merge kernel. Otherwise store partials.
        if WRITE_OUTPUT:
            is_empty = m_i == -float("inf")
            l_safe = tl.where(l_i > 0.0, l_i, 1.0)
            out = acc / l_safe[:, None]
            out = tl.where(is_empty[:, None], 0.0, out)
            o_ptrs = (
                partial_ptr
                + b_id * stride_ob
                + q_head_ids[:, None] * stride_oh
                + offs_d[None, :] * stride_od
            )
            tl.store(o_ptrs, out.to(partial_ptr.dtype.element_ty), mask=q_row_valid[:, None] & (offs_d[None, :] < HEAD_DIM))
        else:
            partial_base = partial_ptr + split_id * stride_pt_split
            acc_ptrs = (
                partial_base
                + task_ids[:, None] * stride_pt_task
                + offs_d[None, :] * stride_pt_entry
            )
            tl.store(acc_ptrs, acc, mask=q_row_valid[:, None] & (offs_d[None, :] < HEAD_DIM))
            lm_ptrs = partial_base + task_ids * stride_pt_task
            tl.store(lm_ptrs + HEAD_DIM * stride_pt_entry, l_i, mask=q_row_valid)
            tl.store(lm_ptrs + (HEAD_DIM + 1) * stride_pt_entry, m_i, mask=q_row_valid)


@triton.jit
def paged_decode_merge_kernel(
    partial_ptr,  # [num_tasks, num_splits, HEAD_DIM + 2] fp32
    o_ptr,
    NUM_SPLITS,
    stride_ob,
    stride_oh,
    stride_od,
    stride_pt_task,
    stride_pt_split,
    stride_pt_entry,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    task_id = tl.program_id(0)

    b_id = task_id // NUM_Q_HEADS
    q_head_id = task_id % NUM_Q_HEADS

    offs_d = tl.arange(0, BLOCK_SIZE_D)

    # first pass: global max m (== -inf iff every split is empty, i.e. seq_len==0)
    m_global = -float("inf")
    for s in range(0, NUM_SPLITS):
        m_s = tl.load(partial_ptr + task_id * stride_pt_task + s * stride_pt_split + (HEAD_DIM + 1) * stride_pt_entry)
        m_global = tl.maximum(m_global, m_s)

    is_empty = m_global == -float("inf")

    # second pass: accumulate l and acc with rescale. Replace m_s == -inf
    # (empty split) with m_global so m_s - m_global == 0 instead of NaN;
    # those splits contribute l_s=0 and acc_s=0 so the rescale is harmless.
    l_global = 0.0
    acc_global = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    for s in range(0, NUM_SPLITS):
        m_s = tl.load(partial_ptr + task_id * stride_pt_task + s * stride_pt_split + (HEAD_DIM + 1) * stride_pt_entry)
        m_s_safe = tl.where(m_s == -float("inf"), m_global, m_s)
        l_s = tl.load(partial_ptr + task_id * stride_pt_task + s * stride_pt_split + HEAD_DIM * stride_pt_entry)
        acc_s = tl.load(
            partial_ptr + task_id * stride_pt_task + s * stride_pt_split + offs_d * stride_pt_entry,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        )
        alpha = tl.math.exp(m_s_safe - m_global)
        l_global += l_s * alpha
        acc_global += acc_s * alpha

    # guard division: l_global is 0 only when is_empty; replace with 1.0 to
    # avoid 0/0 = NaN, then overwrite the whole output with 0 for empty tasks.
    l_safe = tl.where(l_global > 0.0, l_global, 1.0)
    out = acc_global / l_safe
    out = tl.where(is_empty, 0.0, out)

    o_ptrs = o_ptr + b_id * stride_ob + q_head_id * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=offs_d < HEAD_DIM)


def paged_attention_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, page_size, head_dim_cache = key_cache.shape

    max_num_blocks_per_seq = block_tables.shape[1]

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)

    num_tasks = batch_size * num_q_heads
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    # page-aligned, UB-safe KV tile: largest power-of-two divisor of PAGE_SIZE
    # capped at 128 (page=1024 with 256 overflows UB).
    cap = 128
    bsn = cap
    while page_size % bsn != 0:
        bsn //= 2
    BLOCK_SIZE_N = bsn

    # split count: parallelize the KV reduction across programs; only split
    # when the longest sequence actually has many KV blocks so short-seq
    # shapes degenerate to NUM_SPLITS=1.
    max_seq_len = int(seqlens.max().item())
    max_kv_blocks = (max_seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    max_kv_blocks = max(max_kv_blocks, 1)
    NUM_SPLITS_CAP = 8
    num_splits = max(min(NUM_SPLITS_CAP, max_kv_blocks), 1)

    # partial buffer: [num_tasks, num_splits, head_dim + 2] fp32
    # last 2 entries per (task,split): [l_i, m_i]
    partial = torch.empty(
        (num_tasks, num_splits, head_dim + 2),
        dtype=torch.float32,
        device=q.device,
    )

    # group q-heads per kv-head. GROUP_M is a power-of-two >= GROUP_SIZE and
    # >= 8 so tl.dot's M dim is valid for the cube engine.
    group_size = num_q_heads // num_kv_heads
    group_m = triton.next_power_of_2(max(group_size, 8))

    # when num_splits == 1 the partial is the final answer up to the divide;
    # dispatch the partial kernel with WRITE_OUTPUT=True straight into o and
    # skip the merge kernel (and the partial buffer alloc).
    single_kernel = (num_splits == 1)

    grid_partial = (batch_size, num_splits)
    paged_decode_partial_kernel[grid_partial](
        q,
        key_cache,
        value_cache,
        o if single_kernel else partial,
        seqlens,
        block_tables,
        batch_size,
        num_splits,
        max_num_blocks_per_seq,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        partial.stride(0) if not single_kernel else o.stride(0),
        partial.stride(1) if not single_kernel else o.stride(1),
        partial.stride(2) if not single_kernel else o.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        softmax_scale,
        num_q_heads,
        num_kv_heads,
        group_size,
        group_m,
        gqa_interleave,
        head_dim,
        page_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        WRITE_OUTPUT=single_kernel,
        stride_ob=o.stride(0),
        stride_oh=o.stride(1),
        stride_od=o.stride(2),
        multibuffer=False,
    )

    if not single_kernel:
        grid_merge = (num_tasks,)
        paged_decode_merge_kernel[grid_merge](
            partial,
            o,
            num_splits,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            num_q_heads,
            head_dim,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
    return o
