import argparse
import math
import random

# ==============================
# DDR5 参数
# ==============================
BURST_BYTES = 64  # DDR5 Data=Bus Width×Burst Length=16×4=64B

# ==============================
# 基本命令
# ==============================
def gen_ld(addr):
    return f"LD 0x{addr:08x}"

def gen_st(addr):
    return f"ST 0x{addr:08x}"

def burst_align(size):
    return math.ceil(size / BURST_BYTES) * BURST_BYTES


def mode_A(trace, base_addr, L, dhead, kv_heads):

    addr = base_addr

    # ---------------------------
    # K 大小（int4）
    # ---------------------------
    k_vec_bytes = dhead // 2
    k_vec_bytes = burst_align(k_vec_bytes)

    GROUP_SIZE = 16

    total_bytes = 0

    # =========================================================
    # Phase 1: 生成所有 READ（全量读取）
    # =========================================================
    read_trace = []

    total_K = kv_heads * L

    for _ in range(total_K):

        read_bytes = 0
        while read_bytes < k_vec_bytes:
            read_trace.append(gen_ld(addr))
            addr += BURST_BYTES
            read_bytes += BURST_BYTES
            total_bytes += BURST_BYTES

    # =========================================================
    # Phase 2: 生成 WRITE（每16个K → 1个bf16）
    # =========================================================
    write_trace = []

    write_addr = 1 << 30
    write_buffer = 0

    num_results = total_K // GROUP_SIZE  # 每16个K一个结果

    for _ in range(num_results):

        result_bytes = 2  # bf16
        write_buffer += result_bytes

        while write_buffer >= BURST_BYTES:
            write_trace.append(gen_st(write_addr))
            write_addr += BURST_BYTES
            write_buffer -= BURST_BYTES

    # =========================================================
    # Phase 3: READ / WRITE 穿插（仿 mode_B）
    # =========================================================
    gap = int((28 + (dhead / 16) * 4)*(BURST_BYTES/2))

    r_ptr = 0
    w_ptr = 0

    final_trace = []

    reads_per_k = k_vec_bytes // BURST_BYTES
    total_reads = len(read_trace)

    k_counter = 0  # 用于控制每16个K触发一次write机会

    while r_ptr < total_reads:

        # ---------------------------
        # 1. 发一个 K 的 read
        # ---------------------------
        for _ in range(reads_per_k):
            if r_ptr < total_reads:
                final_trace.append(read_trace[r_ptr])
                r_ptr += 1

        k_counter += 1

        # ---------------------------
        # 2. gap（用 read 流自然填充）
        # ---------------------------
        for _ in range(gap):
            if r_ptr < total_reads:
                final_trace.append(read_trace[r_ptr])
                r_ptr += 1

        # ---------------------------
        # 3. 每16个K尝试发一个 write
        # ---------------------------
        if k_counter % GROUP_SIZE == 0:
            if w_ptr < len(write_trace):
                final_trace.append(write_trace[w_ptr])
                w_ptr += 1

    # 剩余 write 补齐
    while w_ptr < len(write_trace):
        final_trace.append(write_trace[w_ptr])
        w_ptr += 1

    trace.extend(final_trace)

    return addr, total_bytes


# =========================================================
# 模式 B：读 K + 写回 QK（支持 GQA）
# =========================================================
def mode_B(trace, base_addr, L, dhead, kv_heads, gqa_ratio):
    addr = base_addr

    # ---------------------------
    # K 大小（int4）
    # ---------------------------
    k_vec_bytes = dhead // 2
    k_vec_bytes = burst_align(k_vec_bytes)

    GROUP_SIZE = 16

    total_bytes = 0

    # =========================================================
    # Phase 1: 生成所有 READ
    # =========================================================
    read_trace = []

    for h in range(kv_heads):

        num_groups = (L + GROUP_SIZE - 1) // GROUP_SIZE
        groups = list(range(num_groups))

        selected_groups = set(
            random.sample(groups, max(1, num_groups // 4))
        )

        for g in groups:

            if g not in selected_groups:
                addr += GROUP_SIZE * k_vec_bytes
                continue

            for i in range(GROUP_SIZE):

                global_idx = g * GROUP_SIZE + i
                if global_idx >= L:
                    break

                read_bytes = 0
                while read_bytes < k_vec_bytes:
                    read_trace.append(gen_ld(addr))
                    addr += BURST_BYTES
                    read_bytes += BURST_BYTES
                    total_bytes += BURST_BYTES

    # =========================================================
    # Phase 2: 生成所有 WRITE（带 burst 聚合）
    # =========================================================
    write_trace = []

    write_addr = 1 << 30
    write_buffer = 0

    # 每个 K 产生一次结果
    total_selected_K = len(read_trace) * BURST_BYTES // k_vec_bytes

    for _ in range(total_selected_K):

        result_bytes = 2 * gqa_ratio
        write_buffer += result_bytes

        while write_buffer >= BURST_BYTES:
            write_trace.append(gen_st(write_addr))
            write_addr += BURST_BYTES
            write_buffer -= BURST_BYTES

    # =========================================================
    # Phase 3: 穿插（核心）
    # =========================================================
    # 每读一个 K 后，间隔 gap 插入 write（如果有）
    gap = int((29 + (dhead / 16) * 4)*(BURST_BYTES/2))

    r_ptr = 0
    w_ptr = 0

    final_trace = []

    # 每个 K 对应多少条 read 指令
    reads_per_k = k_vec_bytes // BURST_BYTES

    total_reads = len(read_trace)

    while r_ptr < total_reads:

        # ---------------------------
        # 1. 发一个 K 的 read
        # ---------------------------
        for _ in range(reads_per_k):
            if r_ptr < total_reads:
                final_trace.append(read_trace[r_ptr])
                r_ptr += 1

        # ---------------------------
        # 2. 插入 gap 间隔（但不再用 dummy）
        #    → 用 read 流本身来“自然间隔”
        # ---------------------------
        gap_reads = gap

        for _ in range(gap_reads):
            if r_ptr < total_reads:
                final_trace.append(read_trace[r_ptr])
                r_ptr += 1

        # ---------------------------
        # 3. 插入一个 write（如果有）
        # ---------------------------
        if w_ptr < len(write_trace):
            final_trace.append(write_trace[w_ptr])
            w_ptr += 1

    # 剩余 write 全部补上
    while w_ptr < len(write_trace):
        final_trace.append(write_trace[w_ptr])
        w_ptr += 1

    trace.extend(final_trace)

    return addr, total_bytes

# =========================================================
# 模式 C：读 attention（QK结果）
# =========================================================
def mode_C(trace, base_addr, L, kv_heads, gqa_ratio):
    read_bytes = kv_heads * L * gqa_ratio * 2 

    addr = base_addr

    total_bytes = 0
    while read_bytes > 0:
        trace.append(gen_ld(addr))
        addr += BURST_BYTES
        read_bytes -= BURST_BYTES
        total_bytes += BURST_BYTES

    return addr, total_bytes

# =========================================================
# 模式 D：读 attention（QK结果）每个头随机分散读 0.02L
# =========================================================
def mode_D(trace, base_addr, L, kv_heads, gqa_ratio):
    head_span_bytes = L * 2

    num_reads_per_head = int(0.02 * L)
    
    if num_reads_per_head < 1:
        num_reads_per_head = 1
        
    current_addr = base_addr

    total_bytes = 0

    for h in range(kv_heads * gqa_ratio):
        head_base = base_addr + h * head_span_bytes
        for _ in range(num_reads_per_head):
            max_offset = max(0, head_span_bytes - BURST_BYTES)
        
            random_offset = random.randint(0, max_offset)
            
            aligned_offset = (random_offset // BURST_BYTES) * BURST_BYTES
            
            read_addr = head_base + aligned_offset

            trace.append(gen_ld(read_addr))
        
            current_addr = read_addr + BURST_BYTES
            total_bytes += 2

    return current_addr, total_bytes


# =========================================================
# 主流程
# =========================================================
def run(dhead, kv_heads, nq_heads, L, mode, output, ddr_num):
    trace = []

    base_addr = 0

    # print("ddr_num:",ddr_num)
    # print("nq_heads:",nq_heads)
    # print("kv_heads:",kv_heads)
    nq_heads = nq_heads // ddr_num
    kv_heads = max(1, kv_heads // ddr_num)
    # print("nq_heads:",nq_heads)
    # print("kv_heads:",kv_heads)

    # GQA ratio
    assert nq_heads % kv_heads == 0
    gqa_ratio = nq_heads // kv_heads

    if mode == "A":
        base_addr, total_bytes = mode_A(trace, base_addr, L, dhead, kv_heads)

    elif mode == "B":
        base_addr, total_bytes = mode_B(trace, base_addr, L, dhead, kv_heads, gqa_ratio)

    elif mode == "C":
        base_addr, total_bytes = mode_C(trace, base_addr, L, kv_heads, gqa_ratio)

    elif mode == "D":
        base_addr, total_bytes = mode_D(trace, base_addr, L, kv_heads, gqa_ratio)

    else:
        raise ValueError("Unknown mode")
    
    total_bytes = total_bytes * max(1, (nq_heads/kv_heads) / 8)
    
    # print("total_bytes:",total_bytes)

    # 写文件
    with open(output, "w") as f:
        for cmd in trace:
            f.write(cmd + "\n")

    print("===================================")
    print("DDR5 Gen Attention Trace (GQA)")
    print("===================================")
    print(f"dhead        : {dhead}")
    print(f"kv_heads     : {kv_heads}")
    print(f"q_heads      : {nq_heads}")
    print(f"GQA ratio    : {gqa_ratio}")
    print(f"seqlen (L)   : {L}")
    print(f"mode         : {mode}")
    print(f"Total cmds   : {len(trace)}")
    print(f"Output       : {output}")
    print(f"Total bytes  : {total_bytes}")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="DDR5 Trace Generator for LLM Attention (Gen + GQA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-dh", "--dhead", type=int, default=128)

    parser.add_argument("-kvh", "--kv_heads", type=int, default=8,
                        help="KV heads (memory relevant)")

    parser.add_argument("-qh", "--q_heads", type=int, default=32,
                        help="Query heads")

    parser.add_argument("-l", "--seqlen", type=int, default=2048)

    parser.add_argument("-m", "--mode", type=str,
                        choices=["A", "B", "C", "D"], default="A")

    parser.add_argument("-o", "--output", type=str,
                        default="./test.trace")
    
    parser.add_argument(
        "--ddr-num",
        type=int,
        default=4,
        help=
        "number of DDR, default = 4"
    )

    args = parser.parse_args()

    run(
        dhead=args.dhead,
        kv_heads=args.kv_heads,
        nq_heads=args.q_heads,
        L=args.seqlen,
        mode=args.mode,
        output=args.output,
        ddr_num=args.ddr_num
    )


if __name__ == "__main__":
    main()