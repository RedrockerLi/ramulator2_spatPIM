import argparse
import math

# ==============================
# DDR5 参数
# ==============================
BURST_BYTES = 32  # DDR5 BL16 x16

# ==============================
# 基本命令
# ==============================
def gen_ld(addr):
    return f"LD 0x{addr:08x}"

def gen_st(addr):
    return f"ST 0x{addr:08x}"

def burst_align(size):
    return math.ceil(size / BURST_BYTES) * BURST_BYTES


# =========================================================
# 模式 A：读 K cache（int4）
# =========================================================
def mode_A(trace, base_addr, L, dhead, kv_heads):
    total_bytes = kv_heads * L * dhead // 2  # int4

    addr = base_addr
    end = base_addr + total_bytes

    while addr < end:
        trace.append(gen_ld(addr))
        addr += BURST_BYTES

    return addr


# =========================================================
# 模式 B：读 K + 写回 QK（支持 GQA）
# =========================================================
def mode_B(trace, base_addr, L, dhead, kv_heads, gqa_ratio):
    addr = base_addr

    k_vec_bytes = dhead // 2
    k_vec_bytes = burst_align(k_vec_bytes)

    write_buffer = 0
    write_addr = 1 << 30

    gap = int(27 + (dhead / 16) * 4)

    for h in range(kv_heads):
        for i in range(L):

            # ---------------------------
            # 读一个 K
            # ---------------------------
            read_bytes = 0
            while read_bytes < k_vec_bytes:
                trace.append(gen_ld(addr))
                addr += BURST_BYTES
                read_bytes += BURST_BYTES

            # ---------------------------
            # compute gap
            # ---------------------------
            for _ in range(gap):
                trace.append(gen_ld(addr))  # 占位
                addr += BURST_BYTES

            # ---------------------------
            # 写回 gqa_ratio 个 score
            # ---------------------------
            result_bytes = 2 * gqa_ratio  # ⭐关键

            write_buffer += result_bytes

            while write_buffer >= BURST_BYTES:
                trace.append(gen_st(write_addr))
                write_addr += BURST_BYTES
                write_buffer -= BURST_BYTES

    return addr


# =========================================================
# 模式 C：读 attention（QK结果）
# =========================================================
def mode_C(trace, base_addr, L, kv_heads, gqa_ratio):
    total_bytes = kv_heads * L * gqa_ratio * 2  # ⭐关键

    addr = base_addr

    while total_bytes > 0:
        trace.append(gen_ld(addr))
        addr += BURST_BYTES
        total_bytes -= BURST_BYTES

    return addr


# =========================================================
# 主流程
# =========================================================
def run(dhead, kv_heads, nq_heads, L, mode, output):
    trace = []

    base_addr = 0

    # GQA ratio
    assert nq_heads % kv_heads == 0
    gqa_ratio = nq_heads // kv_heads

    if mode == "A":
        base_addr = mode_A(trace, base_addr, L, dhead, kv_heads)

    elif mode == "B":
        base_addr = mode_B(trace, base_addr, L, dhead, kv_heads, gqa_ratio)

    elif mode == "C":
        base_addr = mode_C(trace, base_addr, L, kv_heads, gqa_ratio)

    else:
        raise ValueError("Unknown mode")

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
                        choices=["A", "B", "C"], default="A")

    parser.add_argument("-o", "--output", type=str,
                        default="ddr5_gqa.trace")

    args = parser.parse_args()

    run(
        dhead=args.dhead,
        kv_heads=args.kv_heads,
        nq_heads=args.q_heads,
        L=args.seqlen,
        mode=args.mode,
        output=args.output
    )


if __name__ == "__main__":
    main()