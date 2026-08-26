#!/usr/bin/env python3
"""Parse GGUF tensor types and report repack-eligibility.

Quant names lie. One model labelled "Q2_K_XL" was 94.4% IQ2_XS/IQ3_XXS by
parameter count -- neither of which can use llama.cpp's CPU_REPACK path. And
IQ4_XS is NOT repack-eligible despite the Q4-ish name. Run this before
committing to a multi-hundred-GB download.

usage: gguf_types.py <path-to-shard-00001-of-000NN.gguf>
       (globs the whole shard set automatically)
"""
import struct, collections, glob, os, sys, re

TYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
    10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',
    17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',
    23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16',
    34:'TQ1_0',35:'TQ2_0',39:'MXFP4',40:'NVFP4',41:'Q1_0',42:'Q2_0'}

# from ggml_repack_get_optimal_repack_type()
REPACK = {'Q4_0','Q4_K','Q5_K','Q6_K','Q2_K','IQ4_NL','MXFP4','Q8_0'}

BPW = {'IQ1_S':1.56,'IQ1_M':1.75,'IQ2_XXS':2.06,'IQ2_XS':2.31,'IQ2_S':2.5,
    'Q2_K':2.63,'IQ3_XXS':3.06,'Q3_K':3.44,'IQ3_S':3.44,'MXFP4':4.25,
    'IQ4_XS':4.25,'Q4_0':4.5,'Q4_K':4.5,'IQ4_NL':4.5,'Q5_K':5.5,'Q6_K':6.56,
    'Q8_0':8.5,'F16':16,'BF16':16,'F32':32,'I32':32}

def shards(first):
    m = re.sub(r'-\d{5}-of-\d{5}\.gguf$', '', first)
    got = sorted(glob.glob(m + '-*.gguf'))
    return got or [first]

def scan(path, counts, bytes_):
    f = open(path,'rb')
    if f.read(4) != b'GGUF': return
    struct.unpack('<I', f.read(4))
    nt = struct.unpack('<Q', f.read(8))[0]
    nkv = struct.unpack('<Q', f.read(8))[0]
    def rs():
        n = struct.unpack('<Q', f.read(8))[0]; return f.read(n)
    def skip(t):
        if t in (0,1,7): f.read(1)
        elif t in (2,3): f.read(2)
        elif t in (4,5,6): f.read(4)
        elif t == 8: rs()
        elif t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            for _ in range(n): skip(et)
        elif t in (10,11,12): f.read(8)
    for _ in range(nkv):
        rs(); skip(struct.unpack('<I', f.read(4))[0])
    for _ in range(nt):
        rs()
        nd = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack('<I', f.read(4))[0]; struct.unpack('<Q', f.read(8))
        n = 1
        for d in dims: n *= d
        name = TYPES.get(tt, str(tt))
        counts[name] += n
        bytes_[name] += n * BPW.get(name, 4) / 8

def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    counts, bytes_ = collections.Counter(), collections.Counter()
    files = shards(sys.argv[1])
    for p in files: scan(p, counts, bytes_)
    tot, totb = sum(counts.values()), sum(bytes_.values())
    if not tot: sys.exit("no tensors found (shard 1 is often metadata-only)")
    print(f"{len(files)} shard(s), {tot/1e9:.1f}B params, ~{totb/1e9:.1f} GB\n")
    print(f"{'type':<9} {'params':>16} {'share':>7} {'GB':>7}  repack?")
    elig = 0
    for k, v in counts.most_common():
        ok = k in REPACK
        if ok: elig += bytes_[k]
        print(f"{k:<9} {v:>16,} {100*v/tot:>6.2f}% {bytes_[k]/1e9:>6.1f}  {'YES' if ok else 'no'}")
    print(f"\nREPACK-ELIGIBLE BY BYTES: {100*elig/totb:.1f}%  (UPPER BOUND)")
    print()
    print("CAVEAT 1: this is necessary but NOT sufficient. The _8x8 repack traits")
    print("  also require shape constraints (ne[1] %% 8 == 0), so actual repack is")
    print("  often far lower. One model that scored 68%% here achieved only ~25%%.")
    print("  Ground truth is RssAnon vs RssFile in /proc/<pid>/status during load.")
    print("CAVEAT 2: for MoE models this barely matters at decode time -- routed")
    print("  experts run MUL_MAT_ID, which takes the gemv path at batch 1 and")
    print("  gains at most ~1.65x from repack even at 12 rows/expert. See README.")

main()
