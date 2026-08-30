import struct, collections, glob, sys
def read_gguf(path):
    f=open(path,'rb'); f.read(4); struct.unpack('<I',f.read(4))
    n_tensors,=struct.unpack('<Q',f.read(8)); n_kv,=struct.unpack('<Q',f.read(8))
    def rd_str():
        n,=struct.unpack('<Q',f.read(8)); return f.read(n).decode('utf-8','replace')
    def rd_val(t):
        sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t==8: return rd_str()
        if t==9:
            et,=struct.unpack('<I',f.read(4)); n,=struct.unpack('<Q',f.read(8))
            return [rd_val(et) for _ in range(n)]
        f.read(sz[t]); return None
    kv={}
    for _ in range(n_kv):
        k=rd_str(); t,=struct.unpack('<I',f.read(4)); v=rd_val(t)
        if v is not None: kv[k]=v
    TS={0:('F32',4,1),1:('F16',2,1),8:('Q8_0',34,32),12:('Q4_K',144,256),14:('Q6_K',210,256),
        10:('Q2_K',84,256),11:('Q3_K',110,256),13:('Q5_K',176,256),20:('IQ4_NL',18,32),
        16:('IQ2_XS',74,256),23:('IQ3_XXS',98,256),30:('BF16',2,1),17:('IQ2_S',82,256),
        18:('IQ2_S',82,256),19:('IQ3_S',110,256),21:('IQ4_XS',136,256),24:('IQ1_S',50,256)}
    out=[]
    for _ in range(n_tensors):
        name=rd_str(); nd,=struct.unpack('<I',f.read(4))
        dims=[struct.unpack('<Q',f.read(8))[0] for _ in range(nd)]
        tt,=struct.unpack('<I',f.read(4)); struct.unpack('<Q',f.read(8))
        ne=1
        for d in dims: ne*=d
        tn,bs,bel=TS.get(tt,('?',2,1))
        out.append((name,dims,tn, ne//bel*bs if bel>1 else ne*bs))
    return out,kv

pat=sys.argv[1]; used=int(sys.argv[2]) if len(sys.argv)>2 else None
allt=[]; kv={}
for p in sorted(glob.glob(pat)):
    t,k=read_gguf(p); allt+=t; kv.update(k)
tot=sum(t[3] for t in allt)
n_exp=used
g=collections.defaultdict(float)
for n,d,t,b in allt:
    if 'exps' in n: g['routed_all']+=b
    elif 'shexp' in n: g['shared']+=b
    elif 'attn' in n: g['attn']+=b
    elif 'token_embd' in n or 'per_layer_token_embd' in n: g['embed_GATHERED']+=b
    elif n.startswith('output'): g['output']+=b
    elif n.startswith('blk.'): g['layer_other']+=b
    else: g['global']+=b
# find expert count from any exps tensor 3rd dim
nexp=None
for n,d,t,b in allt:
    if 'exps' in n and len(d)==3: nexp=d[2]; break
print(f"file total {tot/1e9:.1f} GB | experts {nexp} used {used}")
active = g['attn']+g['layer_other']+g['output']+g['shared']+g['global']
routed_active = g['routed_all']*used/nexp if (nexp and used) else 0
print(f"  routed experts ALL      {g['routed_all']/1e9:>8.2f} GB -> active {routed_active/1e9:.2f} GB ({used}/{nexp})")
print(f"  attention (dense)       {g['attn']/1e9:>8.2f} GB")
print(f"  layer other             {g['layer_other']/1e9:>8.2f} GB")
print(f"  shared expert           {g['shared']/1e9:>8.2f} GB")
print(f"  output/lm_head          {g['output']/1e9:>8.2f} GB")
print(f"  global                  {g['global']/1e9:>8.2f} GB")
print(f"  embeddings (GATHERED, ~0 active)  {g['embed_GATHERED']/1e9:>8.2f} GB")
print(f"  ===> ACTIVE BYTES/TOKEN {(active+routed_active)/1e9:>8.2f} GB")
ab=(active+routed_active)/1e9
for bw in (360,):
    print(f"  ceiling @ {bw} GB/s: {bw/ab:.1f} tok/s   | 13 tok/s needs {13*ab:.0f} GB/s ({13*ab/bw*100:.0f}% of ceiling)")
