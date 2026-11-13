import re, zlib, sys
data=open(sys.argv[1],'rb').read()
# map object number -> raw object bytes
objs={}
for m in re.finditer(rb'(\d+)\s+0\s+obj(.*?)endobj', data, re.S):
    objs[int(m.group(1))]=m.group(2)
def stream_of(b):
    m=re.search(rb'stream\r?\n(.*?)\r?\nendstream', b, re.S)
    if not m: return None
    try: return zlib.decompress(m.group(1))
    except Exception: return m.group(1)
def parse_cmap(t):
    mp={}
    for blk in re.findall(rb'beginbfchar(.*?)endbfchar', t, re.S):
        for a,b in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
            mp[int(a,16)]=bytes.fromhex(b.decode()).decode('utf-16-be','ignore')
    for blk in re.findall(rb'beginbfrange(.*?)endbfrange', t, re.S):
        for a,b,c in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
            s=int(c,16)
            for i,k in enumerate(range(int(a,16),int(b,16)+1)):
                mp[k]=chr(s+i)
    return mp
# font objects: Type0 with ToUnicode
fontmap={}
for n,b in objs.items():
    if b'/Type0' in b and b'/ToUnicode' in b:
        r=re.search(rb'/ToUnicode\s+(\d+)\s+0\s+R', b)
        t=stream_of(objs[int(r.group(1))])
        fontmap[n]=parse_cmap(t)
# resources: font name -> obj
name2obj={}
for n,b in objs.items():
    for fm in re.findall(rb'/Font\s*<<(.*?)>>', b, re.S):
        for nm,ref in re.findall(rb'/(\w+)\s+(\d+)\s+0\s+R', fm):
            name2obj[nm]=int(ref)
out=[]
for n,b in sorted(objs.items()):
    t=stream_of(b)
    if not t or b'BT' not in t: continue
    cur=None; line=[]
    for tok in re.finditer(rb'/(\w+)\s+[\d.]+\s+Tf|\[(.*?)\]\s*TJ|<([0-9A-Fa-f]*)>\s*Tj|(ET)', t, re.S):
        if tok.group(1): cur=fontmap.get(name2obj.get(tok.group(1)),{})
        elif tok.group(4): line.append('\n')
        else:
            hexes = re.findall(rb'<([0-9A-Fa-f]*)>', tok.group(2)) if tok.group(2) is not None else [tok.group(3)]
            for h in hexes:
                h=h.decode()
                for i in range(0,len(h),4):
                    line.append((cur or {}).get(int(h[i:i+4],16),'?'))
    out.append(''.join(line))
txt='\n'.join(out)
open(sys.argv[2],'w').write(txt)
print(len(txt))
