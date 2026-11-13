import re, zlib, sys
data = open(sys.argv[1],'rb').read()
out=[]
for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S):
    s=m.group(1)
    try: t=zlib.decompress(s)
    except Exception: continue
    # collect text in TJ/Tj
    parts=re.findall(rb'\[(.*?)\]\s*TJ|\((.*?)\)\s*Tj', t, re.S)
    if not parts: continue
    line=[]
    for a,b in parts:
        seg = a if a else b
        strs = re.findall(rb'\(((?:\\.|[^\\)])*)\)', seg) if a else [seg]
        line.append(b''.join(strs))
    out.append(b' '.join(line))
txt=b'\n'.join(out).decode('latin-1')
print(len(txt)); print(txt[:6000])
