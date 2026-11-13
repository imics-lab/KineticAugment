import glob, numpy as np
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity
sk=SMPLXSkeleton(model_path="models")
M=[11,12,13,14,15,16]; B=[(11,13),(13,15),(12,14),(14,16)]
d_all=[];a_all=[];ld=[];la=[];perb={b:[] for b in B}
for p in sorted(glob.glob("data/test_output/*_world.npz")):
    z=np.load(p);P=z["landmarks"].astype(float);ok=np.isfinite(P).all(-1)&(z["visibility"]>=0.5)
    both=ok[:,11]&ok[:,12]; w=np.median(np.linalg.norm(P[both,11]-P[both,12],axis=1))
    for f in range(len(P)):
        try: out=reconstruct_body(fit_body(P[f],ok[f],sk),sk)
        except ValueError: continue
        pr=[m for m in M if ok[f,m]]
        src=out[pr];tgt=P[f,pr]
        al=apply_similarity(src,*similarity_transform(src,tgt))
        d_all+=list(np.linalg.norm(src-tgt,axis=1)/w); a_all+=list(np.linalg.norm(al-tgt,axis=1)/w)
        A=dict(zip(pr,al))
        for a,b in B:
            if ok[f,a] and ok[f,b]:
                obs=np.linalg.norm(P[f,b]-P[f,a])
                e=abs(np.linalg.norm(out[b]-out[a])-obs)/w; ld.append(e); perb[(a,b)].append(e)
                la.append(abs(np.linalg.norm(A[b]-A[a])-obs)/w)
print(f"direct pooled median {np.median(d_all)*100:.2f}%  aligned {np.median(a_all)*100:.2f}%  ratio-to-0.18: direct {np.median(d_all)*100/0.18:.1f}x aligned {np.median(a_all)*100/0.18:.1f}x")
print(f"length err pooled median: direct {np.median(ld)*100:.2f}%  aligned {np.median(la)*100:.2f}%; per bone direct", [round(np.median(v)*100,2) for v in perb.values()])
