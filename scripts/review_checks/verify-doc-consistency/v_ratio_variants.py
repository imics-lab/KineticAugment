import numpy as np
clips=["01986","01987","01988","01991","01992"]
def L(a,i,j,d): return np.linalg.norm(a[:,j,d]-a[:,i,d],axis=1)
for d,lab in [((0,1),"xy"),((0,1,2),"xyz")]:
    res={}
    U={'L':[], 'R':[]}; F={'L':[], 'R':[]}; cm=[]
    for c in clips:
        a=np.load(f"data/test_output/{c}_landmarks.npy")[:,:33].astype(float); a=a[np.any(a!=0,axis=(1,2))]
        U['L'].append(L(a,11,13,d)); F['L'].append(L(a,13,15,d)); U['R'].append(L(a,12,14,d)); F['R'].append(L(a,14,16,d))
        if c=="01986":
            print(lab,"01986 left: mean ratio %.3f median ratio %.3f med/med %.3f mean/mean %.3f"%(np.mean(F['L'][-1]/U['L'][-1]),np.median(F['L'][-1]/U['L'][-1]),np.median(F['L'][-1])/np.median(U['L'][-1]),F['L'][-1].mean()/U['L'][-1].mean()))
        cm.append((np.median(np.r_[F['L'][-1],F['R'][-1]])/np.median(np.r_[U['L'][-1],U['R'][-1]])))
    for s in 'LR':
        u=np.concatenate(U[s]); f=np.concatenate(F[s])
        print(lab,s,"pooled med/med %.3f mean/mean %.3f mean ratio %.3f median ratio %.3f"%(np.median(f)/np.median(u),f.mean()/u.mean(),np.mean(f/u),np.median(f/u)))
    print(lab,"mean of clip med/med %.3f median of clip %.3f"%(np.mean(cm),np.median(cm)))
    # sum of medians per bone pooled (doc table style)
    uL=np.median(np.concatenate(U['L']));uR=np.median(np.concatenate(U['R']));fL=np.median(np.concatenate(F['L']));fR=np.median(np.concatenate(F['R']))
    print(lab,"(fL+fR)/(uL+uR) of pooled medians %.3f"%((fL+fR)/(uL+uR)))
# image-bounds coverage (Finding 5)
tot=0;cnt={}
for c in clips:
    a=np.load(f"data/test_output/{c}_landmarks.npy")[:,:33].astype(float)
    det=np.any(a!=0,axis=(1,2)); tot+=det.sum()
    for i in (15,16,23,24):
        inb=(a[:,i,0]>=0)&(a[:,i,0]<=1)&(a[:,i,1]>=0)&(a[:,i,1]<=1)&det
        cnt[i]=cnt.get(i,0)+inb.sum()
print("image-bounds coverage:",{i:round(cnt[i]/tot*100,1) for i in cnt},"of",tot)
