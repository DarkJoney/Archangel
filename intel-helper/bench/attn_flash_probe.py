import math, time, torch
import torch.nn.functional as F
dev=torch.device("xpu"); H,KV,D,R=24,4,256,2048
def sync(): torch.xpu.synchronize()
def timeit(f,n=3):
    f(); sync(); t=time.perf_counter()
    for _ in range(n): f()
    sync(); return (time.perf_counter()-t)/n
for total in (16384,32768):
    q=torch.randn(1,H,R,D,device=dev,dtype=torch.bfloat16)
    k=torch.randn(1,KV,total,D,device=dev,dtype=torch.bfloat16)
    v=torch.randn(1,KV,total,D,device=dev,dtype=torch.bfloat16)
    kr=k.repeat_interleave(6,1); vr=v.repeat_interleave(6,1)
    for name,f in (("nomask gqa",lambda:F.scaled_dot_product_attention(q,k,v,enable_gqa=True)),
                   ("nomask rep",lambda:F.scaled_dot_product_attention(q,kr,vr))):
        try: print(total,name,"%.3fs"%timeit(f))
        except Exception as e: print(total,name,"FAIL",str(e)[:100])
    try:
        r=torch.ops.aten._scaled_dot_product_fused_attention_overrideable(q,kr,vr,None,0.0,False,False)
        print(total,"overrideable returns",[None if x is None else tuple(x.shape) if hasattr(x,'shape') else x for x in r][:4])
    except Exception as e: print(total,"overrideable FAIL",str(e)[:150])
    del q,k,v,kr,vr; torch.xpu.empty_cache()
