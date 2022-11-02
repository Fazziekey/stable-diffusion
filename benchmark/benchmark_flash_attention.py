from time import time
from ldm.modules.attention import CrossAttention, enable_flash_attention
import torch


def benchmark_flash_attention():
    x1_q = torch.randn((64, 1024, 320)).cuda().half()
    x1_kv = torch.randn((64, 1024, 320)).cuda().half()
    x1_grad = torch.randn((64, 1024, 320)).cuda().half()
    x2_q = torch.randn((64, 256, 640)).cuda().half()
    x2_kv = torch.randn((64, 256, 640)).cuda().half()
    x2_grad = torch.randn((64, 256, 640)).cuda().half()
    att1_q = CrossAttention(320, 320, 8, 40).cuda().half()
    att1_kv = CrossAttention(320, 320, 8, 40).cuda().half()
    att2_q = CrossAttention(640, 640, 8, 80).cuda().half()
    att2_kv = CrossAttention(640, 640, 8, 80).cuda().half()
    
    for _ in range(5):
        o1_q = att1_q(x1_q)
        o1_kv = att1_kv(o1_q, x1_kv)
        o1_kv.backward(x1_grad)
        o2_q = att2_q(x2_q)
        o2_kv = att2_kv(o2_q, x2_kv)
        o2_kv.backward(x2_grad)
    torch.cuda.synchronize()
    time1 = time()
    for _ in range(10):
        o1_q = att1_q(x1_q)
        o1_kv = att1_kv(o1_q, x1_kv)
        o1_kv.backward(x1_grad)
        o2_q = att2_q(x2_q)
        o2_kv = att2_kv(o2_q, x2_kv)
        o2_kv.backward(x2_grad)
    torch.cuda.synchronize()
    time2 = time()
    print("native attention time: %.4fs" % (time2 - time1))
    
    enable_flash_attention()
    for _ in range(5):
        o1_q = att1_q(x1_q)
        o1_kv = att1_kv(o1_q, x1_kv)
        o1_kv.backward(x1_grad)
        o2_q = att2_q(x2_q)
        o2_kv = att2_kv(o2_q, x2_kv)
        o2_kv.backward(x2_grad)
    torch.cuda.synchronize()
    time1 = time()
    for _ in range(10):
        o1_q = att1_q(x1_q)
        o1_kv = att1_kv(o1_q, x1_kv)
        o1_kv.backward(x1_grad)
        o2_q = att2_q(x2_q)
        o2_kv = att2_kv(o2_q, x2_kv)
        o2_kv.backward(x2_grad)
    torch.cuda.synchronize()
    time2 = time()
    print("flash attention time: %.4fs" % (time2 - time1))


if __name__ == '__main__':
    benchmark_flash_attention()
