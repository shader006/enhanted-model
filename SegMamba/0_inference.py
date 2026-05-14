

import torch 
from model_segmamba.segmamba import SegMamba

t1 = torch.rand(1, 4, 128, 128, 128).cuda()


model = SegMamba().cuda()

out = model(t1)

print(out.shape)



