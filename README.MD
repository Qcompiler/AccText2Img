An easy to use pluging for accelerate the SDXL.
Just add the two line code before inference:
```
from mylinear import MixLinear_GEMM
torch.nn.Linear = MixLinear_GEMM
```

For FP16
```
srun -N 1 --gres=gpu:4090:1 python test.py
```

For mixed-int8
```
srun -N 1 --gres=gpu:4090:1 python test.py --mix_linear
```
