import random, time
import numpy as np
from typing import Optional
from extra.lr_scheduler import OneCycleLR
from tinygrad import nn, dtypes, Tensor, Device, GlobalCounters, TinyJit
from tinygrad.nn.state import get_state_dict, get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import Context, BEAM, WINO, getenv, colored, prod
from extra.bench_log import BenchEvent, WallTimeEvent

# return a binary mask in the format of BS x C x H x W where H x W contains a random square mask
def make_square_mask(shape, mask_size) -> Tensor:
  BS, _, H, W = shape
  low_x = Tensor.randint(BS, low=0, high=W-mask_size).reshape(BS,1,1,1)
  low_y = Tensor.randint(BS, low=0, high=H-mask_size).reshape(BS,1,1,1)
  idx_x = Tensor.arange(W, dtype=dtypes.int32).reshape((1,1,1,W))
  idx_y = Tensor.arange(H, dtype=dtypes.int32).reshape((1,1,H,1))
  return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

def random_crop(X:Tensor, crop_size=32):
  mask = make_square_mask(X.shape, crop_size)
  mask = mask.expand((-1,3,-1,-1))

  start_time = time.monotonic()
  mask_np = mask.clone().numpy()
  X_np = X.clone().numpy()
  X_cropped_np = Tensor(X_np[mask_np]).reshape((-1, 3, crop_size, crop_size)).numpy()
  end_time = time.monotonic()
  elapsed_ms = (end_time - start_time) * 1000
  print(f"Numpy masking time: {elapsed_ms:.3f} ms")

  start_time = time.monotonic()
  X_cropped = X.masked_select(mask).reshape((-1, 3, crop_size, crop_size)).numpy()
  end_time = time.monotonic()
  elapsed_ms = (end_time - start_time) * 1000
  print(f"Tinygrad masking time: {elapsed_ms:.3f} ms")
  print(f"check operation result: {np.array_equal(X_cropped, X_cropped_np)}")

if __name__ == "__main__":
  batch_size = [10, 50, 100, 200, 500, 1000]

  for bs in batch_size:
    print(f"Tensor size: {bs * 3 * 36 * 36}")
    test_tensor = Tensor.randn(bs, 3, 36, 36)
    random_crop(test_tensor)
    print("\n")
