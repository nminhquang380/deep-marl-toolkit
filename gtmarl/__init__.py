
import warnings as _w
# CPU-only runs (CUDA_VISIBLE_DEVICES=""): PyTorch warns once at the first backward that CUDA is
# unavailable. It is harmless; silence it so that real errors stand out in the logs.
_w.filterwarnings("ignore", message="CUDA initialization")
