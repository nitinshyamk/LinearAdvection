# LinearAdvection
GPU accelerated linear advection diffusion (1 dimensional) - 10x faster than tests in CPU. Currently 3x slower than highly optimized MATLAB code, but has the advantage of consuming minimal memory.

Future enhancements seeking to improve speed could batch together GPU invocations to avoid overhead of transferring memory back and forth from host to device.
