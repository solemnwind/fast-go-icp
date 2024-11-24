# TODOs

## Functionalities

- [x] upper/lower bound computation
- [x] new nodes spawn
- [ ] compute point clouds' stats, normalize (For scalability)

## Optimizations

- [ ] `set_rotation` for flattened k-d tree, to avoid repeated rotation computation
- [ ] `branch`-`feature/goicp-core/texture`: move data into texture objects to improve performance
- [x] ~~use `cudaStreamSynchronize` instead of `cudaDeviceSynchronize`~~
- [ ] use `float4` for Points, store squared sum as well
