# Super-Resolution Baselines

- All baseline source code can be found [here](./model/baselines/)


## Updates
### 02-12-2023
- Finish EDSR, RCAN, DGNet - based network.
- Train EDSR, RCAN succesfully, not test with DGNet yet.
- Not implement SMSR-like network yet
- All networks follow a similar archintecture: Hourglass architecture with diffent body blocks (Residual + Attention)
![Hourglas](./assets/hourglass.png) 
- Training results:

    | **Network** | **PSNR** | **FLOPs** |
    |-------------|----------|-----------|
    | EDSR        | 32.34    |           |
    | RCAN        | 33.34    |           |
    | DGNet       |          |           |
    | SMSR        |          |           |


- Comments: 
    - Significant gap between EDSR and RCAN might be the difference in number of parameters. The current baseline setup lets RCAN parameters higher than EDSR.
