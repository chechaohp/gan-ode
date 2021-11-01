Repo for my project **GAN with Neural ODEs for Video Generation**

### Prepare the data and requirements
- To prepare the Rotated MNIST run file `utils/images.py`
- To prepare the UCF101 data, run `get_data_ucf101.sh`
- To install all requirement package, run `pip install -r requirements.txt`
### How to run the code
For each stage use the following files and folder
- **Stage 1**: use folder [stage1](./stage1)
- **Stage 2**: use folder [stage2](./stage2)
- **Stage 3**:
    - `mnist_moco_ode.py` : MoCoGAN + Neural ODEs
    - `mnist_moco_sde.py` : MoCoGAN + Neural SDEs
    - `mnist_moco_cde.py` : MoCoGAN + Neural CDEs
    - `mnist_moco_ode_rnn.py` : MoCoGAN + ODE-RNN
    - `mnist_moco_ode_wgan.py` : MoCoGAN + Neural ODEs + Wassersteins loss (model cannot learn to generate anything)
    - `mnist_moco_ode_noise.py`: MoCoGAN + Neural ODEs + add noise to model parameters in attempt to make it escape the local minima (cannot escape)
    - `ucf_moco_ode.py`: MoCoGAN + Neural ODEs for UCF101
