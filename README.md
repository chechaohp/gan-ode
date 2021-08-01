# gan-ode
## preparation command
To run in Google Colab, run the following
```python
import time
from IPython.display import clear_output

# start = time.time()
# !cp drive/MyDrive/processed_UCF101.zip ./UCF101.zip
# print('Copy time:',round((time.time() - start)/60,2))
start = time.time()
!unzip -q drive/MyDrive/processed_UCF101.zip -d UCF101
print('Unzip time:',round((time.time() - start)/60,2))

!git clone https://chechaohp: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@github.com/chechaohp/gan-ode.git
%cd gan-ode
!pip3 install -r requirements.txt
!pip3 install --prefix=/opt/intel/ipp ipp-devel
!pip3 install git+https://github.com/pytorch/accimage
clear_output()
!python3 utils/ucf101_json.py ../UCF101/annotations
```

Current Colab training [Colab](https://colab.research.google.com/drive/1866LALVZWAE4PNTQdB3Vc643rA5HE5pO?authuser=2#scrollTo=rlhdraWwOsYV)

## Check list
- [x] **Stage 0:** **(Done)** refactor  run [DVD-GAN](https://github.com/Harrypotterrrr/DVD-GAN)
- [x] **Stage 1:** **(Done)** Implement DVD-GAN with Neural ODEs: replace ResNet block with ODE function and use [Augmented Neural ODE](https://arxiv.org/abs/1904.01681) (**_FAILED_** need to much RAM to run, even [Google Colab Pro 25GB RAM](https://colab.research.google.com/drive/1x_XYFomv3FWYj-LN7vO3uVaa8m6GOAIO?usp=sharing) cannot run it )
- [ ] **Stage 2:** **(On going)** Implement training with ODE by following [this paper](https://arxiv.org/abs/2010.15040)
    - [x] Euler's Method
    - [x] Heun's Method (RK2)
    - [x] Runge - Kutta 4 (RK4)
- [ ] **Stage 3:** Implement something novel, might need to switch from DVD-GAN to MoCoGan [[PDF](https://arxiv.org/abs/1707.04993) - [Code](https://github.com/sergeytulyakov/mocogan)]