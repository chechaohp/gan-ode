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

## Check list
[x] **Stage 0:** run DVD-GAN
[x] **Stage 1:** Implement DVD-GAN with Neural ODEs (**_FAILED_** need to much RAM to run, even [https://colab.research.google.com/drive/1x_XYFomv3FWYj-LN7vO3uVaa8m6GOAIO?usp=sharing](Google Colab Pro25GB RAM) cannot run it )
[] Stage 2: Implement training with ODE
[] Stage 3: Implement something novel