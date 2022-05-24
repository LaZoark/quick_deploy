# Meta CVAE

## Envrionment
- Python version >= `3.7` 
- Packages
    ```bash
    pip install -r requirements.txt
    ```
    This will install `torch1.8.2-cu101` by default.
    
    *NOTE* If GPU
    resources are not available, it will fallback to CPU automatically.

## Dataset
The folder structure of dataset.
```
./data
├── 水泵聯軸器_20211228.csv
└── 水泵馬達_20211116.csv
```

## Train from scratch
- Train a Vanilla AutoEncoder from scratch.
    ```bash
    python train_vanilla_ae.py
    ```
    The default log directory is `./logs/VanillaAE_YYYYMMDDhhmmss`, where
    `YYYYMMDDhhmmss` is the start time of training.

- Train a Meta-Learned AutoEncoder from scratch.
    ```bash
    python train_meta_ae.py
    ```
    The default log directory is `./logs/MetaAE_YYYYMMDDhhmmss`, where
    `YYYYMMDDhhmmss` is the start time of training.

- Train a Meta-Learned CVAE from scratch.
    ```bash
    python train_meta_cvae.py
    ```
    The default log directory is `./logs/MetaCVAE_YYYYMMDDhhmmss`, where
    `YYYYMMDDhhmmss` is the start time of training.

- Tensorboard shows the training loss curve
    ```bash
    tensorboard --logdir ./logs/ --port 6006
    ```
    View the loss curves at [`localhost:6006`](localhost:6006).

## Inference Example

These inference example randomly tests the reconstruction MSE of a single row
in finetune set.

- Do the inference from pretrained Vanilla AE.
    ```bash
    python inference.py --model vanilla_ae --weights ./logs/VanillaAE_YYYYMMDDhhmmss/finetune/model.pt 
    ```

- Do the inference from pretrained Meta-learned AE.
    ```bash
    python inference.py --model meta_ae --weights ./logs/MetaAE_YYYYMMDDhhmmss/finetune/model.pt 
    ```

- Do the inference from pretrained Meta-learned CVAE.
    ```bash
    python inference.py --model meta_cvae --weights ./logs/MetaCVAE_YYYYMMDDhhmmss/finetune/model.pt 
    ```