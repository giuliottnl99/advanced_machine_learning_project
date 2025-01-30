# Distributed Learning with LocalSGD

This repository contains an implementation of distributed learning using LocalSGD with multiple optimizers, including SGD, AdamW, LARS, and LAMB.

## Installation

### Install Dependencies

You can install the required dependencies using the following command:

```bash
pip install torch torchvision numpy matplotlib torch-optimizer tensorflow-addons
```

**Note:** If you need to use LAMB optimizer, you must also install TensorFlow:

```bash
!pip install torch-optimizer
# !pip install tensorflow==2.12 #this is needed only for LAMB. It is expensive to install
!pip install tensorflow-addons==0.20
```

### Running in Google Colab

If running in Colab, some dependencies are already installed. However, you need to install `torch-optimizer` manually:

```bash
!pip install torch-optimizer
```

## Usage

### Running the Training Script

To train the model, use the following command:

```bash
python train.py --K 1 --J 1 --epochs 150 --batch-size 64 --momentum 0.9 --lr-max 0.01 --lr-min 0.001 --weight-decay 0.0001 --optimizer sgd --save-path ./models/
```
Alternatively, in a Google Colab cell:
```python
import sys
sys.argv = [
    'main.py', '--optimizer', 'sgd', '--J', '16', '--K', '4',  '--lr-max', '0.004', '--lr-min', '0.0', '--momentum', '0.9', '--weight-decay', '0.005', '--transform', 'True',
    '--batch-size', '64', '--epochs', '200', '--start-epoch', '0',
    '--computational-power-unequal', 'True',  '--slowMMO', '0.9'
]
main()
```
### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--K` | Number of splits for training dataset | `1` |
| `--J` | Interval for weight averaging | `1` |
| `--epochs` | Number of training epochs | `150` |
| `--batch-size` | Batch size for training | `64` |
| `--momentum` | Momentum for training. In LAMB, this is Beta-1 value | `0.9` |
| `--lr-max` | Maximum learning rate for cosine annealing | `0.01` |
| `--lr-min` | Minimum learning rate for cosine annealing | `0.001` |
| `--weight-decay` | Weight decay | `0.0001` |
| `--optimizer` | Optimizer to use (`sgd`, `adam`, `lars`, `lamb`) | `sgd` |
| `--save-path` | Path to save CSV and last model | `''` |
| `--starting-model` | Path to starting model (optional) | `None` |
| `--start-epoch` | Set the epoch if resuming training from a checkpoint | `0` |
| `--eval-interval` | Interval for evaluation during training | `5` |
| `--transform` | Apply harder transformations | `True` |
| `--warmup` | Warmup epochs (0 means no warmup) | `0` |
| `--slowMMO` | Applies slowMMO; if 0, applies simple FedAVG | `0.0` |
| `--from-zero-warmup` | Enables special warmup if set to True | `False` |
| `--trust-coefficient` | Trust coefficient in LARS | `0.001` |
| `--beta-2` | Beta-2 value for LAMB optimizer | `0.999` |
| `--dont-use-fragmented-batches` | If set to True, skips the last batch if `size < J` | `False` |
| `--computational-power-unequal` | If True, splits `J` into fractions (1/16, 3/16, 4/16, 8/16) for `K=4` | `False` |
| `--use-weighted-avg-for-J-unequal` | If True, weights averaging by local worker steps when `computational-power-unequal` is enabled | `False` |

### Running with Different Optimizers

#### Using AdamW
```bash
python train.py --optimizer adam --lr-max 0.001 --weight-decay 0.01
```

#### Using LARS
```bash
python train.py --optimizer lars --lr-max 0.1 --momentum 0.9 --weight-decay 0.0005
```

#### Using LAMB
```bash
python train.py --optimizer lamb --lr-max 0.01 --momentum 0.9 --beta-2 0.999 --weight-decay 0.01
```

**Note:** Running with LAMB requires TensorFlow.

## Model Architecture

The implemented model is a variant of **LeNet-5** with modifications for CIFAR-100.

- **Conv1:** 3 × 64 (5 × 5, stride=1, padding=2) → ReLU → MaxPool(2 × 2)
- **Conv2:** 64 × 64 (5 × 5, stride=1) → ReLU → MaxPool(2 × 2)
- **FC1:** 64 × 6 × 6 → 384 → ReLU
- **FC2:** 384 → 192 → ReLU
- **FC3:** 192 → 100 (output classes)

## Evaluation

The script automatically evaluates the model on the validation set and saves accuracy/loss metrics in a CSV file.

To manually evaluate a trained model:

```bash
python evaluate.py --model-path ./models/last_model.pth
```

## Results

After training, results such as training loss, validation accuracy, and test accuracy are displayed. A csv file is generated

## Use of LAMB
If you are using the LAMB optimizer, you should import the correct version of tensorflow and run the LAMB class first. Then you run command_Collab.py normally.

## Use of graphGenerator.py
Run the script changing, in the final line, 'csv_to_render.csv' with the name of the actual csv file you want to render. It will generates the graphs of the losses and the graphs of the accuracies.