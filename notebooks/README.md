# KVASIR Dataset (Simple Neural Network)

Dataset Link: [KVASIR Dataset](https://datasets.simula.no/kvasir/data/kvasir-dataset-v2.zip)

## Simple Neural Network

![Base Model](https://www.pyimagesearch.com/wp-content/uploads/2018/09/keras_tutorial_simplenn_arch.png)

### Simple Neural Network [Normal]

![Simple Neural Network](simple-nn/chart.png)

| PARAMETER            | VALUE |
|----------------------|-------|
| Image Resolution     | 32x32 |
| Test Data Percentage | 20%   |
| Augmented            | No    |
| Learning Rate        | 0.01  |
| Batch Size           | 32    |
| Number of Epochs     | 50    |
| Average Precision    | 71%   |
| Average Recall       | 70%   |

[IPython Notebook](SIMPLE_NN.ipynb)

### Simple Neural Network [Greyscaled]

![Simple Neural Network](simple-nn-grey/chart.png)

| PARAMETER            | VALUE |
|----------------------|-------|
| Image Resolution     | 32x32 |
| Test Data Percentage | 20%   |
| Augmented            | No    |
| Learning Rate        | 0.01  |
| Batch Size           | 32    |
| Number of Epochs     | 50    |
| Average Precision    | 50%   |
| Average Recall       | 51%   |

[IPython Notebook](SIMPLE_NN_GREY.ipynb)

## Small VGGNet

### Small VGGNet [Normal]

![Simple Neural Network](small-vggnet/chart.png)

| PARAMETER            | VALUE |
|----------------------|-------|
| Image Resolution     | 32x32 |
| Test Data Percentage | 20%   |
| Augmented            | Yes   |
| Learning Rate        | 0.01  |
| Batch Size           | 32    |
| Number of Epochs     | 50    |
| Average Precision    | 74%   |
| Average Recall       | 71%   |

[IPython Notebook](SMALL_VGGNET.ipynb)

### Small VGGNet [64x64 Images]

![Simple Neural Network](small-vggnet-64/chart.png)

| PARAMETER            | VALUE |
|----------------------|-------|
| Image Resolution     | 64x64 |
| Test Data Percentage | 20%   |
| Augmented            | Yes   |
| Learning Rate        | 0.01  |
| Batch Size           | 32    |
| Number of Epochs     | 50    |
| Average Precision    | 73%   |
| Average Recall       | 60%   |

[IPython Notebook](SMALL_VGGNET_64.ipynb)

### Small VGGNet [128x128 Images]

![Simple Neural Network](small-vggnet-128/chart.png)

| PARAMETER            | VALUE   |
|----------------------|---------|
| Image Resolution     | 128x128 |
| Test Data Percentage | 15%     |
| Augmented            | Yes     |
| Learning Rate        | 0.01    |
| Batch Size           | 64      |
| Number of Epochs     | 100     |
| Average Precision    | 79%     |
| Average Recall       | 74%     |

Generated using python script.
