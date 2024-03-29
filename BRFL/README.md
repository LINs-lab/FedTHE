# **BRFL**: a **B**enchmark for **R**obust **F**ederated **L**earning

This folder contains the datasets and data pipeline for the paper [Test-Time Robust Personalization for Federated Learning](https://arxiv.org/abs/2205.10920) (ICLR 2023) by Liangze Jiang and Tao Lin, separated from the main codebase for better reuse.

The aim of this benchmark is to properly evaluate the ID performance and OOD robustness of Federated Learning algorithms during test-time (deployment). 
To this end, diverse distribution shifts are taken into consideration, including [common corruptions](https://github.com/hendrycks/robustness), label distribution shift, [natural distribution shift](https://github.com/modestyachts/CIFAR-10.1), and a mixture of ID / OOD test.

## How it works
The benchmark currently contains CIFAR-10 and [ImageNet32](https://patrykchrabaszcz.github.io/Imagenet32/) (downsampled ImageNet). You can use `run.py` to obtain the `Dataset` and `Dataloader` of them.
### CIFAR10
<img align="center" src="assets/data_pipeline.png" width="750">

1. The CIFAR-10 Train & Test sets are merged and then split into K non-i.i.d pieces by Dirichlet distribution.
2. Each of the K pieces is a client's local data, and is uniformly and randomly partitioned into `local train, val, and test` sets.
3. `Corrupted test` set is obtained by randomly applying a corruption to each local test samples.
4. `Out-of-Client test` set is obtained by randomly sampling other clients' test set, mimicing the label distribution shift.
5. `Natural shift test` set is obtained by splitting CIFAR10.1 to each client according to their local label distributions.
6. Finally, `Mixture of tests` is obtained by randomly sampling the above ID/OOD test sets.

### ImageNet32

1. The data pipeline of ImageNet32 follows the same procedure as CIFAR-10, except that ImageNet[-A](https://github.com/hendrycks/natural-adv-examples) / [-R](https://github.com/hendrycks/imagenet-r) and [-V2](https://github.com/modestyachts/ImageNetV2) are considered as OOD test sets (check `run.py` for more details).

## Visualizations
We provide notebooks [`visualization_cifar10.ipynb`](visualization_cifar10.ipynb) and [`visualization_imagenet.ipynb`](visualization_imagenet.ipynb) to visualize and explore the local data of each client.
### CIFAR10 (non-i.i.d alpha 0.1)
<img align="center" src="assets/cifar.png" width="750">

### ImageNet32 (non-i.i.d alpha 0.01)
<img align="center" src="assets/imagenet.png" width="750">

## Requirements
* See `extra_requirements.sh`
* For ImageNet32, please first download (registration required) and extract the [train & val dataset](https://image-net.org/download-images) to `./imagenet32/imagenet32/`

## Citation

If you find this useful in your research, please consider citing:

```
@inproceedings{jiang2023test,
  title={Test-Time Robust Personalization for Federated Learning},
  author={Jiang, Liangze and Lin, Tao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2023}
}
```