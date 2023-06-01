# Test-time Robust Personalization for Federated Learning

This is an official implementation of the paper [Test-Time Robust Personalization for Federated Learning](https://arxiv.org/abs/2205.10920).

## Abstract
Federated Learning (FL) is a machine learning paradigm where many clients collaboratively learn a shared global model with decentralized training data. 
Personalization on FL model additionally adapts the global model to different clients, achieving promising results on consistent local training & test distributions. 
However, for real-world personalized FL applications, it is crucial to go one step further: robustifying FL models under the evolving local test set during deployment, where various types of distribution shifts can arise. 

In this work, we identify the pitfalls of existing works under test-time distribution shifts and propose <ins>F</ins>ederated <ins>T</ins>est-time <ins>H</ins>ead <ins>E</ins>nsemble <ins>plus</ins> tuning (FedTHE+), which personalizes FL models with robustness to various test-time distribution shifts. 
We illustrate the advancement of FedTHE+ (and its degraded computationally efficient variant FedTHE) over strong competitors, for training various neural architectures (CNN, ResNet, and Transformer) on CIFAR10 and ImageNet and evaluating on diverse test distributions. 
Along with this, we build a benchmark for assessing the performance and robustness of personalized FL methods during deployment. 


## Code
A separate code for [BRFL: A Benchmark For Robust FL](https://github.com/LINs-lab/FedTHE/tree/master/BRFL) can be found in this repository.

The complete code for FedTHE variants will be ready in 1-2 weeks (Please also check out our uncleaned code [here](https://github.com/lins-lab/fedthe/tree/draft_code))!

## Bibliography
If you find this repository helpful for your project, please consider citing:

```
@inproceedings{jiang2023test,
  title={Test-Time Robust Personalization for Federated Learning},
  author={Jiang, Liangze and Lin, Tao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2023}
}
```