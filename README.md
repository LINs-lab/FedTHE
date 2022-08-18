# Test-time Robust Personalization for Federated Learning

This is an official implementation for the paper
[Test-Time Robust Personalization for Federated Learning](https://arxiv.org/abs/2205.10920)

## Abstract
Federated Learning (FL) is a machine learning paradigm where many clients collaboratively learn a shared global model with decentralized training data. 
Personalization on FL model additionally adapts the global model to different clients, achieving promising results on consistent local training & test distributions. 
However, for real-world personalized FL applications, it is crucial to go one step further: robustifying FL models under evolving local test set during deployment, where various types of distribution shifts can arise. 

In this work, we identify the pitfalls of existing works under test-time distribution shifts and propose a novel test-time robust personalization method, namely <ins>F</ins>ederated <ins>T</ins>est-time <ins>H</ins>ead <ins>E</ins>nsemble <ins>plus</ins> tuning (FedTHE+). 
We illustrate the advancement of FedTHE+ (and its degraded computationally efficient variant FedTHE) over strong competitors, for training various neural architectures (CNN, ResNet, and Transformer) on CIFAR10 and ImageNet and evaluating on diverse test distributions. 
Along with this, we build a benchmark for assessing performance and robustness of personalized FL methods during deployment.

## Code
Coming soon (please also check out our uncleaned code [here](https://github.com/lins-lab/fedthe/tree/draft_code))!

## References
If you use the code, please cite the following paper:

```
@article{jiang2022test,
  title={Test-Time Robust Personalization for Federated Learning},
  author={Jiang, Liangze and Lin, Tao},
  journal={arXiv preprint arXiv:2205.10920},
  year={2022}
}
```