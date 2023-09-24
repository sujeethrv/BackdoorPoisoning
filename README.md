# Backdoor Poisoning - Patch Attacks on MNIST Dataset

## Introduction
This project explores the concept of backdoor poisoning attacks using a ResNet-18 model trained on the MNIST dataset. Backdoor attacks aim to compromise the robustness of neural networks by introducing subtle modifications to the training data. This README provides an overview of the project, its methodology, results, and usage instructions for the code.

## Project Overview
Neural networks have become a crucial tool in the fields of machine learning and artificial intelligence, finding applications in various domains. Ensuring the robustness and security of these networks is of paramount importance, especially in the face of adversarial attacks. In this project, we investigate the effectiveness of backdoor poisoning attacks using patch-based techniques on a ResNet model trained on the MNIST dataset.

## Methodology
### Overview
The methodology of this project involves selecting a source class and a target class and then poisoning a percentage of the training images from the source class. The ResNet model is trained on this modified MNIST dataset, and the attack's success rate is measured by applying the same patch used during training to the test set images from the source class.

### Key Parameters
- Source Class: 0
- Target Class: 1
- Patch Size: 3x3
- Patch Pixel Values: 0.2
- Training Set Size: 60,000 images
- Training Epochs: 5 (15 for spatially invariant attacks)
- Learning Rate: 0.001 (ADAM optimizer)

## Results
### Poisoning Rate vs. Attack Success Rate
- Poisoning Rate of 1%: ASR - 97.14%
- Poisoning Rate of 5%: ASR - 100%
- Poisoning Rate of 7%: ASR - 100%
- Poisoning Rate of 10%: ASR - 100%

### Patch Intensity (L2 Norm) vs. Attack Success Rate
- Patch Intensity of 0.1: ASR - 99.69%
- Patch Intensity of 0.2: ASR - 99.77%
- Patch Intensity of 0.3: ASR - 99.89%

### Patch Size vs. Attack Success Rate
- Patch Size 1x1: ASR - 64.80%
- Patch Size 3x3: ASR - 99.39%
- Patch Size 5x5: ASR - 100%

### Patch Location Variations (Radius) vs. Attack Success Rate
- Radius 1: ASR - 99.90%
- Radius 3: ASR - 19.39%
- Radius 5: ASR - 0.10%

### Random Patch Locations vs. Attack Success Rate
- Random Patch Locations (5 epochs, 3x3 patch): ASR - 15.92%
- Random Patch Locations (15 epochs, 3x3 patch): ASR - 98.57%

### Combining Patch Size and Random Locations (15 epochs)
- Patch Size 1x1: ASR - 0%
- Patch Size 3x3: ASR - 98.57%
- Patch Size 5x5: ASR - 100%

## Conclusion
This project demonstrates the vulnerability of neural networks to backdoor poisoning attacks, even with minimal poisoning rates. The results show that the attack's success rate can be high, posing a serious security threat. Further evaluation on larger datasets is recommended to assess the generalizability of these findings.

## Usage Instructions
1. Run `backDoorPoisoningMain.py` to conduct experiments with different parameters.
2. Modify the parameters in the script to explore various configurations and attack scenarios.
