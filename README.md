# Classification of depression by EEG signals using neural networks
This code is part of the results of a scientific project: "Development of a neural network approach to recognize early predictors of depression with subsequent diagnosis." The project was supported by the "Umnik" program: * Grant No. 15548GU / 2020 *. Currently the project is terminated without completion at the initiative of the contractor.
The code is provided "as is" and will not be updated in the future. Some parts of the code were borrowed from other repositories and may have their own license.
# Results and comments
The dataset used to evaluate neural networks was downloaded from: https://figshare.com/articles/dataset/EEG_Data_New/4244171.
Seven architectures have been tested:
 1. **CnnZero** (cnn.py) is simplest convolutional network with two cores (one per class label). With a total of 500 parameters, it reaches 0.70 ± 0.18 F1-score. It worked very quickly and surprisingly well.
 2. **CnnFirst** (cnn.py) is a network with a large number of cores (32), but still single layer. This network gives much better results than previous variant: 0.81 ± 0.07 F1-score. The optimal choice of a fully convolutional network for a given dataset.
 3. **CnnSecond** (cnn.py) is network with 32 cores in the first layer and 64 in the second. This network showed worse results than the previous version, only 0.77 ± 0.08 F1-score. Probably the reason is a sub-optimal optimization method.
 4. **RNN** (rnn.py) is recurrent architecture with GRU based on CnnSecond. This improves the results to 0.83 ± 0.08 F1-score. The architecture has not been finetuned and this type of network can probably perform better.
 5. **RIM** (rim.py) is a recurrent independent mechanism network based on code from: https://github.com/dido1998/Recurrent-Independent-Mechanisms. This code is very different from the canonical implementation, but still works well: 0.85 ± 0.06 F1-score.
 6. **FNet** (fnet.py) is a pseudo-transformer architecture in which attention blocks are replaced by non-parametric Fourier transform blocks. The code was borrowed from: https://github.com/rishikksh20/FNet-pytorch. A very small number of parameters and very high requirements for computing resources. Works great: 0.86 ± 0.04 F1 score - the highest of all networks tested.
 7. **BENDR** (bendr.py) is transformer for processing EEG signals. The code and pretrained weights were borrowed from: https://github.com/SPOClab-ca/BENDR. Results: 0.86 ± 0.05 F1-score. The same as FNet, but this network has almost three orders of magnitude more parameters (more than 157 million versus 130 thousand).

The training code is in train.py and a Google Colaboratory Notebook will be added shortly with instructions on how to reproduce the results achieved.

