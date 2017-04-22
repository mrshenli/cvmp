1. unzip the package
2. cd computer_vision_MP_Cifar_100
3. run setup.m everytime start Matlab
4. if Gpu is supported, then run setup('useGpu',true)
5. cd computer_vision_MP/code
6. if Gpu is supported, change the useGpu parameter to "true" on line 29 in cnn_cifar.m
7. run cnn_cifar('fine') and a 'cifar_prediction.csv' will be generated after training is done
8. upload 'cifar_prediction.csv' to Kaggle
9. run net_dissection_mnist.m and analyze your network