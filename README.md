# DMC-GAwAR


Distance-based Mutual Congestion (MPR) is a frequency-based filter ranking method (applicable to binary datasets), which belongs to a series of frequency-based rankers:

1- [Mutual Congestion (MC)](https://www.sciencedirect.com/science/article/pii/S0888754318304245)   Publication year: **2019**

2- [Sorted Label Interference (SLI)](https://www.sciencedirect.com/science/article/pii/S0306437921000259#!)   Publication year: **2021**

3- [Sorted Label Interference-gamma (SLI-gamma)](https://link.springer.com/article/10.1007/s11227-022-04650-w)   Publication year: **2022**

4- [Extended Mutual Congestion (EMC)](https://https://www.sciencedirect.com/science/article/pii/S1568494622007487#!).  Publication year: **2022**

5- [Maximum Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0957417424003865)  Publication year:  **2024**

6- Distance-based Mutual Congestion


Instruction:

After loading the corresponding dataset from your local drive:

1- Run lines 62-84 to calculate the summation of samples per label

2- Run lines 125-213 to calculate DMC measure for features of the dataset

3- Run lines 254-331 for corresponding functions of GAwAR

4- Run lines 336-374 for clustering the top 5% features recognized by DMC

5- Run lines 382-611 for GAwAR

For your convenience, we have uploaded the corresponding alphas (ranking of DMC for each dataset). You have the option to either load the corresponding alpha directly or calculate the respective alpha by implementing step 2.

