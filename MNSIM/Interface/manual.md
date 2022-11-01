# How to embed MNSIM in the NAS process

## NAS framework

We recommend [aw_nas](https://github.com/walkerning/aw_nas) as the Neural Architecture Search (NAS) framework.
In aw\_nas, you can utilize WeightsManager as the super net in the NAS process.
A super net is a neural network that ensembles all neural network candidates and their shared weights.
And different neural network candidates can be generated from the super net by different network genotypes (called rollout).

## Embedding MNSIM in aw_nas

1. Sampling a candidate from the super net
To test the hardware performance of a neural network candidate based on MNSIM, you need to sample a design candidate from the super net.
Based on the sampled rollout, you can generate a candidate net (cand\_net) as follows:![sample_cand_net](https://raw.githubusercontent.com/ILTShade/blob_images/master/images20221101185615.png)

2. Generate new hardware config
The hardware config is also can be searched in the NAS process.
Therefore, you should modify the hardware config based on the genotypes of the neural network candidates.
We recommend to modify the base hardware config file to get the new hardware config file as follows:![modify_config](https://raw.githubusercontent.com/ILTShade/blob_images/master/images20221101191226.png)
And the function goes as:![modify_function](https://raw.githubusercontent.com/ILTShade/blob_images/master/images20221101191324.png)

3. Initialize MNSIM Interface
In this branch of the MNSIM, we add a new class called AWNASTrainTestInterface as the interface in the file ``MNSIM/Interface/awnas\_interface.py'' between MNSIM and aw\_nas.
Based on the neural network candidate and the new hardware config, you can initialize the interface as follows:![interface](https://raw.githubusercontent.com/ILTShade/blob_images/master/images20221101192015.png)
*self is one type of the objective function in aw_nas*

4. Get the hardware performance
Now you can get the hardware performance based on MNSIM.
![evaluate](https://raw.githubusercontent.com/ILTShade/blob_images/master/images20221101192338.png)

* *hardware_evaluate* for the accuracy of the neural network candidate on the hardware
* *energy_evaluate* for the energy consumption of the neural network candidate on the hardware / mJ
* *latency_evaluate* for the latency of the neural network candidate on the hardware / ms
* *area_evaluate* for the area of the neural network candidate on the hardware / mm^2
