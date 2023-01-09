# SEN2DWATER: A Novel Multitemporal Dataset and Deep Learning Benchmark For Water Resources Analysis
#### Authors: Francesco Mauro, [Alessandro Sebastianelli](https://alessandrosebastianelli.github.io/), Silvia Liberata Ullo

![](imgs/dataset.png)

This repository contains the code related to our paper submitted to IGARSS2023

Please note that in this repository you can find a small subset of the dataset, made available here to explore the structure of the final dataset and to have a quick look to the data. 

[Click Here to Download the Final Dataset]()

## Notes

Please note that the results and the TensorBoard records are available in the [tmp](tmp) folder. Each model has a separated folder in which you can find:

- Visual and numerical results (res folder):
	- df: dataframes for each epoch of training containing numerical results
	- gt: ground truth images for each epoch of training
	- pr: predicted images for each epoch of training
- TensorBoard records for training and validation (train and validation folder)
- Model weights at the last epoch: model.h5

## Citations
Please cite our works when using our code and dataset.

The dataset has been built with a modified version of the scripts proposed in:

	@article{sebastianelli2021automatic,
 		 title={Automatic dataset builder for Machine Learning applications to satellite imagery},
  		 author={Sebastianelli, Alessandro and Del Rosso, Maria Pia and Ullo, Silvia Liberata},
 		 journal={SoftwareX},
 	 	 volume={15},
  		 pages={100739},
  		 year={2021},
  		 publisher={Elsevier}
	}

while the code relates to:

	@inproceedings{mauro2023water,
		title={SEN2DWATER: A Novel Multitemporal Dataset and Deep Learning Benchmark For Water Resources Analysis},
		author={Maruo, Francesco and Sebastianelli, Alessandro and Ullo, Silvia Liberata},
		booktitle={IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium},
		pages={},
		year={2023},
		organization={IEEE}
	}


