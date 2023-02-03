# SEN2DWATER: A Novel Multitemporal Dataset and Deep Learning Benchmark For Water Resources Analysis
#### Authors: Francesco Mauro, [Alessandro Sebastianelli](https://alessandrosebastianelli.github.io/), Silvia Liberata Ullo

![](imgs/dataset.png)

The final dataset contains the data used in the IGARSS2023 paper, but this is an evolving dataset growing time by time, so its size may change. We are currently working on its extensions and on its versioning. 

[Click Here to Download the Final Dataset](https://drive.google.com/drive/folders/1AYt8UYmgnTpeIDLY_4IFA_5LEBwYwaY8?usp=share_link)

In this repository you can find a small subset of the dataset, made available here to quickly explore the structure dataset and to have a quick look to the data.

## Usage

### Models training

To train the models run the following command:

`
	python train.py
`

If you want to change some settings, you can edit the [config file](config.py). Here you can define:
- the dataset you want to use and its train-validation split 
- image settings
- models settings (there is only one entry that it is used for all the models)

### TensorBoard

You can visualize the training process (or the results provided by us) using the following command:

`
tensorboard --logdir tmp
`
 
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


