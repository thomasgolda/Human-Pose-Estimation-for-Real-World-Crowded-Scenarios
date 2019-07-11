## Simple Baseline & OccNet

### Requirements

- [pycocotools](https://github.com/cocodataset/cocoapi)
- [crowdposetools](https://github.com/Jeff-sjtu/CrowdPose)
- tensorflow-gpu == 1.11.0
- numpy
- path.py
- easydict
- PyYAML
- opencv-python
- setproctitle
- Cython>=0.19.2
- numpy>=1.7.1
- scipy>=0.13.2

### Setup
* Run `pip install -r requirement.txt` to install required modules.
* Run `cd ${Project_ROOT}/lib` and `make` to build NMS modules.
* Download imagenet pre-trained resnet models from [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim) and place them in the `${Project_ROOT}/imagenet_weights`.
* Before running any experiments the paths to the datasets needs to be adjusted to point to the location of the datasets. 
Therefore, simply edit the paths in the yaml-files in `datasets` to fit your needs. 
A valid yaml-file for CrowdPose could look the following way:

```
name: CrowdPose
num_kpts: 14
kpt_format: CrowdPose
root: ../../data/CrowdPose
train_set: annotations/json/crowdpose_train.json
val_set: annotations/json/crowdpose_val.json
test_set: annotations/json/crowdpose_test.json
human_dets: dets/human_detection_test.json
train_images: images
val_images: images
test_images: images
```
More examples can be found [here]().

### Usage

#### Training

In the `main` folder run 

`python train.py --gpu 0 --cfg ../experiments/crowdpose_baseline.yaml`

to train the network on gpu 0 with the given cfg.

Remaining arguments for `train.py` are listed below:

|Argument        | Example| Explaination  |
| ------------- |----------| -----|
| `--gpu`      | `--gpu 0` | Select one ore more gpus with their ids. When providing more than one gpu id name use `,` to seperate the ids.|
| `--cfg`      | `--cfg ../experiments/crowdpose_baseline.yaml`      |  Path to the config for the training experiment  |
| `--continue`      | `--continue`      |  Continue training with the last saved checkpoint  |

#### Validation

In the `main` folder run 

`python test.py --gpu 0 --cfg ../experiments/crowdpose_baseline.yaml --epoch 140`

to test the results of the experiment `crowdpose_baseline.yaml` on gpu 0 on epoch 140.

In order to validate the results on a different, dataset simply provide a path to the yaml-cfg of the dataset e.g. 
`python test.py --gpu 0 --cfg ../experiments/crowdpose_baseline.yaml --epoch 140 --dataset ../datasets/crowdPE.yaml`

An explaination of all the arguments for `test.py` can be seen below:

|Argument        | Example| Explaination  |
| ------------- |----------| -----|
| `--gpu`      | `--gpu 0` | Select one ore more datasets which should be visualized. When providing more than one dataset name use `,` to seperate the names. Use `all` to select every dataset in the datasets folder.  |
| `--cfg`      | `--cfg ../experiments/crowdpose_baseline.yaml`      |  Path to the config for the experiment which should be validated  |
| `--test_epoch`      | `--test_epoch 140`      |  Select which checkpoint should be used for testing.  |
| `--use_dets`      | `--use_dets`      |  Use human detections which are set in the dataset config.  |
| `--vis`      | `--vis`      |  Visualize detections during evaluation. Pose predictions are drawn onto the images.  |
| `--vis_occ`      | `--vis_occ`      |  Visualize occluded predictions during evaluation. Keypoints are drawn on person cutouts..  |
| `--dataset`      | `--dataset `      |  Evaluate on a different dataset than specified in the experiment config by providing a path to a different yaml-config. Note this paramater is not required if you want to evaluate on the same dataset you trained on. Furthermore it is required that the keypoint format is aligned with the training dataset because the same loader will be utilized.|

#### Adjusting Configs

In order to create your own config either use `experiments/create_cfg.py` or simply edit an existing config. The effects of each parameter are explained [here](./CONIFGSETUP.md).

### Acknowledgements

This project is a modified version of [mks0601's reimplemntation](https://github.com/mks0601/TF-SimpleHumanPose/)
  of the [Simple Baseline Model of [1]](https://github.com/Microsoft/human-pose-estimation.pytorch).
  
### References

[1] Xiao, Bin, Haiping Wu, and Yichen Wei. "Simple Baselines for Human Pose Estimation and Tracking". ECCV 2018.

[2] Fabbri, M., Lanzi, F., Calderara, S., Palazzi, A., Vezzani, R. and Cucchiara, R. "Learning to Detect and Track Visible and Occluded Body Joints in a Virtual World", ECCV 2018