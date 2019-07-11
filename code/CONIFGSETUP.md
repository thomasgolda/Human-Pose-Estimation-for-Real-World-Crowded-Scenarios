## Experiment config setup

### Explaination of selected settings

|Setting                            |Use                                                                                            |
|-----------------------------------|-----------------------------------------------------------------------------------------------|
|EXPERIMENT_NAME                    | Name of the experiment. The output will be placed in a directory with the same name           |
|DATASET                            | Path to the dataset config the experiment should be run on                                    |
|MODEL.backbone                     | Choose backbone network either: `resnet50` `resnet101` `resnet152`                            |
|MODEL.occluded_cross_branch        | Enable occlussion detection with OccNetCB                                                     |
|MODEL.occluded_detection           | Enable occlussion detection with OccNet                                                       |
|MODEL.occluded_hard_loss           | Enable loss that explicitly punishes if a keypoint is detected in the wrong branch.           |
|MODEL.occluded_loss_weight         | Weight factor to increase the importance of the occluded loss                                 |
|MODEL.stop_crossbranch_grad        | Disable gradient calculation in OccNetCB between the occluded and visible branch              |
|MODEL.interference_joints          | Train network to detect every joint within the target persons bounding box. Similar to AlphaPose|
|TRAIN.coco_cutout                  | Enable cutout augmentation with objects from COCO Instance Segmenation.                       |
|TRAIN.coco_person_cutout           | Enable cutout augmenation with body parts from COCO Keypoint Dataset.                         |
|TRAIN.fullbodycut                  | Enable cutout augmenation with full persons from COCO Keypoint Dataset. Note that person will no be placed at the center of the image to avoid complete overlap|
|TRAIN.random_cutout                | Randomly switch between two enabled cutout methods.                                           |
|TRAIN.reset_v_flags                | Reset visibility flags to only 0 and 1. This option should only be used if OccNet/OccNetCB is trained on CrowdPose|
|TRAIN.structure_aware_loss         | Enable structure aware loss                                                                   |
|TRAIN.structure_aware_loss_weight  | Weighting favtor for the structure aware loss.                                                |
