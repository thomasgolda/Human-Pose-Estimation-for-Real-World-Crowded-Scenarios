# Human Pose Estimation for Real-World Crowded Scenarios
_Accepted for 2019 16th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)_

[![Crowded Pose Estimation header](https://github.com/thomasgolda/Human-Pose-Estimation-for-Real-World-Crowded-Scenarios/blob/master/header.png?raw=true)](https://www.iosb.fraunhofer.de/servlet/is/12481/)

## Abstract
Human pose estimation has recently made significant progress with the adoption of deep convolutional neural networks. Its many applications have attracted tremendous interest in recent years. However, many practical applications require pose estimation for human crowds, which still is a rarely addressed problem. In this work, we explore methods to optimize pose estimation for human crowds, focusing on challenges introduced with dense crowds, such as occlusions, people in close proximity to each other, and partial visibility of people. In order to address these challenges, we evaluate three aspects of a pose detection approach: i) a data augmentation method to introduce robustness to occlusions, ii) the explicit detection of occluded body parts,  and iii) the use of the synthetic generated datasets. The first approach to improve the accuracy in crowded scenarios is to generate occlusions at training time using person and object cutouts from the object recognition dataset COCO (Common Objects in Context). Furthermore, the synthetically generated dataset JTA (Joint Track Auto) is evaluated for the use in real-world crowd applications. In order to overcome the transfer gap of JTA originating from a low pose variety and less dense crowds, an extension dataset is created to ease the use for real-world applications. Additionally, the occlusion flags provided with JTA are utilized to train a model, which explicitly distinguishes between occluded and visible body parts in two distinct branches. The combination of the proposed additions to the baseline method help to improve the overall accuracy by 4.7\% AP and thereby provide comparable results to current state-of-the-art approaches on the respective dataset.

## Citation
We believe in open research and we are happy if you find our work inspiring. If you use our code and results, please cite our [work](link zu arxiv oder avss paper nach submission).

```latex
@inproceedings{golda2019crowdposeestimation,
   title     = {{H}uman {P}ose {E}stimation for {R}eal-{W}orld {C}rowded {S}cenarios},
   author    = {Golda, Thomas and Kalb, Tobias and Schumann, Arne and Beyerer, J\"uergen},
   booktitle = {2019 16th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},
   year      = {2019}
 }
```

## Results
| Method          | AP         | AP_easy    | AP_medium   | AP_hard    |
|-----------------|:----------:|:----------:|:-----------:|:----------:|
| Xiao et al. [1] | 60.8       | 71.4       | 61.2        | 51.2       |
| Li et al. [2]   | **66.6**   | **75.7**   | 66.3        | **57.4**   |
| Ours            | 65.5       | 75.2       | **66.6**    | 53.1       |

<div style="text-align: center;"><img src="https://github.com/thomasgolda/Human-Pose-Estimation-for-Real-World-Crowded-Scenarios/blob/master/avss2019_crowd-paper-qualitative-results.png?raw=true" alt="Qualitative results" /></div>

## JTA Extension
For our experiments we created an extension to the dataset provided by Fabbri et al. which can be requested [here](https://github.com/fabbrimatteo/JTA-Dataset). The extension dataset for JTA can be downloaded from [here](annotationen für JTA). We think that Fabbri et al. did great work, so please [cite them](https://github.com/fabbrimatteo/JTA-Mods) as well when relating to our results.

The annotation files for each sequence can be found in the respective directories as `coords.csv`. The annotation format 
is aligned to the base JTA format, with the exception that the entries `row[10] - row[16]` are not part of the original JTA annotations.
In order to fully align the format, `row[10] - row[16]` can simply be dropped.


| Element   | Name          | Description                                                  |
| --------  | ------------- | ------------------------------------------------------------ |
| `col[0]`  | frame number  | number of the frame to which the joint belongs               |
| `col[1]`  | person ID     | unique identifier of the person to which the joint belongs   |
| `col[2]`  | joint type    | identifier of the type of joint; see 'Joint Types' subsection |
| `col[3]`  | x2D           | 2D _x_ coordinate of the joint in pixel                      |
| `col[4]`  | y2D           | 2D _y_ coordinate of the joint in pixel                      |
| `col[5]`  | x3D           | 3D _x_ coordinate of the joint in meters                     |
| `col[6]`  | y3D           | 3D _y_ coordinate of the joint in meters                     |
| `col[7]`  | z3D           | 3D _z_ coordinate of the joint in meters                     |
| `col[8]`  | occluded      | `1` if the joint is occluded; `0` otherwise                  |
| `col[9]`  | self-occluded | `1` if the joint is occluded by its owner; `0` otherwise     |
| `col[10]` | cam_3D_x      | 3D _x_ coordinate of the camera (not included in JTA base)   |
| `col[11]` | cam_3D_y      | 3D _y_ coordinate of the camera (not included in JTA base)   |
| `col[12]` | cam_3D_z      | 3D _z_ coordinate of the camera (not included in JTA base)   |
| `col[13]` | cam_rot_x     | _x_ rotation of the camera (not included in JTA base)        |
| `col[14]` | cam_rot_y     | _y_ rotation of the camera (not included in JTA base)        |
| `col[15]` | cam_rot_z     | _z_ rotation of the camera (not included in JTA base)        |
| `col[16]` | fov           | fov of the camera (not included in JTA base)                 |


## Code
Our code for training will be available here soon.

## References
[1] Xiao, B., Wu, H., & Wei, Y. (2018). _Simple baselines for human pose estimation and tracking. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 466-481)_.

[2] Li, J., Wang, C., Zhu, H., Mao, Y., Fang, H. S., & Lu, C. (2018). _CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark. arXiv preprint arXiv:1812.00324_.
