# Human Pose Estimation for Real-World Crowded Scenarios
_Submitted for 2019 IEEE International Conference on Advanced Video and Signal-based Surveillance_

![Crowded Pose Estimation header](https://github.com/thomasgolda/Human-Pose-Estimation-for-Real-World-Crowded-Scenarios/blob/master/header.png?raw=true)

## Abstract
Human pose estimation has recently made significant progress with the adoption of deep convolutional neural networks. Its many applications have attracted tremendous interest in recent years. However, many practical applications require pose estimation for human crowds, which still is a rarely addressed problem. In this work, we explore methods to optimize pose estimation for human crowds, focusing on challenges introduced with dense crowds, such as occlusions, people in close proximity to each other, and partial visibility of people. In order to address these challenges, we evaluate three aspects of a pose detection approach: i) a data augmentation method to introduce robustness to occlusions, ii) the explicit detection of occluded body parts,  and iii) the use of the synthetic generated datasets. The first approach to improve the accuracy in crowded scenarios is to generate occlusions at training time using person and object cutouts from the object recognition dataset COCO (Common Objects in Context). Furthermore, the synthetically generated dataset JTA (Joint Track Auto) is evaluated for the use in real-world crowd applications. In order to overcome the transfer gap of JTA originating from a low pose variety and less dense crowds, an extension dataset is created to ease the use for real-world applications. Additionally, the occlusion flags provided with JTA are utilized to train a model, which explicitly distinguishes between occluded and visible body parts in two distinct branches. The combination of the proposed additions to the baseline method help to improve the overall accuracy by 4.7\% AP and thereby provide comparable results to current state-of-the-art approaches on the respective dataset.

## Citation
We believe in open research and we are happy if you find our work inspiring. If you use our code and results, please cite our [work](link zu arxiv oder avss paper nach submission).

```latex
@inproceedings{golda2019crowdposeestimation,
   title     = {{H}uman {P}ose {E}stimation for {R}eal-{W}orld {C}rowded {S}cenarios},
   author    = {Golda, Thomas and Kalb, Tobias and Schumann, Arne and Beyerer, J\"uergen},
   booktitle = {2019 IEEE International Conference on Advanced Video and Signal-based Surveillance (AVSS)},
   year      = {2019}
 }
```

## Results
| Method          | AP         | AP_easy    | AP_medium   | AP_hard    |
|-----------------|:----------:|:----------:|:-----------:|:----------:|
| Xiao et al. [1] | 60.8       | 71.4       | 61.2        | 51.2       |
| Li et al. [2]   | **66.6**   | **75.7**   | 66.3        | **57.4**   |
| Ours            | 65.5       | 75.2       | **66.6**    | 53.1       |

## JTA Extension
For our experiments we created an extension to the dataset provided by Fabbri et al. which can be requested [here](link zum zip datensatz mit annotationen). Since we have our own format of annotation, we provide adjusted annotation for the original JTA dataset as well, which can be found [here](annotationen für JTA). We think that Fabbri et al. did great work, so please [cite them](https://github.com/fabbrimatteo/JTA-Mods) as well when relating to our results.

## Code
Our code for training will be available here soon.

## References
[1] Xiao, B., Wu, H., & Wei, Y. (2018). _Simple baselines for human pose estimation and tracking. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 466-481)_.

[2] Li, J., Wang, C., Zhu, H., Mao, Y., Fang, H. S., & Lu, C. (2018). _CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark. arXiv preprint arXiv:1812.00324_.
