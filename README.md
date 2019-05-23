# Human Pose Estimation for Real-World Crowded Scenarios
_Submitted for 2019 IEEE International Conference on Advanced Video and Signal-based Surveillance_

## Abstract
Human pose estimation has recently made significant progress with the adoption of deep convolutional neural networks. Its many applications have attracted tremendous interest in recent years. However, many practical applications require pose estimation for human crowds, which still is a rarely addressed problem. In this work, we explore methods to optimize pose estimation for human crowds, focusing on challenges introduced with dense crowds, such as occlusions, people in close proximity to each other, and partial visibility of people. In order to address these challenges, we evaluate three aspects of a pose detection approach: i) a data augmentation method to introduce robustness to occlusions, ii) the explicit detection of occluded body parts,  and iii) the use of the synthetic generated datasets. The first approach to improve the accuracy in crowded scenarios is to generate occlusions at training time using person and object cutouts from the object recognition dataset COCO (Common Objects in Context). Furthermore, the synthetically generated dataset JTA (Joint Track Auto) is evaluated for the use in real-world crowd applications. In order to overcome the transfer gap of JTA originating from a low pose variety and less dense crowds, an extension dataset is created to ease the use for real-world applications. Additionally, the occlusion flags provided with JTA are utilized to train a model, which explicitly distinguishes between occluded and visible body parts in two distinct branches. The combination of the proposed additions to the baseline method help to improve the overall accuracy by 4.7\% AP and thereby provide comparable results to current state-of-the-art approaches on the respective dataset.

Link to paper: [link]

## Results
* Table #1
* Table #2

## JTA Extension
* Link zum Datensatz mit Annotationen
* inkludiere evtl ergänzende Annotationen für original JTA
* link zum JTA Datensatz

## Code
Our code for training will be available here soon.
