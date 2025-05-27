Code Structure:
Adversarial Detection
   |----Image Contrast Enhancement Module
   |----High-frequency Information Filter Module
   |----Training Detector Module
   |----Detector Module

1.Image Contrast Enhancement Module:
This module includes main_2.m, duqu.m, and rgb_eq.m. By running main_2.m and providing the path to the adversarial examples along with the sliding window size, you can obtain the images processed with local histogram equalization.

2.High-frequency Information Filter Module:
This module includes main_2.m, duqu.m, btws.m, and duqu.m. By running main_2.m and inputting the path to the adversarial examples along with the cutoff frequency, you can obtain the filtered images.

3.Training Detector Module：
This module includes train.py and model.py. By running train.py and setting the appropriate parameters, you can train the detector.

4.Detector Module：
This module includes yuce.py. By running yuce.py, you can select different models to recognize the dataset and output the recognition accuracy.


Citation:
If you use this code, please cite the following paper:
Title: Adversarial Examples Detection with Enhanced Image Difference Features based on Local Histogram Equalization
Author: Zhaoxia Yin, Shaowei Zhu, Hang Su, Jianteng Peng, Wanli Lyu, Bin Luo
Journal: IEEE Transactions on Dependable and Secure Computing
Link: https://ieeexplore.ieee.org/abstract/document/10910243/
Bibtex:
@article{yin2025adversarial,
  title={Adversarial examples detection with enhanced image difference features based on local histogram equalization},
  author={Yin, Zhaoxia and Zhu, Shaowei and Su, Hang and Peng, Jianteng and Lyu, Wanli and Luo, Bin},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2025},
  publisher={IEEE}
}