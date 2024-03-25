# EpiSurfEmb

This is the code for [Multi-view object pose estimation from correspondence distributions and epipolar geometry
ICRA 2023](https://arxiv.org/pdf/2210.00924.pdf).

The tless-model used for the experiments is provided under releases. 
In contrast to [original surfemb](https://github.com/rasmushaugaard/surfemb), this model is trained on amodal mask crops, used in EpiSurfEmb.

For more information, see the [surfemb paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Haugaard_SurfEmb_Dense_and_Continuous_Correspondence_Distributions_for_Object_Pose_Estimation_CVPR_2022_paper.pdf).


Citing EpiSurfEmb:
```bibtex
@inproceedings{haugaard2023multi,
  title={Multi-view object pose estimation from correspondence distributions and epipolar geometry},
  author={Haugaard, Rasmus Laurvig and Iversen, Thorbjorn Mosekjaer},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={1786--1792},
  year={2023},
  organization={IEEE}
}
```

Citing SurfEmb:
```bibtex
@inproceedings{haugaard2022surfemb,
  title={Surfemb: Dense and continuous correspondence distributions for object pose estimation with learnt surface embeddings},
  author={Haugaard, Rasmus Laurvig and Buch, Anders Glent},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6749--6758},
  year={2022}
}
```