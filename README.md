# Multi-modal-biological-brain-age-prediction

This repository contains the code associated with the publication: [Multimodal biological brain age prediction using magnetic resonance imaging and angiography with the identification of predictive regions](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.25805)

* sfcn_model.py: Contains the brain age prediction model implementation.
* multimodal_saliency_maps.py: Contains the smoothgrad regression salincy maps implementation, in a multimodal setting.
* tfceR.R: Extracts significant clusters from the SmoothGrad saliency maps using Probabilistic Threshold-free Cluster Enhancement.


If you use this code, please cite:

@article{mouches2022multimodal,
  title={Multimodal biological brain age prediction using magnetic resonance imaging and angiography with the identification of predictive regions},
  author={Mouches, Pauline and Wilms, Matthias and Rajashekar, Deepthi and Langner, S{\"o}nke and Forkert, Nils D},
  journal={Human Brain Mapping},
  year={2022},
  publisher={Wiley Online Library}
}
