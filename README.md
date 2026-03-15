## Neural Additive Models: Interpretable Machine Learning with Neural Nets

# [![Website](https://img.shields.io/badge/www-Website-green)](https://neural-additive-models.github.io) [![Visualization Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E3_t7Inhol-qVPmFNq1Otj9sWt1vU_DQ?usp=sharing)


This repository contains open-source code
for the paper
[Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912).

<img src="https://i.imgur.com/Hvb7sb2.jpg" width="50%" alt="Neural Additive Model" >

Currently, the repository ships a PyTorch implementation of NAM together with
PyTorch-based training, evaluation, and visualization scripts. The training
stack supports automatic device selection in the order `cuda -> mps -> cpu`,
while still exposing a `--device` flag for explicit overrides. The
`nam_train.py` file provides the example of a training script on a single
dataset split.

Use `./run.sh` test script to ensure that the setup is correct.

## Multi-task NAMs
The code for multi task NAMs can be found at [https://github.com/lemeln/nam](https://github.com/lemeln/nam).

## COMPAS Single-task and Multitask Experiment

This repository now includes a local COMPAS experiment runner that reproduces
the paper's single-task vs multitask NAM setting without changing the project
layout. The script trains:

- a single-task NAM ensemble for recidivism prediction
- a multitask NAM ensemble with separate outputs for women and men
- 5-fold cross-validation AUC summaries
- a Figure 10 style visualization comparing single-task and multitask shape plots

Example commands:

```bash
python compas_experiment.py --mode cv --n_models 20 --training_epochs 50
python compas_experiment.py --mode figure --n_models 20 --training_epochs 50
python compas_experiment.py --mode all --n_models 100 --training_epochs 80
```

Outputs are written under `output/compas_experiment/`.

## compass_FM Experiment

The repository also includes a `compass_FM` experiment for overall COMPAS
recidivism AUROC comparison on the full dataset without splitting by gender.
It compares:

- a standard single-task NAM baseline
- a NAM augmented with a `FactorizedMachine` interaction term added directly to the output logit

Example command:

```bash
python compass_FM.py --n_models 20 --training_epochs 50 --fm_rank 8
```

Outputs are written under `output/compass_FM/`.

## Dependencies

The code uses these packages:

- torch>=2.1
- numpy>=1.24,<2
- scikit-learn>=1.3
- pandas>=2.1
- matplotlib>=3.8

## Datasets

The datasets used in the paper (except MIMIC-II) can be found in the <a href="https://console.cloud.google.com/storage/browser/nam_datasets/data"> public GCP bucket</a> `gs://nam_datasets/data`, which can be downloaded using [gsutil][gsutil]. To install gsutil, follow the instructions [here][gsutil_install]. The preprocessed version of MIMIC-II dataset, used in the NAM paper, can be
shared only if you provide us with the signed data use agreement to the MIMIC-III Clinical
Database on the <a href="https://mimic.mit.edu/docs/gettingstarted/#physionet-credentialing">PhysioNet website</a>.

Citing
------
If you use this code in your research, please cite the following paper:

> Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana,
> R., & Hinton, G. E. (2021). Neural additive models: Interpretable machine > learning with neural nets. Advances in Neural Information Processing
> Systems, 34.

    @article{agarwal2021neural,
      title={Neural additive models: Interpretable machine learning with neural nets},
      author={Agarwal, Rishabh and Melnick, Levi and Frosst, Nicholas and Zhang, Xuezhou and Lengerich, Ben and Caruana, Rich and Hinton, Geoffrey E},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }

---

*Disclaimer about COMPAS dataset: It is important to note that
developing a machine learning model to predict pre-trial detention has a
number of important ethical considerations. You can learn more about these
issues in the Partnership on AI
[Report on Algorithmic Risk Assessment Tools in the U.S. Criminal Justice System](https://www.partnershiponai.org/report-on-machine-learning-in-risk-assessment-tools-in-the-u-s-criminal-justice-system/).
The Partnership on AI is a multi-stakeholder organization -- of which Google
is a member -- that creates guidelines around AI.*

*We’re using the COMPAS dataset only as an example of how to identify and
remediate fairness concerns in data. This dataset is canonical in the
algorithmic fairness literature.*

*Disclaimer: This is not an official Google product.*

[gsutil_install]: https://cloud.google.com/storage/docs/gsutil_install#install
[gsutil]: https://cloud.google.com/storage/docs/gsutil
