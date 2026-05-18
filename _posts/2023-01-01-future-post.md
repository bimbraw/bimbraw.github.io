---
title: "Estimating Finger Force from Forearm Ultrasound"
date: 2023-08-28
permalink: /posts/2023/08/ultrasound-force/
categories:
  - Research
tags:
  - ultrasound
  - sonomyography
  - force estimation
  - human-machine interaction
  - robotics
author_profile: true
---

<div class="research-hero">
  <div class="research-kicker">IEEE IUS 2023 · Forearm Ultrasound · Force Estimation</div>

  <p>
    This work studies whether B-mode forearm ultrasound can estimate force exerted by the fingers.
    We evaluate the problem as both binary force/no-force classification and continuous force regression
    using synchronized ultrasound images and force-sensor measurements.
  </p>

  <div class="research-actions">
    <a href="https://doi.org/10.1109/IUS51837.2023.10306652" target="_blank">DOI</a>
    <a href="https://ieeexplore.ieee.org/abstract/document/10306652" target="_blank">IEEE</a>
    <a href="https://bimbraw.github.io/files/2023208571_approved.pdf" target="_blank">PDF</a>
  </div>
</div>

<div class="research-metrics">
  <div class="research-metric">
    <div class="metric-value">98.4%</div>
    <div class="metric-label">Shuffled force/no-force classification accuracy</div>
  </div>

  <div class="research-metric">
    <div class="metric-value">82%</div>
    <div class="metric-label">Non-shuffled force/no-force classification accuracy</div>
  </div>

  <div class="research-metric">
    <div class="metric-value">0.2 N</div>
    <div class="metric-label">Non-shuffled RMSE for continuous force estimation</div>
  </div>
</div>

## Core Idea

Finger force estimation is important for robotic teleoperation, rehabilitation robotics, haptic feedback, and human-machine interaction. In these applications, the system may need to know not only the user's hand gesture, but also the force being applied during interaction.

Forearm ultrasound provides a view of internal muscle and tendon deformation. When a finger applies force, the forearm anatomy changes in a way that can be captured by ultrasound. This work investigates whether those image-level changes can be mapped to finger force.

The learning problem can be written compactly as

$$
x_i \mapsto f_i,
$$

where \(x_i\) is a forearm ultrasound image and \(f_i\) is the corresponding force measurement.

## Task Formulation

We considered two related tasks.

The first task is binary force classification:

$$
y_i = \mathbb{1}[f_i > \tau],
$$

where \(y_i=1\) indicates force application and \(y_i=0\) indicates no force.

The second task is continuous force regression:

$$
\hat{f}_i = g_\theta(x_i).
$$

The classification task asks whether force is being applied. The regression task asks how much force is being applied.

## Experimental Setup

A wireless ultrasound probe was strapped to the forearm to acquire B-mode ultrasound images. A force sensor provided the corresponding ground-truth force measurement. The synchronized ultrasound-force pairs were used for both classification and regression.

Both shuffled and non-shuffled train-test splits were evaluated. The shuffled split measures performance when samples are randomly mixed from the same acquisition distribution. The non-shuffled split is more restrictive because it preserves greater temporal separation between training and testing samples.

## Results

For force/no-force classification, the model achieved **98.4%** average accuracy on shuffled data and **82%** average accuracy on non-shuffled data.

For continuous force estimation, the average RMSE was **0.02 N** for shuffled data and **0.2 N** for non-shuffled data. Since the maximum force range was **5 N**, the non-shuffled RMSE corresponds to

$$
\frac{0.2}{5} \times 100 = 4\%
$$

of the maximum force range.

These results suggest that forearm ultrasound contains force-relevant information and can support both force-state classification and continuous force estimation.

## Interpretation

This result extends ultrasound-based hand decoding beyond gesture classification. Gesture classification estimates hand state. Force estimation adds information about interaction intensity.

For robotic and virtual interaction, this distinction is important. A teleoperated robot, prosthetic device, or haptic interface may need to infer both what the user intends to do and how strongly the user intends to interact with the environment.

In this context, forearm ultrasound is useful because it provides a musculoskeletal view of the forearm rather than only a surface-level electrical measurement.

## Limitations and Next Steps

The non-shuffled results show a clear performance drop relative to the shuffled split, indicating that temporal separation, acquisition variability, and probe placement remain important challenges.

Future work should evaluate cross-session and cross-subject generalization, improve robustness to probe placement, and integrate force estimation into real-time human-machine interfaces. A natural next step is joint estimation of hand gesture, hand pose, and finger-applied force from the same ultrasound stream.

## Citation

```bibtex
@inproceedings{bimbraw2023force,
  title={Estimating Force Exerted by the Fingers Based on Forearm Ultrasound},
  author={Bimbraw, Keshav and Zhang, Haichong K.},
  booktitle={2023 IEEE International Ultrasonics Symposium (IUS)},
  pages={1--4},
  year={2023},
  organization={IEEE},
  doi={10.1109/IUS51837.2023.10306652}
}
```