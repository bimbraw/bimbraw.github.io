---
title: "GPT Sonography: Decoding Hand Gestures from Forearm Ultrasound with GPT-4o"
date: 2026-05-17
permalink: /posts/2026/05/gpt-sonography/
categories:
  - Research
tags:
  - ultrasound
  - GPT-4o
  - large vision-language models
  - in-context learning
  - retrieval augmented generation
  - human-machine interaction
author_profile: true
share: false
related: false
---

<div class="research-hero">
  <div class="research-kicker">IEEE Access 2026 · GPT-4o · Forearm Ultrasound · In-Context Learning</div>

  <p>
    Can a large vision-language model decode hand gestures from forearm ultrasound images without task-specific fine-tuning?
    In <em>GPT Sonography</em>, we studied this question by framing forearm ultrasound gesture decoding as an
    in-context learning problem with GPT-4o.
  </p>

  <div class="research-actions">
    <a href="https://ieeexplore.ieee.org/abstract/document/11494645" target="_blank" rel="noopener noreferrer">IEEE Access</a>
    <a href="https://openreview.net/forum?id=k1x4G8NttM" target="_blank" rel="noopener noreferrer">NeurIPS Paper</a>
  </div>
</div>

<div class="blog-figure">
  <img src="/images/Graphical_Abstract_GPT_Sonography.png" alt="Graphical abstract for GPT Sonography">
  <p>
    Graphical abstract: Forearm ultrasound images are provided to GPT-4o with text instructions
    and in-context examples for hand gesture decoding.
  </p>
</div>

<div class="research-metrics">
  <div class="research-metric">
    <div class="metric-value">19.3% → 74.0%</div>
    <div class="metric-label">Within-session accuracy improved from 0-shot to 2-shot ICL</div>
  </div>

  <div class="research-metric">
    <div class="metric-value">20.0% → 61.3%</div>
    <div class="metric-label">Cross-session accuracy improved from 0-shot to 3-shot ICL</div>
  </div>

  <div class="research-metric">
    <div class="metric-value">No fine-tuning</div>
    <div class="metric-label">The LVLM was used as a frozen model through prompting</div>
  </div>
</div>

<div class="research-metrics">
  <div class="research-metric">
    <div class="metric-value">99–100%</div>
    <div class="metric-label">Within-session RAG-ICL reached near-ceiling accuracy from 1-shot to 3-shot prompting</div>
  </div>

  <div class="research-metric">
    <div class="metric-value">RAG-ICL</div>
    <div class="metric-label">Retrieval-augmented prompting outperformed matching-network baselines using pixel and SigLIP2 similarity</div>
  </div>

  <div class="research-metric">
    <div class="metric-value">Scaling effect</div>
    <div class="metric-label">Larger GPT variants produced higher and more stable 1-shot ICL accuracy than mini and nano variants</div>
  </div>
</div>

## Motivation

Forearm ultrasound is a promising sensing modality for human-machine interaction because it can image internal muscle and tendon deformation associated with hand movement. In prior work, ultrasound-based gesture decoding typically required training task-specific machine learning models using labeled ultrasound data.

However, collecting labeled biosignal and medical imaging data can be expensive, subject-dependent, and sensitive to session-to-session variability. Large vision-language models provide a different interface: instead of updating model parameters, they can use task instructions and examples provided directly in the prompt.

The central question of this work was:

$$
\text{Can a frozen LVLM perform ultrasound gesture decoding through in-context learning?}
$$

## Problem Formulation

Let an ultrasound image be

$$
x_i \in \mathbb{R}^{H \times W},
$$

and let the corresponding hand gesture label be

$$
y_i \in \{1,2,\ldots,C\}.
$$

In conventional supervised learning, we train a model \(f_\theta\) by minimizing a loss over labeled examples:

$$
\theta^\star =
\arg\min_{\theta}
\sum_{i=1}^{N}
\mathcal{L}(f_\theta(x_i), y_i).
$$

In GPT Sonography, we do not fine-tune the model. Instead, we construct a prompt containing a task description and, optionally, a small set of labeled examples:

$$
\mathcal{P}
=
\{(x_1, y_1), \ldots, (x_k, y_k), x_{\text{test}}\}.
$$

The LVLM then predicts the gesture label directly from the prompt:

$$
\hat{y}_{\text{test}}
=
\operatorname{LVLM}(\mathcal{P}).
$$

This changes the adaptation mechanism from parameter learning to prompt conditioning.

## Zero-Shot and Few-Shot ICL

We evaluated zero-shot and few-shot prompting.

In the zero-shot setting, the model receives the test ultrasound image and a text description of the classification task, but no labeled ultrasound examples. This is difficult because forearm ultrasound is a specialized image domain, and the visual differences between gestures are not necessarily semantic in the same way as natural images.

In the few-shot setting, we include \(k\) labeled examples in the prompt:

$$
\mathcal{P}_k =
\{(x_1,y_1), \ldots, (x_k,y_k), x_{\text{test}}\}.
$$

The model can then compare the test image against the labeled examples in context. This substantially improved performance.

Within-session accuracy improved from **19.3%** in the zero-shot setting to **74.0%** with 2-shot in-context learning. Cross-session accuracy improved from **20.0%** in the zero-shot setting to **61.3%** with 3-shot in-context learning.

## Retrieval-Augmented In-Context Learning

A key component of the method was retrieval-augmented in-context learning. Instead of selecting support examples randomly, we retrieve examples that are visually relevant to the test image and include them in the prompt.

Given a query ultrasound image $x_q$ and a labeled support set $\mathcal{S}$, the retrieved support examples can be written as

$$
\mathcal{N}_k(x_q)
=
\operatorname{TopK}_{(x_j,y_j)\in\mathcal{S}}
s(x_q, x_j),
$$

where $s(x_q,x_j)$ denotes the image similarity between the query image and a candidate support image.

The retrieved examples are then inserted into the LVLM prompt along with their labels. In this framing, the in-context examples are not merely additional labels; they define the local visual comparison set available to the model.

This retrieval step was especially effective in within-session experiments, where retrieved ultrasound frames can be visually close to the query frame. At the same time, this result requires careful interpretation. Ultrasound frame-based tasks can contain strong local similarity, so retrieval behavior, frame similarity, and generalization need to be analyzed carefully.

## Interpretation

The main result is that zero-shot GPT-4o performed near chance on this specialized ultrasound task, but performance improved substantially when labeled examples were provided in context.

This suggests that the LVLM should not be interpreted as a pretrained ultrasound gesture expert. Rather, it can act as a flexible visual reasoning model that uses task instructions and example-based context to perform a specialized classification task.

The key distinction is the adaptation mechanism. Conventional supervised learning adapts through parameter updates:

$$
\theta^\star
=
\arg\min_{\theta}
\sum_i
\mathcal{L}(f_\theta(x_i), y_i).
$$

In-context learning adapts through the prompt:

$$
\hat{y}
=
\operatorname{LVLM}
\left(
\text{instructions},
\text{examples},
x_{\text{test}}
\right).
$$

## Implications for biosensing, medical imaging and human-machine interfacing

For biosignal and medical imaging problems, full fine-tuning can be difficult because datasets are often small, labels are expensive, and performance can vary across subjects, sessions, and sensor placements.

This work does not suggest that LVLMs replace task-specific ultrasound models today. Instead, it shows that ICL may provide a useful complementary approach for specialized sensing problems where rapid adaptation and data efficiency are important.

For forearm ultrasound-based human-machine interfacing, this is especially relevant because the sensing problem is highly subject- and session-dependent. A prompt-based interface could allow future systems to adapt using a small number of representative examples rather than requiring a full model retraining pipeline.

## Connection to My PhD Work

This journal article extends our NeurIPS 2024 AIM-FM workshop paper and is also part of my PhD thesis on biosignal-based deep learning for human-machine interfacing, robotic teleoperation, and control.

Across my PhD work, I studied how forearm ultrasound can be used to estimate hand gestures, hand pose, and force-related information. GPT Sonography explores a newer question within that broader direction: whether foundation models can provide a lightweight interface for decoding specialized biosignal images without conventional fine-tuning.

## Citation

```bibtex
@article{bimbraw2026gptsonography,
  title={GPT Sonography: Hand Gesture Decoding From Forearm Ultrasound Images via a Large Vision-Language Model},
  author={Bimbraw, Keshav and Wang, Ye and Liu, Jing and Koike-Akino, Toshiaki},
  journal={IEEE Access},
  year={2026},
  doi={10.1109/ACCESS.2026.3687477}
}
```

## Note

This post is intended as a soft introduction to the work rather than a complete technical treatment. The full paper contains more detailed experiments, ablations, comparisons, and discussion on prompting strategies, retrieval-augmented in-context learning, and the challenges of interpreting ultrasound frame-based classification results.

Please refer to the paper for the complete methodology and results. If you have questions or would like to discuss the work, feel free to reach out. The experimental design and discussion around LVLMs for specialized biosignal imaging tasks are especially interesting.

## Acknowledgment

I am grateful to Toshiaki Koike-Akino, Jing Liu, and Ye Wang for their guidance and mentorship during my internship at Mitsubishi Electric Research Laboratories (MERL) in Summer 2024. I especially want to thank Toshi for his persistence, encouragement, and thoughtful guidance in helping bring this work through to the final journal publication.