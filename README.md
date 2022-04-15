# RESAIL&#x26F5;: Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis (CVPR2022)

<div align="center">
<img src="./contents/multi-results.png">
</div>



>**Abstract:** _Semantic image synthesis is a challenging task with
many practical applications. Albeit remarkable progress
has been made in semantic image synthesis with spatially-
adaptive normalization, existing methods usually normal-
ize the feature activations under the coarse-level guidance
(e.g., semantic class). However, different parts of a seman-
tic object (e.g., wheel and window of car) are quite differ-
ent in structures and textures, making blurry synthesis re-
sults usually inevitable due to the missing of fine-grained
guidance. In this paper, we propose a novel normaliza-
tion module, termed as REtrieval-based Spatially Adap-
tIve normaLization (RESAIL), for introducing pixel level
fine-grained guidance to the normalization architecture.
Specifically, we first present a retrieval paradigm by find-
ing a content patch of the same semantic class from train-
ing set with the most similar shape to each test seman-
tic mask. Then, the retrieved patches are composited into
retrieval-based guidance, which can be used by RESAIL for
pixel level fine-grained modulation on feature activations,
thereby greatly mitigating blurry synthesis results. More-
over, distorted ground-truth images are also utilized as al-
ternatives of retrieval-based guidance for feature normal-
ization, further benefiting model training and improving vi-
sual quality of generated images. Experiments on several
challenging datasets show that our RESAIL performs favor-
ably against state-of-the-arts in terms of quantitative met-
rics, visual quality, and subjective evaluation._

<div align="center">
<img src="./contents/training_art.png" width="88%" height="88%">
</div>

[Paper in arxiv.org]((https://arxiv.org/abs/2204.02854))
