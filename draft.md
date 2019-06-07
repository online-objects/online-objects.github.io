## Abstract

We propose a self-supervised approach for learning representations of objects from monocular videos and
demonstrate it is particularly useful in situated settings such as robotics.
The main contributions of this paper are:
1) a self-supervising objective trained with contrastive learning that can discover and disentangle object attributes from video without using any labels;
2) we leverage object self-supervision for online adaptation: the longer our online model looks at objects in a video, the lower the object identification error, while the offline baseline remains with a large fixed error;
3) to explore the possibilities of a system entirely free of human supervision, we let a robot collect its own data, train on this data with our self-supervise scheme, and then show the robot can point to objects similar to the one presented in front of it, demonstrating generalization of object attributes.

An interesting and perhaps surprising finding of this approach is that given a limited set of objects, object correspondences will naturally emerge when using contrastive learning without requiring explicit positive pairs.
Videos illustrating online object adaptation and robotic pointing are available as supplementary material.

[here]: https://online-objects.github.io

<!-- <div class="figure">
<video class="b-lazy" data-src="assets/mp4/lmp_8tasks960x400.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: a single Play-LMP policy (right) performing 8 tasks in a row by being conditioned on 8 different goals provided by the user (right).
</figcaption>
</div> -->

______

<!-- <div class="figure">
<img src="assets/fig/lmp_teaser5.svg" style="margin: 0; width: 60%;"/>
<figcaption>
Figure 2. Play-LMP: A single model that self-supervises control from play data, then generalizes to a wide variety of manipulation tasks. Play-LMP 1) samples random windows of experience from a memory of play data. 2) learns to recognize and organize a repertoire of behaviors executed during play in a latent plan space, 3) trains a policy, conditioned on current state, goal state, and a sampled latent plan to reconstruct the actions in the selected window.
Latent plan space is shaped by two stochastic encoders: plan recognition and plan proposal. Plan recognition takes the entire sequence, recognizing the exact behavior executed. Plan proposal takes the initial and final state, outputting a distribution over all possible behaviors that connect initial state to final. We minimize the KL divergence between the two encoders, making plan proposal assign high likelihood to behaviors that were actually executed during play.
</figcaption>
</div> -->

## Introduction

<!-- \begin{figure}[h]
  \begin{center}
  \includegraphics[width=1.0\linewidth]{images/teaser_views_v3.jpg}\\[2mm]
  \includegraphics[width=1.0\linewidth]{images/teaser_chart_v2.pdf}
  \end{center}
    \vspace{-6mm}
  \caption{\textbf{The longer our model looks at objects in a video, the lower the object identification error.} Top: example frames of a work bench video along with the detected objects. Bottom:~result of online training on the same video. Our model self-supervises object representations as the video progresses and converges to 2\% error while the offline baseline remains at 52\% error.}
  \label{fig:online}  
  \vspace{-5mm}
\end{figure} -->

<div class="figure">
<img src="assets/fig/teaser_chart_v2.svg" style="margin: 0; width: 100%;"/>
<video class="b-lazy" data-src="assets/mp4/work_bench_consecutive_1.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: the longer our model looks at objects in a video, the lower the object identification error. Top: example frames of a work bench video along with the detected objects. Bottom: result of online training on the same video. Our model self-supervises object representations as the video progresses and converges to 2% error while the offline baseline remains at 52% error.
</figcaption>
</div>

One of the biggest challenges in real world robotics is robustness and adaptability to new situations. A robot deployed in the real world is likely to encounter a number of objects it has never seen before. Even if it can identify the class of an object, it may be useful to recognize a particular instance of it.
Relying on human supervision in this context is unrealistic.
Instead if a robot can self-supervise its understanding of objects, it can adapt to new situations when using online learning.
Online self-supervision is key to robustness and adaptability and arguably a prerequisite to real-world deployment. Moreover, removing human supervision has the potential to enable learning richer and less biased continuous representations than those obtained by supervised training and a limited set of discrete labels. Unbiased representations can prove useful in unknown future environments different from the ones seen during supervision, a typical challenge for robotics. Furthermore, the ability to autonomously train to recognize and differentiate previously unseen objects as well as to infer general properties and attributes is an important skill for robotic agents.


In this work we focus on situated settings (i.e. an agent is embedded in an environment), which allows us 
to use temporal continuity as the basis for self-supervising correspondences between different views of objects. We present a self-supervised method that learns representations to disentangle perceptual and semantic object attributes such as class, function, and color. We automatically acquire training data by capturing videos with a real robot; a robot base moves around a table to capture objects in various arrangements. Assuming a pre-existing objectness detector, we extract objects from random frames of a scene containing the same objects, and let a metric learning system decide how to assign positive and negative pairs of embeddings. Representations that generalize across objects naturally emerge despite not being given groundtruth matches. Unlike previous methods, we abstain from employing additional self-supervisory training signals such as depth or those used for tracking. The only input to the system are monocular videos. This simplifies data collection and allows our embedding to integrate into existing end-to-end learning pipelines. We demonstrate that a trained Object-Contrastive Network (OCN) embedding allows us to reliably identify object instances based on their visual features such as color and shape. Moreover, we show that objects are also organized along their semantic or functional properties. For example, a cup might not only be associated with other cups, but also with other containers like bowls or vases. 

Figure 1 shows the effectiveness of online self-supervision: by training on randomly selected frames of a continuous video sequence (top) OCN can adapt to the present objects and thereby lower the object identification error. While the supervised baseline remains at a constant high error rate (52.4%), OCN converges to a 2.2\% error. The graph (bottom) shows the object identification error obtained by training on  progressively longer sub-sequences of a 200 seconds video. 



The key contributions of this work are: 
1) a self-supervising objective trained with contrastive learning that can discover and disentangle object attributes from video without using any labels;
2) we leverage object self-supervision for online adaptation: the longer our online model looks at objects in a video, the lower the object identification error, while the offline baseline remains with a large fixed error;
3) to explore the possibilities of a system entirely free of human supervision: we let a robot collect its own data, then train on this data with our self-supervised training scheme, and show the robot can point to objects similar to the one presented in front of it, demonstrating generalization of object attributes.




## Method

We propose a model called Object-Contrastive Network (OCN) trained with a metric learning loss (Figure 2). The approach is very simple: 
1) extract object bounding boxes using a general off-the-shelf objectness detector <dt-cite key="NIPS2015_5638"></dt-cite>, 
2) train a deep object model on each cropped image extracted from any random pair of frames from the video, using the following training objective: nearest neighbors in the embedding space are pulled together from different frames while being pushed away from the other objects from any frame (using n-pairs loss <dt-cite key="NIPS2016_6200"></dt-cite>). This does not rely on knowing the true correspondence between objects. 

The fact that this works at all despite not using any labels might be surprising. One of the main findings of this paper is that given a limited set of objects, object correspondences will naturally emerge when using metric learning. One advantage of self-supervising object representation is that these continuous representations are not biased by or limited to a discrete set of labels determined by human annotators. We show these embeddings discover and disentangle object attributes and generalize to previously unseen environments.

<div class="figure">
<img src="assets/fig/overview5.svg" style="margin: 0; width: 100%;"/>
<figcaption>
Figure2: Object-Contrastive Networks (OCN): by attracting nearest neighbors in embedding space and repulsing others using metric learning, continuous object representations naturally emerge. In a video collected by a robot looking at a table from different viewpoints, objects are extracted from random pairs of frames. Given two lists of objects, each object is attracted to its closest neighbor while being pushed against all other objects. Noisy repulsion may occur when the same object across viewpoint is not matched against itself. However the learning still converges towards disentangled and semantically meaningful object representations. </figcaption>
</div>

We propose a self-supervised approach to learn object representations for the following reasons: (1) make data collection simple and scalable, (2) increase autonomy in robotics by continuously learning about new objects without assistance, (3) discover continuous representations that are richer and more nuanced than the discrete set of attributes that humans might provide as supervision which may not match future and new environments. 
All these objectives require a method that can learn about objects and differentiate them without supervision. To bootstrap our learning signal we leverage two assumptions: (1) we are provided with a general objectness model so that we can attend to individual objects in a scene, (2) during an observation sequence the same objects will be present in most frames (this can later be relaxed by using an approximate estimation of ego-motion). Given a video sequence around a scene containing multiple objects, we randomly select two frames $I$ and $\hat{I}$ in the sequence and detect the objects present in each image. Let us assume $N$ and $M$ objects are detected in image $I$ and $\hat{I}$, respectively. Each of the $n$-th and $m$-th cropped object images are embedded in a low dimensional space, organized by a metric learning objective. Unlike traditional methods which rely on human-provided similarity labels to drive metric learning, we use a self-supervised approach to mine synthetic similarity labels.

**Objectness Detection:** To detect objects, we use Faster-RCNN <dt-cite key="NIPS2016_6200"></dt-cite> trained on the COCO object detection dataset <dt-cite key="10.1007/978-3-319-10602-1_48"></dt-cite>. Faster-RCNN detects objects in two stages: first generate class-agnostic bounding box proposals of all objects present in an image (Figure 1), second associate detected objects with class labels. We use OCN to discover object attributes, and only rely on the first *objectness* stage of Faster-R-CNN to detect object candidates. 

**Metric Loss for Object Disentanglement:** We denote a cropped object image by $x \in \mathcal{X}$ and compute its embedding based on a convolutional neural network $f(x): \mathcal{X} \rightarrow K$.
Note that for simplicity we may omit $x$ from $f(x)$ while $f$ inherits all superscripts and subscripts. Let us consider two pairs of images $I$ and $\hat{I}$ that are taken at random from the same contiguous observation sequence. Let us also assume there are $n$ and $m$ objects detected in $I$ and $\hat{I}$ respectively. We denote the $n$-th and $m$-th objects in the images $I$ and $\hat{I}$ as $x_n^{I}$ and $x_m^{\hat{I}}$, respectively. We compute the distance matrix $D_{n,m} = \sqrt{(f_{n}^{I} - f_{m}^{\hat{I}}})^2,~n\in1..N,~m\in1..M$. For every embedded *anchor* $f_{n}^{I},~n\in1..N$, we select a *positive* embedding $f_{m}^{\hat{I}}$ with minimum distance as *positive*: $f_{n+}^{\hat{I}} = argmin(D_{n,m})$. 

Given a batch of (*anchor*, *positive*) pairs $\{(x_i, x_i^+)\}_{i=1}^N$, the n-pair loss is defined as follows <dt-cite key="NIPS2016_6200"></dt-cite>: 


$\mathcal{L}_{N-pair}\big(\{(x_i, x_i^+)\}_{i=1}^N;f\big) = 
\frac{1}{N} \sum_{i=1}^N log \Big(1 + \sum_{j \neq i} exp(f_i^\intercal f_j^+ - f_i^\intercal f_i^+) \Big)$

The loss learns embeddings that identify ground truth (anchor, positive)-pairs from all other (anchor, negative)-pairs in the same batch. It is formulated as a sum of softmax multi-class cross-entropy losses over a batch, encouraging the inner product of each (anchor, positive)-pair ($f_i$, $f_i^+$) to be larger than all (anchor, negative)-pairs ($f_i$, $f_{j\neq i}^+$). The final OCN training objective over a sequence is the sum of npairs losses over all pairs of individual frames:

$\mathcal{L}_{OCN} = \mathcal{L}_{N-pair}\big(\{(x_n^{I}, x_{n+}^{\hat{I}})\}_{n=1}^N;f\big) + \mathcal{L}_{N-pair}\big(\{(x_m^{\hat{I}}, x_{m+}^{I})\}_{m=1}^M;f\big)$

**Architecture and Embedding Space:** OCN takes a standard ResNet50 architecture until layer *global\_pool* and initializes it with ImageNet pre-trained weights. We then add three additional ResNet convolutional layers and a fully connected layer to produce the final embedding. The network is trained with the n-pairs metric learning loss as discussed in Section "Metric Loss for Object Disentanglement". Our architecture is depicted in Figure 3.

<div class="figure">
<img src="assets/fig/models2.svg" style="margin: 0; width: 100%;"/>
<figcaption>
Figure 3: Models and baselines: for comparison purposes all models evaluated in Section "Results" share the same architecture of a standard ResNet50 model followed by additional layers. While the architectures are shared, the weights are not across models. While the unsupervised model (left) does not require supervision labels, the 'softmax' baseline as well as the supervised evaluations (right) use attributes labels provided with each object. We evaluate the quality of the embeddings with two types of classifiers: linear and nearest neighbor.</figcaption>
</div>

**Object-centric Embeding Space:** By using multiple views of the same scene and by attending to individual objects, our architecture allows us to differentiate subtle variations of object attributes. Observing the same object across different views facilitates learning invariance to scene-specific properties, such as scale, occlusion, lighting, and background, as each frame exhibits variations of these factors. The network solves the metric loss by representing object-centric attributes, such as shape, function, or color, as these are consistent for (anchor, positive)-pairs, and dissimilar for (anchor, negative)-pairs.  

**Discussion:** One might expect that this approach may only work if it is given an initialization so that matching the same object across multiple frames is more likely than random chance. While ImageNet pretraining certainly helps convergence as shown in Table 3, it is not a requirement to learn meaningful representations as shown in Section "Random Weights". When all weights are random and no labels are provided, what can drive the network to consistently converge to meaningful embeddings? We estimate that the co-occurrence of the following hypotheses drives this convergence: (1) objects often remains visually similar to themselves across multiple viewpoints, (2) limiting the possible object matches within a scene increases the likelihood of a positive match, (3) the low-dimensionality of the embedding space forces the model to generalize by sharing abstract features across objects, (4) the smoothness of embeddings learned with metric learning facilitates convergence when supervision signals are weak, and (5) occasional true-positive matches (even by chance) yield more coherent gradients than false-positive matches which produce inconsistent gradients and dissipate as noise, leading over time to an acceleration of consistent gradients and stronger initial supervision signal.

## Experiments

**Online Results:** we quantitatively evaluate the online adaptation capabilities of our model through the object identification error of entirely novel objects. In Figure 1 we show that a model observing objects for a few minutes from different angles can self-teach to identify them almost perfectly while the offline supervised approach cannot. OCN is trained on the first 5, 10, 20, 40, 80, and 160 seconds of the 200 seconds video, then evaluated on the identification error of the last 40 seconds of the video for each phase. The supervised offline baseline stays at a 52.4% error, while OCN improves down to 2% error after 80s, a 25x error reduction.

**Robotic Experiments:** here we let a robot collect its own data by looking at a table from multiple angles (Figure 2 and Figure 5).
It then trains itself with OCN on that data, and is asked to point to objects similar to the one presented in front of it. Objects can be similar in terms of shape, color or class. If able to perform that task, the robot has learned to distinguish and recognize these attributes entirely on its own, from scratch and by collecting its own data. We find in Table 7 that the robot is able to perform the pointing task with 72% recognition accuracy of 5 classes, and 89% recognition accuracy of the binary is-container attribute.

**Offline Analysis:** to analyze what our model is able to disentangle, we quantitatively evaluate performance on a large-scale synthetic dataset with 12k object models (e.g. Figure 10), as well as on a real dataset collected by a robot and show that our unsupervised object understanding generalizes to previously unseen objects. In Table 3 we find that our self-supervised model closely follows its supervised equivalent baseline when trained with metric learning. As expected the cross-entropy/softmax supervised baseline approach performs best and establishes the error lower bound while the ResNet50 baseline are upper-bound results.

## Data Collection and Training

We generated three datasets of real and synthetic objects for our experiments. For the real data we arrange objects in table-top configurations and use frames from continuous camera trajectories. The labeled synthetic data is generated from renderings of 3D objects in a similar configuration. Details about the datasets are reported in Table 4.

**Real Data for Online Training:** for the online adaptation experiment, we captured videos of table-top object configurations in the 5 environments (categories): kids room, kitchen, living room, office, and work bench (Figures 1, 4, and 6). We show objects common to each environment (e.g. toys for kids room, tools for work bench) and arrange them randomly; we captured 3 videos for each environment and used 75 unique objects. To allow capturing the objects from multiple view points we use a head-mounted camera and interact with the objects (e.g. turning or flipping them). Additionally, we captured 5 videos of more challenging object configurations (referred to as "challenging") with cluttered objects or where objects are not permanently in view. Finally, we selected 5 videos from the Epic-Kitchens <dt-cite key="Damen2018EPICKITCHENS"></dt-cite> dataset to show that OCN can also operate on even more realistic video sequences. 

From all these videos we take the first 200 seconds and sample the sequence with 15 FPS to extract 3,000 frames. We then use the first 2,400 frames (160s) for training OCN and the remaining 600 frames (40s) for evaluation. We manually select up to 30 reference objects (those we  interacted with) as cropped images for each video in order of their appearance from the beginning of the video (Figure 14). Then we use object detection to find the bounding boxes of these objects in the video sequence and manually correct these boxes (add, delete) in case object detection did not identify an object. This allows us to prevent artifacts of the object detection to interfere with the evaluation of OCN.


<div class="figure">
<img src="assets/fig/online_dataset.jpg" style="margin: 0; width: 100%;"/>
<figcaption>
Figure 4: Six of the environments we used for our self-supervised online experiment. Top: living room, office, kitchen. Bottom: one of our more challenging scenes, and two examples of the Epic-Kitchens.</figcaption>
</div>

**Automatic Real Data Collection:** to explore the possibilities of a system entirely free of human supervision we automated the real world data collection by using a mobile robot equipped with an HD camera (Figure 11). For this dataset we use 187 unique object instances spread across six categories including 'balls', 'bottles & cans', 'bowls', 'cups & mugs', 'glasses', and 'plates'.  Table 5 provides details about the number of objects in each category and how they are split between training, test, and validation. Note that we distinguish between cups & mugs and glasses categories based on whether it has a handle. Figure 5 shows our entire object dataset. 

<div class="figure">
<img src="assets/fig/dataset_v2.jpg" style="margin: 0; width: 100%;"/>
<figcaption>
Figure 5: We use 187 unique object instance in the real world experiments: 110 object for training (left), 43 objects for test (center), and 34 objects for validation (right). The degree of similarity makes it harder to differentiate these objects.</figcaption>
</div>

At each run, we place about 10 objects on the table and then trigger the capturing process by having the robot rotate around the table by 90 degrees (Figure 11). On average 130 images are captured at each run. We select random pairs of frames from each trajectory for training OCN. We performed 345, 109, and 122 runs of data collection for training, test, and validation dataset. In total 43,084 images were captured for OCN training and 15,061 and 16,385 were used for test and validation, respectively.


**Synthetic Data Generation:** to generate diverse object configurations we use 12 categories (airplane, car, chair, cup, bottle, bowl, guitars, keyboard, lamp, monitor, radio, vase) from  ModelNet<dt-cite key="WuSKYZTX15"></dt-cite>. The selected categories cover around 8k models of the 12k models available in the entire dataset. ModelNet provides the object models in a 80-20 split for training and testing. We further split the testing data into models for test and validation, resulting in a 80-10-10 split for training, validation, and test. For validation purposes, we manually assign each model labels describing the semantic and functional properties of the object, including the labels 'class', 'has lid', 'has wheels', 'has buttons', 'has flat surface', 'has legs', 'is container', 'is sittable', 'is device'.

We randomly define the number of objects (up to 20) in a scene  (Figure 12). Further, we randomly define the positions of the objects and vary their sizes, both so that they do not intersect. Additionally, each object is assigned one of eight predefined colors. We use this setup to generate 100K scenes for training, and 50K scenes for each, validation and testing. For each scene we generate 10 views and select random combination of two views for detecting objects. In total we produced 400K views (200K pairs) for training and 50K views (25K pairs) for each, validation and testing.  

**Training:** OCN is trained based on two views of the same synthetic or real scene. We randomly pick two frames of a video sequence and detect objects to produce two sets of cropped images. The distance matrix $D_{n,m}$ (Section "Metric Loss for Object Disentanglement") is constructed based on the individually detected objects for each of the two frames. The object detector was not specifically trained on any of our datasets. 


## Experimental Results

We evaluated the effectiveness of OCN embeddings on identifying objects through self-supervised online training, a real robotics pointing tasks, and  large-scale synthetic data. 

**Online Object Identification:**

Our self-supervised online training scheme enables to train and to evaluate on unseen objects and scenes. This is of utmost importance for robotic agents to ensure adaptability and robustness in real world scenes. To show the potential of our method for these situations we use OCN embeddings to identify instances of objects across multiple views and over time. 