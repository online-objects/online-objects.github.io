## Abstract

We propose a self-supervised approach for learning representations of objects from monocular videos and
demonstrate it is particularly useful in situated settings such as robotics.
The main contributions of this paper are:
1) <strong>a self-supervised objective trained with contrastive learning</strong> that can discover and disentangle object attributes from video without using any labels;
2) we leverage object self-supervision for online adaptation: <strong>the longer our online model looks at objects in a video, the lower the object identification error,</strong> while the offline baseline remains with a large fixed error;
3) to explore the possibilities of a system entirely free of human supervision, we let a <strong>robot collect its own data</strong>, train on this data with our self-supervise scheme, and then show the robot can point to objects similar to the one presented in front of it, demonstrating generalization of object attributes.

An interesting and perhaps surprising finding of this approach is that given a limited set of objects, object correspondences will naturally emerge when using contrastive learning without requiring explicit positive pairs.
Videos illustrating online object adaptation and robotic pointing are available as supplementary material.

[here]: https://online-objects.github.io

______


## Introduction

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

**Discussion:** One might expect that this approach may only work if it is given an initialization so that matching the same object across multiple frames is more likely than random chance. While ImageNet pretraining certainly helps convergence as shown in Table 6, it is not a requirement to learn meaningful representations as Table 6a. When all weights are random and no labels are provided, what can drive the network to consistently converge to meaningful embeddings? We estimate that the co-occurrence of the following hypotheses drives this convergence: (1) objects often remains visually similar to themselves across multiple viewpoints, (2) limiting the possible object matches within a scene increases the likelihood of a positive match, (3) the low-dimensionality of the embedding space forces the model to generalize by sharing abstract features across objects, (4) the smoothness of embeddings learned with metric learning facilitates convergence when supervision signals are weak, and (5) occasional true-positive matches (even by chance) yield more coherent gradients than false-positive matches which produce inconsistent gradients and dissipate as noise, leading over time to an acceleration of consistent gradients and stronger initial supervision signal.

## Experiments

**Online Results:** we quantitatively evaluate the online adaptation capabilities of our model through the object identification error of entirely novel objects. In Figure 1 we show that a model observing objects for a few minutes from different angles can self-teach to identify them almost perfectly while the offline supervised approach cannot. OCN is trained on the first 5, 10, 20, 40, 80, and 160 seconds of the 200 seconds video, then evaluated on the identification error of the last 40 seconds of the video for each phase. The supervised offline baseline stays at a 52.4% error, while OCN improves down to 2% error after 80s, a 25x error reduction.

**Robotic Experiments:** here we let a robot collect its own data by looking at a table from multiple angles (Figure 2 and Figure 5).
It then trains itself with OCN on that data, and is asked to point to objects similar to the one presented in front of it. Objects can be similar in terms of shape, color or class. If able to perform that task, the robot has learned to distinguish and recognize these attributes entirely on its own, from scratch and by collecting its own data. We find in Table 5 that the robot is able to perform the pointing task with 72% recognition accuracy of 5 classes, and 89% recognition accuracy of the binary is-container attribute.

**Offline Analysis:** to analyze what our model is able to disentangle, we quantitatively evaluate performance on a large-scale synthetic dataset with 12k object models (e.g. Figure 10), as well as on a real dataset collected by a robot and show that our unsupervised object understanding generalizes to previously unseen objects. In Table 6 we find that our self-supervised model closely follows its supervised equivalent baseline when trained with metric learning. As expected the cross-entropy/softmax supervised baseline approach performs best and establishes the error lower bound while the ResNet50 baseline are upper-bound results.

## Data Collection and Training

We generated three datasets of real and synthetic objects for our experiments. For the real data we arrange objects in table-top configurations and use frames from continuous camera trajectories. The labeled synthetic data is generated from renderings of 3D objects in a similar configuration. Details about the datasets are reported in Table 1.

<div class="figure">
<img src="assets/fig/table4.jpg" style="margin: 0; width: 60%;"/>
<figcaption>
Table 1: Details on our three datasets: head-mounted videos for online training, automatically captured by a robot, and synthetic. </figcaption>
</div>


**Real Data for Online Training:** for the online adaptation experiment, we captured videos of table-top object configurations in the 5 environments (categories): kids room, kitchen, living room, office, and work bench (Figures 1, 4, and 6). We show objects common to each environment (e.g. toys for kids room, tools for work bench) and arrange them randomly; we captured 3 videos for each environment and used 75 unique objects. To allow capturing the objects from multiple view points we use a head-mounted camera and interact with the objects (e.g. turning or flipping them). Additionally, we captured 5 videos of more challenging object configurations (referred to as "challenging") with cluttered objects or where objects are not permanently in view. Finally, we selected 5 videos from the Epic-Kitchens <dt-cite key="Damen2018EPICKITCHENS"></dt-cite> dataset to show that OCN can also operate on even more realistic video sequences. 

From all these videos we take the first 200 seconds and sample the sequence with 15 FPS to extract 3,000 frames. We then use the first 2,400 frames (160s) for training OCN and the remaining 600 frames (40s) for evaluation. We manually select up to 30 reference objects (those we  interacted with) as cropped images for each video in order of their appearance from the beginning of the video (Figure 14). Then we use object detection to find the bounding boxes of these objects in the video sequence and manually correct these boxes (add, delete) in case object detection did not identify an object. This allows us to prevent artifacts of the object detection to interfere with the evaluation of OCN.


<div class="figure">
<img src="assets/fig/online_dataset.jpg" style="margin: 0; width: 100%;"/>
<figcaption>
Figure 4: Six of the environments we used for our self-supervised online experiment. Top: living room, office, kitchen. Bottom: one of our more challenging scenes, and two examples of the Epic-Kitchens.</figcaption>
</div>

**Automatic Real Data Collection:** to explore the possibilities of a system entirely free of human supervision we automated the real world data collection by using a mobile robot equipped with an HD camera. For this dataset we use 187 unique object instances spread across six categories including 'balls', 'bottles & cans', 'bowls', 'cups & mugs', 'glasses', and 'plates'.  Table 2 provides details about the number of objects in each category and how they are split between training, test, and validation. Note that we distinguish between cups & mugs and glasses categories based on whether it has a handle. Figure 5 shows our entire object dataset. 

<div class="figure">
<img src="assets/fig/dataset_v2.jpg" style="margin: 0; width: 100%;"/>
<figcaption>
Figure 5: We use 187 unique object instance in the real world experiments: 110 object for training (left), 43 objects for test (center), and 34 objects for validation (right). The degree of similarity makes it harder to differentiate these objects.</figcaption>
</div>

<div class="figure">
<img src="assets/fig/table5.jpg" style="margin: 0; width: 80%;"/>
<figcaption>
Table 2: Real object dataset: we use 187 unique object instances spread across six categories. </figcaption>
</div>

At each run, we place about 10 objects on the table and then trigger the capturing process by having the robot rotate around the table by 90 degrees. On average 130 images are captured at each run. We select random pairs of frames from each trajectory for training OCN. We performed 345, 109, and 122 runs of data collection for training, test, and validation dataset. In total 43,084 images were captured for OCN training and 15,061 and 16,385 were used for test and validation, respectively.


**Synthetic Data Generation:** to generate diverse object configurations we use 12 categories (airplane, car, chair, cup, bottle, bowl, guitars, keyboard, lamp, monitor, radio, vase) from  ModelNet<dt-cite key="WuSKYZTX15"></dt-cite>. The selected categories cover around 8k models of the 12k models available in the entire dataset. ModelNet provides the object models in a 80-20 split for training and testing. We further split the testing data into models for test and validation, resulting in a 80-10-10 split for training, validation, and test. For validation purposes, we manually assign each model labels describing the semantic and functional properties of the object, including the labels 'class', 'has lid', 'has wheels', 'has buttons', 'has flat surface', 'has legs', 'is container', 'is sittable', 'is device'.

We randomly define the number of objects (up to 20) in a scene. Further, we randomly define the positions of the objects and vary their sizes, both so that they do not intersect. Additionally, each object is assigned one of eight predefined colors. We use this setup to generate 100K scenes for training, and 50K scenes for each, validation and testing. For each scene we generate 10 views and select random combination of two views for detecting objects. In total we produced 400K views (200K pairs) for training and 50K views (25K pairs) for each, validation and testing.  

**Training:** OCN is trained based on two views of the same synthetic or real scene. We randomly pick two frames of a video sequence and detect objects to produce two sets of cropped images. The distance matrix $D_{n,m}$ (Section "Metric Loss for Object Disentanglement") is constructed based on the individually detected objects for each of the two frames. The object detector was not specifically trained on any of our datasets. 


## Experimental Results

We evaluated the effectiveness of OCN embeddings on identifying objects through self-supervised online training, a real robotics pointing tasks, and  large-scale synthetic data. 

**Online Object Identification:** our self-supervised online training scheme enables to train and to evaluate on unseen objects and scenes. This is of utmost importance for robotic agents to ensure adaptability and robustness in real world scenes. To show the potential of our method for these situations we use OCN embeddings to identify instances of objects across multiple views and over time. 

<div class="figure">
<!-- <img src="assets/fig/dataset_v2.jpg" style="margin: 0; width: 100%;"/> -->
<video class="b-lazy" data-src="assets/mp4/kids_room_0_consecutive_1.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<!-- <video class="b-lazy" data-src="assets/mp4/challenging_2_consecutive_1.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video> -->
<video class="b-lazy" data-src="assets/mp4/challenging_1_consecutive_1.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<!-- <video class="b-lazy" data-src="assets/mp4/epic_kitchen_2_consecutive_1.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video> -->
<figcaption>
Figure 6: Comparison of identifying objects with ResNet50 (top) and OCN (bottom) embeddings for the environments kids room (left) and challenging (right). Red bounding boxes indicate a mismatch of the ground truth and associated index.</figcaption>
</div>


We use sequences of videos showing objects in random configurations in different environments (Section "Real Data for Online Training", Figure 4) and train an OCN on the first 5, 10, 20, 40, 80, and 160 seconds of a 200 seconds video. Our dataset provides object bounding boxes and unique identifiers for each object as well as reference objects and their identifiers. The goal of this experiment is to assign the identifier of a reference object to the matching object detected in a video frame. We evaluate the identification error (ground truth index vs. assigned index) of objects present in the last 40 seconds of each video and for each training phase to then compare our results to a ResNet50 (2048-dimensional vectors) baseline. 

We train an OCN for each video individually. Therefore, we only split our dataset into validation and testing data. For the categories kids room, kitchen, living room, office, and work bench we use 2 videos for validation and 1 video for testing; for the categories 'challenging' and epic kitchen we use 3 videos for validation and 2 for testing. We jointly train on the validation videos to find meaningful hyperparameters across the categories and use the same hyperparameters for the test videos. 

Figure 6 shows the same video frames of two scenes from our dataset. Objects with wrongly matched indices are shown with a red bounding box, correctly matched objects are shown with random colors. 
In Figure 7 and Table 3 we report the average error of OCN object identification across our videos compared to the ResNet50 baseline. As the supervised model cannot adapt to unknown objects OCN outperforms this baseline by a large margin. Furthermore, the optimal result among the first 50K training iterations closely follows the overall optimum obtained after 1000K iterations. We report results for 5 categories (kids room, kitchen, living room, office, work bench), that we specifically captured for evaluating OCN and the whole dataset (7 categories). The latter data also shows cluttered objects which are more challenging to detect. 
To evaluate the degree of how object detection is limiting application of OCN we counted the number of manually added bounding boxes of the evaluation sequences. On average the evaluation sequences of the 5 categories have 5,122 boxes (468 added, 9.13%), while the whole dataset (7 categories) has 5,002 boxes on average (1183 added, 25.94%).  

<div class="figure">
<img src="assets/fig/chart_avg_all_v2.svg" style="margin: 0; width: 80%;"/>
<figcaption>
Figure 7: Evaluation of online adaptation: we train an OCN on the first 5, 10, 20, 40, 80, and 160 seconds of each 200 second test video and then evaluate on the remaining 40 seconds. Here we report the lowest average error of all videos (over 1000K iterations) of online adaptation. Results are shown for 5 and 7 categories and compared to the ResNet50 baseline.</figcaption>
</div>

<div class="figure">
<img src="assets/fig/table1.jpg" style="margin: 0; width: 70%;"/>
<figcaption>
Table 3: Evaluation of online adaptation: we report the lowest error among 50K and 1000K iterations of online adaptation in %. [S], [A] = average error for 5 and 7 categories.</figcaption>
</div>


Figure 8 illustrates how objects of one view (anchors) are matched to the objects of another view. We can find the nearest neighbors (positives) in the scene through the OCN embedding space as well as the closest matching objects with descending similarity (negatives). For our synthetic data we report the quality of finding corresponding objects in Table 4 and differentiate between 'attribute errors', that indicate a mismatch of specific attributes (e.g. a blue cup is associated to a red cup), and 'object matching errors', which measure when objects are not of the same instance. An OCN embedding significantly improves detecting object instances across multiple views. 

<div class="figure">
<img src="assets/fig/view_to_view1.jpg" style="margin: 0; width: 100%;"/>
<figcaption>
Figure 8: View-to-view object correspondences: the first column shows all objects detected in one frame (anchors). Each object is associated to the objects found in the other view, objects in the second column are the nearest neighbors. The third column shows the embedding space distance of objects. The other objects are shown from left to right in descending order according to their distances to the anchor (not all objects shown).</figcaption>
</div>


<div class="figure">
<img src="assets/fig/table2.jpg" style="margin: 0; width: 60%;"/>
<figcaption>
Table 4: Object correspondences errors: attribute error indicates a mismatch of an object attribute, while an object matching error is measured when the matched objects are not the same instance.</figcaption>
</div>


**Robot Experiment:** to evaluate OCN for real world robotics scenarios we defined a robotics pointing task. The goal of the task is to enable a robot to point to an object that it deems most similar to the object directly in front of it (Figure 9). The objects on the rear table are randomly selected from the object categories (Table 2). We consider two sets of these target objects. The quantitative experiment in Table 5 uses three query objects per category and is ran three times for each combination of query and target objects (3 $\times$ 2 $\times$ 18 = 108 experiments performed).

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/pointing.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 9: The robot experiment of pointing to the best match of a query object (placed in front of the robot on the small table). </figcaption>
</div>

A quantitative evaluation of OCN performance for this experiment is shown in Table 5. We report on errors related to 'class' and 'container' attributes. While the trained OCN model is performing well on the most categories, it has  difficulty on the object classes 'cups \& mugs' and 'glasses'. These categories are generally mistaken with the category 'bowls'. As a result the network performs much better in the attribute 'container' since all the three categories 'bowls', 'bottles \& cans', and 'glasses' refer to the same attribute.

<div class="figure">
<img src="assets/fig/table7.jpg" style="margin: 0; width: 50%;"/>
<figcaption>
Table 5: Evaluation of robotic pointing: we report on two attribute errors: `class' and `container'. An error for 'class' is reported when the robot points to an object of a different class among the 5 categories: balls, plates, bottles, cups, bowls. An error for 'container' is reported when the robot points to a non-container object when presented with a container object, and vice-versa.</figcaption>
</div>

At the beginning of each experiment the robot captures a snapshot of the scene. We then split the captured image into two images: the upper portion of the image that contains the target object set and the lower portion of the image that only contains the query object. We detect the objects and find the nearest neighbor of the query object in the embedding space to find the closest match. 


**Object Attribute Classification:** one way to evaluate the quality of unsupervised embeddings is to train attribute classifiers on top of the embedding using labeled data. Note however, that this may not entirely reflect the quality of an embedding because it is only measuring a discrete and small number of attributes while an embedding may capture more continuous and larger number of abstract concepts.

**Classifiers:** we consider two types of classifiers to be applied on top of existing embeddings in this experiment: linear and nearest-neighbor classifiers. The linear classifier consists of a single linear layer going from embedding space to the 1-hot encoding of the target label for each attribute. It is trained with a range of learning rates and the best model is retained for each attribute. The nearest-neighbor classifier consists of embedding an entire 'training' set, and for each embedding of the evaluation set, assigning to it the labels of the nearest sample from the training set. Nearest-neighbor classification is not a perfect approach because it does not necessarily measure generalization as linear classification does and results may vary significantly depending on how many nearest neighbors are available. It is also less subject to data imbalances. We still report this metric to get a sense of its performance because in an unsupervised inference context, the models might be used in a nearest-neighbor fashion).

**Baselines:** we compare multiple baselines (BL) in Table 6 and Table 6a. The 'Softmax' baseline refers to the model described in Figure 3, i.e. the exact same architecture as for OCN except that the model is trained with a supervised cross-entropy/softmax loss. The 'ResNet50' baseline refers to using the unmodified outputs of the ResNet50 model <dt-cite key="He2016DeepRL"></dt-cite> (2048-dimensional vectors) as embeddings and training a nearest-neighbor classifier as defined above. We consider 'Softmax' and 'ResNet50' baselines as the lower and upper error-bounds for standard approaches to a classification task. The 'OCN supervised' baseline refers to the exact same OCN training described in Figure 3, except that the positive matches are provided rather than discovered automatically. 'OCN supervised' represents the metric learning upper bound for classification. Finally we indicate as a reference the error rates for random classification.


<div class="figure">
<img src="assets/fig/table3_1.jpg" style="margin: 0; width: 70%;"/>
<figcaption>
Table 6: Attributes classification errors: using attribute labels, we train either a linear or nearest-neighbor classifier on top of existing fixed embeddings. The supervised OCN is trained using labeled positive matches, while the unsupervised one decides on positive matches on its own. All models here are initialized and frozen with ImageNet-pretrained weights for the ResNet50 part of the architecture, while the additional layers above are random and trainable.</figcaption>
</div>

<div class="figure">
<img src="assets/fig/table6.jpg" style="margin: 0; width: 70%;"/>
<figcaption>
Table 6a: we find that models that are not pretrained with ImageNet supervision perform worse but still yield reasonable results. This indicates that the approach does not rely on a good initialization to bootstrap itself without labels. Even more surprisingly, when freezing the weights of the ResNet50 base of the model to its random initialization, results degrade but still remain far below chance as well as below the 'ResNet50 embeddings' baseline. Obtaining reasonable results with random weights has already been observed in prior work such as <dt-cite key="jarrett2009best"></dt-cite>, <dt-cite key="saxe2011random"></dt-cite> and <dt-cite key="DBLP:journals/corr/abs-1711-10925"></dt-cite>.</figcaption>
</div>


**Results:** we quantitatively evaluate our unsupervised models against supervised baselines on the labeled synthetic datasets (train and test) introduced in Section "Synthetic Data Generation". Note that there is no overlap in object instances between the training and the evaluation set.
The first take-away is that unsupervised performance closely follows its supervised baseline when trained with metric learning. As expected the cross-entropy/softmax approach performs best and establishes the error lower bound while the ResNet50 baseline are upper-bound results. 
In Figure 10 we show results of nearest neighbor objects discovered by OCN.

<div class="figure">
<img src="assets/fig/objects_mn12_hv_flat.jpg" style="margin: 0; width: 60%;"/>
<figcaption>
Figure 10: An OCN embedding organizes objects along their visual and semantic features. For example, a red bowl as query object is associated with other similarly colored objects and other containers. The leftmost object (black border) is the query object and its nearest neighbors are listed in descending order. The top row shows renderings of our synthetic dataset, while the bottom row shows real objects. Please note that these are the nearest neighbors among all objects in the respective dataset.</figcaption>
</div>

## Related Work

**Object discovery from visual media:** identifying objects and their attributes has a long history in computer vision and robotics <dt-cite key="Tuytelaars09"></dt-cite>. Traditionally, approaches focus on identifying regions in unlabeled images to locate and identify objects <dt-cite key="1541280,DBLP:conf/cvpr/RussellFESZ06,4270036,4587803,4587502"></dt-cite>. Discovering objects based on the notion of 'objectness' instead of specific categories enables more principled strategies for object recognition <dt-cite key="UijlingsIJCV2013,Romea-2011-7355"></dt-cite>. Several methods address the challenge to discover, track, and segment objects in videos based on supervised <dt-cite key="42961"></dt-cite> or unsupervised <dt-cite key="kwak2015,18cb3eb6fa104c8da9fcaccb96837a5b,Haller_2017_ICCV"></dt-cite> techniques. The spatio-temporal signal present in videos can also help to reveal additional cues that allow to identify objects <dt-cite key="Wang_UnsupICCV2015,DBLP:conf/cvpr/JainXG17"></dt-cite>. In the context of robotics, methods also focus on exploiting depth to discover objects and their properties <dt-cite key="6225107,Karpathy_ICRA2013"></dt-cite>.

Many recent approaches exploit the effectiveness of convolutional deep neural networks to detect objects <dt-cite key="NIPS2015_5638,44872,Lin2017"></dt-cite>, and to even provide pixel-precise segmentations <dt-cite key="he2017maskrcnn"></dt-cite>. While the detection efficiency of these methods is unparalleled, they rely on supervised training procedures and therefore require large amounts of labeled data. Self-supervised methods for the discovery of object attributes mostly focus on learning representations by identifying features in multi-view imagery <dt-cite key="1712-07629,7299135"></dt-cite> and videos <dt-cite key="Wang_UnsupICCV2015"></dt-cite>, or by stabilizing the training signal through domain randomization <dt-cite key="doersch2015unsupervised,zhang2018mixup"></dt-cite>. 

Some methods not only operate on RGB images but also employ additional signals, such as depth<dt-cite key="florence2018,Pot2018SelfsupervisorySF"></dt-cite> or egomotion <dt-cite key="LSM2015"></dt-cite> to self-supervise the learning process. It has been recognized, that contrasting observations from multiple views can provide a view-invariant training signal allowing to even differentiate subtle cues as relevant features that can be leveraged for instance categorization and imitation learning tasks<dt-cite key="Sermanet2017TCN"></dt-cite>.  

**Unsupervised representation learning:** unlike supervised learning techniques, unsupervised methods focus on learning representations directly from data to enable image retrieval <dt-cite key="7410376"></dt-cite>, transfer learning <dt-cite key="zhang2017split"></dt-cite>, image denoising <dt-cite key="Vincent:2008:ECR:1390156.1390294"></dt-cite>, and other tasks <dt-cite key="DumoulinBPLAMC16,KumarCR15"></dt-cite>.
Using data from multiple modalities, such as imagery of multiple views <dt-cite key="Sermanet2017TCN"></dt-cite>, sound <dt-cite key="owens16a,aytar2016soundnet"></dt-cite>, or other sensory inputs <dt-cite key="s17122735"></dt-cite>, along with the often inherent spatio-temporal coherence <dt-cite key="doersch2015unsupervised,DBLP:journals/corr/RadfordMC15"></dt-cite>, can facilitate the unsupervised learning of representations and embeddings. For example, <dt-cite key="Zagoruyko_2015_CVPR"></dt-cite> explore multiple architectures to compare image patches and <dt-cite key="pathakCVPR17learning"></dt-cite> exploit temporal coherence to learn object-centric features. <dt-cite key="DBLP:journals/corr/GaoJG16"></dt-cite> rely of spatial proximity of detected objects to determine attraction in metric learning, OCN operates similarly but does not require spatial proximity for positive matches, it does however take advantage of the likely presence of a same object in any pair of frames within a video. <dt-cite key="DBLP:journals/corr/abs-1710-02139"></dt-cite> also take a similar unsupervised metric learning approach for tracking specific faces, using tracking trajectories and heuristics for matching trajectories and obtain richer positive matches. While our approach is simpler in that it does not require tracking or 3D matching, it could be augmented with extra matching signals.

In robotics and other real-world scenarios where agents are often only able obtain sparse signals from their environment, self-learned embeddings can serve as an efficient representation to optimize learning objectives. <dt-cite key="pathakICMl17curiosity"></dt-cite> introduce a curiosity-driven approach to obtain a reward signal from visual inputs; other methods use similar strategies to enable grasping <dt-cite key="7487517"></dt-cite> and manipulation tasks <dt-cite key="Sermanet2017TCN"></dt-cite>, or to be pose and background agnostic <dt-cite key="Held2015DeepLF"></dt-cite>. <dt-cite key="mitash2017self"></dt-cite> jointly uses 3D synthetic and real data to learn a representation to detect objects and estimate their pose, even for cluttered configurations. <dt-cite key="DBLP:journals/corr/abs-1801-08985"></dt-cite> learn semantic classes of objects in videos by integrating clustering into a convolutional neural network. 

## Conclusion and Future Work

We introduced a self-supervised objective for object representations able to disentangle object attributes, such as color, shape, and function.
We showed this objective can be used in online settings which is particularly useful for robotics to increase robustness and adaptability to unseen objects. We demonstrated a robot is able to discover similarities between objects and pick an object that most resembles one presented to it. In summary, we find that within a single scene with novel objects, the more our model looks at objects, the more it can recognize them and understand their visual attributes, despite never receiving any labels for them.

Current limitations include relying on all objects to be present in all frames of a video. Relaxing this limitation will allow to use the model in unconstrained settings. Additionally, the online training is currently not real-time as we first set out to demonstrate the usefulness of online-learning in non-real-time. Real-time training requires additional engineering that is beyond the scope of this research. Finally, the model currently relies on an off-the-self object detector which might be noisy, an avenue for future research is to back-propagate gradients through the objectness model to improve detection and reduce noise.
