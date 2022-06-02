Code is in code/contrastive approach/

# Frequently Asked Questions

### Q: What the fuck does the code do?

The basic model is bert-base-uncased, which is a ~100M parameter transformer model that was pretrained for natural language tasks, like the one we have at hand. The input data is tokenized and fed through that model.

binary_classification_baseline.py is the baseline which we will use as a comparison. It just uses regular binary cross-entropy as a training objective.

main.py contains the contrastive learning version which was (mostly) inspired by "MoCoV1" ([link](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)) and DyHardCode ([link](https://openreview.net/pdf?id=eiAkrltBTh4)). For a soft introduction to contrastive learning and momentum encoders, you could skim through the first 2-3 chapters of my bachelor's thesis [here](https://pub.tik.ee.ethz.ch/students/2021-HS/BA-2021-25.pdf), where I used a similar framework, but with a "Natural Language to Programming Language" problem. 

As a TLDR: The contrastive loss objective essentially tries to make the encodings of corresponding elements "as close as possible" in some sense. For our problem, the goal is that in the end there will be two clusters in the latent space, one corresponding to positive-sentiment texts and one to negative-sentiment texts.

### Q: How the fuck do I run the code?

Make sure to unzip twitter-datasets before running anything. Then just do

    python main.py
    
Or whatever your favourite way of running python code is.

Requirements to run:

    Python 3.9+
    torch 1.11+
    transformers 4.9.1+
    An internet connection (because the pre-trained model needs to be downloaded first)
    Some time to deal with errors
    
In particular, the second-to-last point is a problem with the Euler cluster, because by default your programs aren't allowed to access the internet outside of the ETH intranet. Somewhere on the Euler website there's a simple command that activates a proxy (whatever that means, but it makes the internet work I guess). Also, make sure to figure out how to enable Python 3.9+ and to update PyTorch to the appropriate version. I noticed that with the default version the program sometimes crashes because they recently something about the loss function. No idea how to update that stuff, I somehow just barely got it to run. If you need any help I may be able to help you, but don't count on it. I have no idea what I'm doing.
