The code can be found under "code/contrastive approach/" and the appendix with the ablations we did on the alternative methods is "Paper Appendix (Ablations on other methods).pdf".

For the code, "binary_classification_baseline.py" is what we call BCE BERT in the paper, "grubert.py" is GRUBERT, "main.py" is the contrastive approach with MoCo, "main_visualization.py" is what we used to visualize the latent space and "naive_bayes_baseline.py" is what we used for Linear SVC, Logistic Regression, Bernouli NB and Gaussian NB.

# Frequently Asked Questions

### Q: What the fuck does the code do?

The basic model we used is bert-base-uncased, which is a ~110M parameter transformer model that was pretrained for natural language tasks, like the one we have at hand. The input data is tokenized and fed through that model.

binary_classification_baseline.py is the baseline which we used as a comparison. It just uses regular binary cross-entropy as a training objective.

main.py contains the contrastive learning version which was (mostly) inspired by "MoCoV1" ([link](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)) and DyHardCode ([link](https://openreview.net/pdf?id=eiAkrltBTh4)).

As a TLDR: The contrastive loss objective essentially tries to make the encodings of corresponding elements "as close as possible" in some sense. It pulls together corresponding pairs and pushes apart stuff stuff that does not belong to each other. For our problem, the goal is that in the end there will be two clusters in the latent space, one corresponding to positive-sentiment texts and one to negative-sentiment texts.

### Q: How the fuck do I run the code?

Make sure to unzip twitter-datasets before running anything. Navigate to "code/contrastive approach/" (this is important because relative paths were used!), then just do

    python main.py
    
or whatever your favourite way of running python code is.

Requirements to run:

    Python 3.9+
    torch 1.11+
    transformers 4.9.1+
    An internet connection (because the pre-trained model needs to be downloaded first)
    
You can inspect the training progress either in your console or via tensorboard. The logs should be written into the (possibly not yet existing) folder "code/contrastive approach/runs" (maybe you'll have to make that yourself before running so it doesn't crash, idk).
