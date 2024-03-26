# Computationnal Imaging course : Multimodal Style Transfer via Graph Cuts
Pytorch(1.0+) implementation(unofficial) of ICCV 2019 paper ["Multimodal Style Transfer via Graph Cuts"](https://arxiv.org/abs/1904.04443).

Original tensorflow implementations from the author can be found [here](https://github.com/yulunzhang/MST).

This repository provides a pre-trained model for you to generate your own image given content image and style image. Also, you can download the training dataset or prepare your own dataset to train the model from scratch.

## Requirements

- Python 3.6+
- PyTorch 1.0+
- PyMaxflow
- Pillow
- TorchVision

Optional but recommended for training
- GPU environment 

## Installation : 

In order to install the necessary libraries run the following lines in your terminal : 

```[bash]
conda create -n MST python=3.9
conda activate MST
pip install -r requirements.txt
```
create a folder named output, the generated images will be saved in this folder.  

## Contribution 

From the original fork, we have added 
- a requirement.txt
- a bash script to run a large number of experiments
- the possibility to visually display in 3D space the style features
- an automatic way to choose the number of clusters (Elbow curve, Silouhette)
- a new criterium to find the optimal number of clusters using MSE between the output and the style using the style as content (see notebook)
- a dBscan clustering algorithm instead of kmeans 

## Run the code

```[bash]
sh script.sh
```
In the script the preprocessing of the data is called, it will automatically check within the folder to see if the data needs to be preprocessed to run style transfer, if it needs to do so the modified dataset code will do it. Then the style transfer can be done, the number of clusters varies in a loop but can be fixed to a particular number, there are also two arguments for choosing do display are not the cluster using tsne (`--print_tsne "True" \ --print_cluster_criterium "True" `)

# Previous work before the fork :

## I have provided a jupyter notebook with instructions to run the training and testing python files using the Google Colab's free GPU. 
If you wish to run them on your local machine you can follow the instructions below.
Note: Testing does not require GPU.

## train

1. Download [COCO](http://cocodataset.org/#download) for content dataset and [Wikiart](https://www.kaggle.com/c/painter-by-numbers) for style dataset and unzip them, rename them as `content` and `style` respectively (recommended).

2. Modify the argument in the` train.py` such as the path of directory, epoch, learning_rate or you can add your own training code.

3. Train the model using gpu.

4. ```python
   python train.py
   ```

   ```
    usage: train.py [-h] [--batch_size BATCH_SIZE] [--epoch EPOCH] [--gpu GPU]
                    [--learning_rate LEARNING_RATE]
                    [--snapshot_interval SNAPSHOT_INTERVAL]
                    [--n_cluster N_CLUSTER] [--alpha ALPHA] [--lam LAM]
                    [--max_cycles MAX_CYCLES] [--gamma GAMMA]
                    [--train_content_dir TRAIN_CONTENT_DIR]
                    [--train_style_dir TRAIN_STYLE_DIR]
                    [--test_content_dir TEST_CONTENT_DIR]
                    [--test_style_dir TEST_STYLE_DIR] 
                    [--save_dir SAVE_DIR] [--reuse REUSE]
   ```

## test

1. Clone this repository 

   ```bash
   git clone https://github.com/Rakshit-Shetty/Multimodal-Style-Transfer-via-Graph-Cuts.git
   cd Multimodal-Style-Transfer-via-Graph-Cuts
   ```

2. Prepare your content image and style image. I provide some in the `content` and `style` and you can try to use them easily.

3. Download the pretrained model [here](https://drive.google.com/)

4. Generate the output image. 3 outputs in total. A transferred output image with and without style image and a nested_demo_like image like those before and after image will be generated.

   ```python
   python test.py -c content_image_path -s style_image_path
   ```

   ```
    usage: test.py [-h] [--content CONTENT] [--style STYLE]
                [--output_name OUTPUT_NAME] [--n_cluster N_CLUSTER]
                [--alpha ALPHA] [--lam LAM] [--max_cycles MAX_CYCLES]
                [--gpu GPU] [--model_state_path MODEL_STATE_PATH]
   ```

   If output_name is not given, it will use the combination of content image name and style image name.


------


# Result

Some results of content image will be shown here.

![image](https://github.com)
