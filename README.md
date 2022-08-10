# BRAIN TUMOR DETECTION

This repository contains an attempt at Brain Tumor detection by using image classification. The dataset used can be found <a href="https://www.kaggle.com/code/emrearslan123/brain-tumor-detection-on-mri-images/data">here</a>. A good introduction to this project is in the *tumor_detection.ipynb* notebook. It will walk you through each step of the training process, from data importation to manual testing of the resulting model.

# Run the project

### **_Requirements_**

-   #### _With **GPU** installed_
    -   If you have a **GPU** installed and whish to use it during training:
        -   open *requirements.txt*,
        -   replace <code>tensorflow-cpu==x.x.x</code> with <code>tensorflow==x.x.x</code>.
-   #### _Installation_
    -   To install requirements run: <br />
        -   <code>pip install -r requirements.txt</code>
-   #### _Image Folder Tree_
    -   I recommend spliting your test/train dataset following a 20/80 repartition depending on the number of images. 
    -   To split the data you can use the <code>train_test_split.py</code> script before manually classifying your data, it will move your images in a test and train folder. To run it use: <code>python train_test_split.py</code> you can add <code>-p</code> to define the percentage to be used for training and <code>-i</code> to define the folder in which the non classified images are, exemple:
        -   <code>python utils/train_test_split.py -p 90 -i brain_tumor_dataset</code> for a 90/10 split on the *brain_tumor_dataset/* folder.
    -   **Binary** Classification:<br />
        ```
        defects
        ├── test
        │   ├── CLEAN
        │   │   └── *.jpg
        │   └── DEFECT
        │       └── *.jpg
        ├── train
        │   ├── CLEAN
        │   │   └── *.jpg
        │   └── DEFECT
        │       └── *.jpg
        └── validation
            ├── CLEAN
            │   └── *.jpg
            └── DEFECT
                └── *.jpg
        ```

### **_Train a model_**

-   Uses<br />
    -   The *train.py* script is by default meant to **binary** classify images of brian tumors,
    -   If you modify the **layers** of the model you can use it to classify any other type of images,
    -   By changing the **activation function** of the output layer, and tuning the other layers for precision, you can use it to **categoricaly** classify images.
-   Arguments:
    -   <code>-i --input_path</code>. Path to Traning and Testing images folder.
        -   Type: <code>string</code>,
        -   Default: <code>./defects/</code>.
    -   <code>-s --img_size</code>. Size of the images (note that when resizing width = height).
        -   Type: <code>integer</code>,
    -   <code>-e --epochs</code>. Number of training epochs.
        -   Type: <code>integer</code>,
        -   Default: <code>40</code>.
    -   <code>-w --workers</code>. Number of workers for training.
        -   Type: <code>integer</code>,
        -   Default: <code>32</code>.
    -   <code>-b --batch_size</code>. Batch Size for training, allways make sure it is greater or equal to the number of images in each folder.
        -   Type: <code>integer</code>,
        -   Default: <code>32</code>.

-   To train a model run: <br />
    -   <code>python train.py</code>

### **_Predict_**

-   Uses
    -   The *test.py* script can be used with any previsously trained models if:
        -   The model has been trained to classify **images**,
        -   Use the proper **arguments** as listed bellow.
-   Arguments:

    -   <code>-i --input_path</code>. Path to folder of image to classify.
        -   Type: <code>string</code>.
        -   Default: <code>input/</code>.
    -   <code>-m --model</code>. Name of the model to use for classification.
        -   Type <code>string</code>,
        -   Default: <code>15_a76_f80_1_REFERENCE</code>.
    -   <code>-n --number</code>. Number of images to classify.
        -   Type: <code>integer</code>,
        -   Default: <code>10</code>.

### **_Contact_**

-   This repository was created and maintained by <a href="https://github.com/LuluLaGlue">Lucien SIGAYRET</a> feel free to contact me.