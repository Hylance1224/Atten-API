## 1. Atten-API

### 1.1 Usage

1.  **Generating training data and testing data**

    The data used for experiment is stored in sub folder of **"data"**. After executing **load\_data.py**, two sub-folders named **"training data"** and **"testing data"** will be created. These two folders respectively store five-fold data for training and testing, respectively.

2.  **Model training**

    Simply start Python in COMMAND LINE mode, then use the following statement (one line in the COMMAND Prompt window) to execute **train\_model.py**:

        python train_model.py --train_dataset training_0.json --weight_decay 0.0001 --lr 0.01 
        --continue_training 0 --train_batch_size 1024 --epoch 2

    Once the program execution is complete, it will generate a folder named **"model\_Atten\_TPL"**, where the trained model is stored.

3.  **Model testing**

    Start Python in COMMAND LINE mode, then use the following statement (one line in the COMMAND Prompt window) to execute **test\_model.py**:

        python test_model.py --test_dataset testing_0_3.json

    Once the program execution is complete, it will generate a folder named "output", where the recommendation results is stored.

4.  **Metrics evaluation**

    Start Python in COMMAND LINE mode, then use the following statement (one line in the COMMAND Prompt window) to execute **metrics.py**:

        python metrics.py --rm 3

    Then, you may receive the results as follows:

        MP_rm: 0.665159, MR_rm: 0.665159, MF_rm: 0.665159, MAP_rm: 0.887698, COV_rm: 0.585845
        MP_2rm: 0.391802, MR_2rm: 0.783604, MF_2rm: 0.522403, MAP_2rm: 0.852650, COV_2rm: 0.639581

### 1.2 Environment Settup

Our code has been tested under Python 3.12.3. The experiment was conducted via PyTorch, and thus the following packages are required:

    torch == 1.3.1
    numpy == 1.18.1
    scipy == 1.3.2
    sklearn == 0.21.3

Updated version of each package is acceptable.&#x20;

### 1.3 Description of Essential Folders and Files

| Name                 | Type   | Description                                                                                                                                       |
| -------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| original dataset     | Folder | Data files required for the experiment. Specifically: **relation.json** contains the mashups used in the experiment and their corresponding APIs. |
| train\_model.py      | File   | Model training python file of Atten-API                                                                                                           |
| test\_model.py       | File   | Model testing python file of Atten-API                                                                                                            |
| model\_Atten\_API.py | File   | Model modules of Atten-API                                                                                                                        |
| utility              | Folder | Tools and essential libraries used by Atten-API                                                                                                   |


