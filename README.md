# NL2TL (EMNLP 2023)
Webpage: https://yongchao98.github.io/MIT-realm-NL2TL/

Demo Website: http://realm-02.mit.edu:8444

Paper Link: https://arxiv.org/pdf/2305.07766.pdf

Dataset Link: https://drive.google.com/drive/folders/10F-qyOhpqEi83o9ZojymqRPUtwzSOcfq?usp=sharing

Model Link: [https://drive.google.com/drive/folders/1vSaKOunMPA3uiOdx6IDbe-gmfREXQ9uO?usp=share_link](https://drive.google.com/drive/folders/1ZfZoYovWoy5z247VXZWZBniNrCOONX4N?usp=share_link)

To access the Demo Website, please send email to ycchen98@mit.edu or yongchaochen@fas.harvard.edu for **password**

This project is to transform human natural languages into Signal temporal logics (STL). Here to enhance the generalizability, in each natural language the specific atomic proposition (AP) is represented as prop_1, prop_2, etc. In this way, the trained model can be easier to transfer into various specific domains. The APs refer to some specific specifications like grad the apple, or go to the room.

Also in the current work, the co-reference is not considered. Therefore, **each prop_i should only appear once in each sentence**. One inference example is as the following:

Input natural language:

```
If ( prop_2 ) happens and continues to happen until at some point during the 176 to 415 time units that ( prop_1 ) , and also if ( prop_3 ) , then the scenario is equivalent to ( prop_4 ) .
```

Output Signal temporal logic:

```
( ( ( prop_2 until [176,415] prop_1 ) and prop_3 ) equal prop_4 )
```

The operations we used are U(until), F(finally), G(globally), |(or), &(and), ->(imply), <->(equal), negation. Also we allow the time interval definition, like U[0,5], F[12,100], and G[30,150]. The time numer right now is constrained into integer, and can use infinite to express all the time in the future, like [5,infinite]. The following are the illustrations. More NL-TL pair examples at https://drive.google.com/file/d/1f-wQ8AKInlTpXTYKwICRC0eZ-JKjAefh/view?usp=sharing
```
prop_1 U[0,5] prop_2 : There exits one time point t between 0 and 5 timesteps from now, that prop_1 continues to happen until at this timestep, and prop_2 happens at this timestep.
```
```
F[12,100] prop_2 : There exits one time point t between 12 and 100 timesteps from now, that prop_2 happens at this timestep.
```
```
G[30,150] prop_2 : For all the time between 30 and 150 timesteps from now, that prop_2 always happens.
```
```
prop_1 -> prop_2 : If prop_1 happens, then prop_2 also happens.
```
```
prop_1 <-> prop_2: prop_1 happens if and only if prop_2 happens.
```

## Description

Signal Temporal Logic (STL) is a formal language for specifying properties of signals. It is used to specify properties of continuous-time signals, such as signals from sensors or control systems, in a way that is precise and easy to understand.

STL has a syntax that is similar to the temporal logic used in computer science, but it is specialized for continuous-time signals. It includes operators for describing the values of a signal, as well as operators for combining and modifying those descriptions.

For example, the STL formula F[0, 2] (x > 0.5) specifies the property that the signal x is greater than 0.5 for all time points between 0 and 2 seconds. This formula can be read as "the signal x is eventually greater than 0.5 for a period of at least 2 seconds".

STL can be used to verify that a signal satisfies a given property, or to synthesize a controller that ensures that a signal satisfies a given property. It is a powerful tool for reasoning about the behavior of continuous-time systems.

While STL is quite powerful, humans are more familiar with defining the tasks via natural languages. Here we try to bridge this gap via fine-tuning large languages models.

## Getting Started

### Dependencies

* The inference model should run on GPU, you can run the notebook file Run.ipynb on Google Colab, or run_trained_model.py on your own GPU environment.
* As for setting the environment, here we install our environmrnt via Minoconda. You can first set up it via https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
* Then it is time to install packages:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pip
conda install python
conda install numpy
conda install pandas
pip install transformers
pip install SentencePiece
```

### Installing

* First download the whole directory with command
```
git clone git@github.com:yongchao98/NL2TL.git
```
* Then download the trained wieghts (e.g., checkpoint-62500) of our model in [https://drive.google.com/file/d/19uiB_2XnnnVmDInaLbQeoZq25ghUdg4D/view](https://drive.google.com/drive/folders/1ZfZoYovWoy5z247VXZWZBniNrCOONX4N?usp=sharing)
* After downloading both the code and model weights, put the model weights checkpoint-62500 into your self-defined directory.

### Other codes and datasets

* As for other codes and datasets published on github, please read the **Illustration of Code and Dataset.pdf** for specific explanation of their utilities.

## Authors

Contributors names and contact info

Yongchao Chen (Harvard University, Massachusetts Institute of Technology, Laboratory of Information and Decision Systems)

yongchaochen@fas.harvard.edu or ycchen98@mit.edu

## Citation for BibTeX

@article{chen2023nl2tl,
  title={NL2TL: Transforming Natural Languages to Temporal Logics using Large Language Models},
  author={Chen, Yongchao and Gandhi, Rujul and Zhang, Yang and Fan, Chuchu},
  journal={arXiv preprint arXiv:2305.07766},
  year={2023}
}
}

## Version History

* 0.1
    * Initial Release on May 12, 2023

## License

This corresponding paper of this project will be attached here in the next months. This project can only be commercially used under our permission.

## Recommended Work

[AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers](https://arxiv.org/pdf/2306.06531.pdf)

[Scalable Multi-Robot Collaboration with Large Language Models: Centralized or Decentralized Systems?](https://yongchao98.github.io/MIT-REALM-Multi-Robot/)


## Transfer Learning with Customized Command-LTL datasets
* As of 2024-11-24, the transfer learning training and testing are implemented on Colab and environment. The “transfer_learning_LTL2.ipynb” is the main file to run this transfer learning.

### Dataset
* The used custom dataset is “command_LTL_dataset_v01.csv”. This file is already located in the directory named “NL2TL/dataset/”.
* Because the format of .csv file is different from the NL2TL’s, internally the realigned .csv file is generated.
* As guided in this code, LTL symbols need to be converted to corresponding words like `F -> finally, & -> and, → -> imply`, etc for better training.

### Data Augmentation
* For data augmentation, the pair of commandt-LTL object (`room-InRoom`, `spray-Spray`, `cup-Cup`, etc.) appeared in the original dataset is augmented with other objects (`living room-InLivingRoom`, `hairspray-HairSpray`, `glass-Glass`, etc.).
* In this way, the number of dataset is increased 15.8x from 2082 to 33080.


### Model
* The used pretrained model is “checkpoint-62500 of t5-base” which is provided by above model [link](https://drive.google.com/drive/folders/1ZfZoYovWoy5z247VXZWZBniNrCOONX4N). Please download and place this model in the directory named “NL2TL/model/t4-base-epch20-infix-word-04-21/”. if not exits, please make the directory.
* Transfer-learned model is uploaded to [t5-base-LTL_koreauniv-epoch20-trainpoint2082](https://drive.google.com/drive/folders/1bVb1VUJbwuVjkqilez1eLPATKbRLAme9?usp=drive_link). The data augmented transfer-learned model is uploaded to [t5-base-LTL_koreauniv-epoch20-trainpoint33080](https://drive.google.com/drive/folders/10pFGzj1fwN6_g9tVxk26WmdkzqOVt9jG?usp=drive_link).
* Please download and place this model in the directory named “NL2TL/model/t5-base-transfer-learning/” and check the filename is correctly applied in the code.


### Notes
* As guided in this code, parenthetis correction for the predicted output is applied which is shown to enhance the accuracy a little higher.
* Multi-step training with train-test partitioning with different ratio is NOT applied because it seems not effective in accuracy performance but also very time-consuming up to finish training.
* As well as top-1 accuracy, which counts the number of perfectly matched LTL, bleu score and precision are added as an evaluation metric. These are popular metrics used for seq2seq tasks (translation).
* The predicted output is compared with multi-labels because comparison with just a single label yields very low accuracy result.

### Results
* The detailed result can be shown in the `result.txt` in each model checkpoint directory.


* Prediction samples
  ```
  input:
  Transform the following sentence into Signal Temporal logic: Navigate to room A, pick the spray and arrange it at site C
  label:
  finally((InRoomA) imply (SprayPickedUp) imply (SprayPlacedInSiteC))
  finally((InRoomA) imply ((SprayPickedUp) imply (SprayPlacedInSiteC)))
  finally((InRoomA) imply ((SprayPickedUp and (SprayPlacedInSiteC))))
  finally(((InRoomA and SprayPickedUp) imply (SprayPlacedInSiteC)))
  finally(((InRoomA imply SprayPickedUp) and (SprayPlacedInSiteC)))
  finally((InRoomA) imply (SprayPickedUp and SprayPlacedInSiteC))
  finally((InRoomA) imply globally(SprayPickedUp) imply globally(SprayPlacedInSiteC))
  finally((InRoomA) imply globally(SprayPickedUp and SprayPlacedInSiteC))
  finally((InRoomA) imply globally((SprayPickedUp) imply (SprayPlacedInSiteC)))
  finally((InRoomA) imply globally((SprayPickedUp and (SprayPlacedInSiteC))))
  pred:
  finally((InRoomA) imply globally(SprayPickedUp) imply globally(SprayPlacedInSiteC))



* Pre-trained model (t5-base-epch20-infix-word-04-21/checkpoint-62500)
  ```
  The test data number = 208
  Top-1 accuracy = 0.0
  Bleu score = 0.0021368774128934388
  Bleu precision = [0.21775697585757095, 0.04541883762940858, 0.0063233932830088265, 0.0006033182503770739] 
  ```
* Transfer-learning model (t5-base-LTL_koreauniv-epoch20-trainpoint2082/checkpoint-2340)
  ```
  The test data number = 208
  Top-1 accuracy = 0.9230769230769231
  Bleu score = 0.971368353166469
  Bleu precision = [0.9935897435897437, 0.9877705627705626, 0.9763507326007327, 0.9629700989518801]
  ```
* Transfer-learning model with data augmentation (t5-base-LTL_koreauniv-epoch20-trainpoint33080/checkpoint-28500)
  ```
  The test data number = 208
  Top-1 accuracy = 0.9855769230769231
  Bleu score = 0.9855769230769231
  Bleu precision = [0.9913461538461539, 0.9903846153846154, 0.9855769230769231, 0.9855769230769231]
  ```

