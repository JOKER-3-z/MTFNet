
# MTFNET
paper link: [MTFNet: Multi-Scale Transformer Framework for Robust Emotion Monitoring in Group Learning Settings](http://www.apsipa2024.org/files/papers/78.pdf)
- Abstrct:  
Identifying students' learning states in authentic classroom settings is a prominent topic in educational technology.
This study addresses the challenges posed by complex facial environments and the scarcity of data in such settings. 
we propose a Multi-Scale Transformer with Frame Shuffled Order Predict Network (MTFNet), based on a spatial-temporal feature
extraction structure, to perform effective learning-related facial expression recognition in a primary school classroom. Specifically,
we combine a Multi-Scale Facial Feature Fusion Module (MFFF) based on Grouped Spatial Convolution(GS Conv) to effectively
capture multi-level facial features and improve the model's robustness in complex environments. Additionally, a Frame-wise 
Shuffle Order Prediction Module (FSOP) is introduced to enhance the model's ability to understand the dynamic changes of
emotional intensity by predicting the emotion expression sequence. Experiments on both the DFEW dataset and our dataset
demonstrate excellent performance and generalization in realworld applications.

- Experimental show that:  
On the [DFEW](https://dfew-dataset.github.io/) dataset: 53.55%(UAR) & 67.59%(WAR)  
On the Group Learning Setting Dataset : 67.85%(UAR) & 64.65%(WAR)

## Dependencies

Please make sure Python>=3.8

Required packages are listed in requirements.txt. You can install them by running:

```
pip install -r requirements.txt
```
## explain the name of model weights
```
seed_(random seed)/epochs_(number of epoch in best acc)_（best acc on evaluate set）_(model name)
```
## model trainning
```
run_singleout.py 
```

## Testing
set the parameter in the config.json file
```
predict.py  
```

