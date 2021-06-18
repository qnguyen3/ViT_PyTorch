# ViT - Vision Transformer

This is an implementation of ViT - Vision Transformer by Google Research Team through the paper [**"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**](https://arxiv.org/abs/2010.11929)


## ViT Architecture
![Architecture of Vision Transformer](https://neurohive.io/wp-content/uploads/2020/10/rsz_cov.png)

## Configs
You can config the network by yourself through the `config.txt` file

```
64 \t#batch_size
50 #epoch
0.001 #learning_rate
0.7 #gamma
256 #img_size
16 	#patch_size
100	#num_class
768	#d_model
12	#n_head
12  #n_layers
3072  #d_mlp
3		  #channels
0.	  #dropout
cls		#pool
```

## Training
Currently, you can only train this model on CIFAR-100 with the following commands:

`> pip3 install einops`\
`> git clone https://github.com/quanmario0311/ViT_PyTorch.git`\
`> cd ViT_PyTorch`\
`> python3 train.py`
