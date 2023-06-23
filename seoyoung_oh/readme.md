# train_deepsup.py
unet 3+  
image resolution: input 512x512 output 512x512  
interpolation: 2048

# train_deepsup2.py
the same except for the purpose(augmentation experiments)

# train_2048_deepsup.py
unet 3+  
image resolution: input 512x512 output 2048x2048  
added convolutional network for interpolation  

# train_all_2048_deepsup.py
unet 3+  
image resolution: input 512x512 output 2048x2048 </br>
added convolutional network for interpolation </br>
the same except for the purpose(group k-fold ensemble)

# train_eff_2048.py
unet 3+  
image resolution: input 1024x1024 output 1024x1024

# soft_voting.py
soft voting ensemble in segmentation

