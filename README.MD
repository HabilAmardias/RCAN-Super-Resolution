## Image Super Resolution (On Progress)

This project implements a deep learning model based on RCAN (Residual Channel Attention Network)[1], for image super-resolution. RCAN model enhances low-resolution images by learning to recover high-frequency details, resulting in sharper and clearer images. The architecture leverages a channel attention mechanism to improve feature representation, enabling the model to perform well on various types of image data. Model is trained on [image dataset][kaggle] from Kaggle  using PyTorch with batch size 40, 2 steps Gradient Accumulation and Mixed Precision Training.

1. Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., & Fu, Y. (2018). Image super-resolution using very deep residual channel attention networks. In Proceedings of the European conference on computer vision (ECCV) (pp. 286-301).

[kaggle]:https://www.kaggle.com/datasets/quadeer15sh/image-super-resolution-from-unsplash