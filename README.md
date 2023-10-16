# [ClipFaceShop](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Bring_Clipart_to_Life_ICCV_2023_paper.pdf)

<img src="./figures/fig1.png" >

## Abstract

The development of face editing has been boosted since the birth of StyleGAN. Many works have been proposed to help users express their ideas through different interactive methods, such as sketching and exemplar photos. However, they are limited either in expressiveness or generality. We propose a new interaction method by guiding the editing with abstract clipart, composed of a set of simple semantic parts, allowing users to control across face photos with simple clicks. However, this is a challenging task given the large domain gap between colorful face photos and abstract clipart with limited data. To solve this problem, we introduce a framework called ClipFaceShop built on top of StyleGAN. The key idea is to take advantage of $\mathcal{W}+$ latent code encoded rich and disentangled visual features, and create a new lightweight selective feature adaptor to predict a modifiable path toward the target output photo. Since no pairwise labeled data exists for training, we design a set of losses to provide supervision signals for learning the modifiable path. Experimental results show that ClipFaceShop generates realistic and faithful face photos, sharing the same facial attributes as the reference clipart. We demonstrate that ClipFaceShop supports clipart in diverse styles, even in form of a free-hand sketch.

---------------------

Code is under collated and will be released in next few months.

