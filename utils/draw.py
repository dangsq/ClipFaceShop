from PIL import Image

import matplotlib.pyplot as plt

def draw(imagefolder,tail,title,mode):
    
    plt.figure(figsize=(20,25),dpi=40)
    plt.title(imagefolder+title)
    plt.subplot(5,4,2)
    plt.title('base   '+mode)
    plt.imshow(Image.open(imagefolder+'base.png'))
    plt.axis('off')
    plt.subplot(5,4,3)
    plt.title('ref   '+mode)
    plt.imshow(Image.open(imagefolder+'target.png'))
    plt.axis('off')
    if mode!='f_At':
        plt.subplot(5,4,4)
        plt.title('b')
        plt.imshow(Image.open(imagefolder+'b0.png'))
        plt.axis('off')
    if mode=='f_At' or mode=='f_At+t+b' or mode=='f_At+b':
        plt.subplot(5,4,1)
        plt.title('A')
        plt.imshow(Image.open(imagefolder+'A0.png'))
        plt.axis('off')
    plt.subplot(5,4,5)
    plt.title('t')
    plt.imshow(Image.open(imagefolder+'img_gen_0.png'))
    plt.axis('off')
    plt.subplot(5,4,6)
    plt.title('t')
    plt.imshow(Image.open(imagefolder+'img_gen_1.png'))
    plt.axis('off')
    plt.subplot(5,4,7)
    plt.title('t')
    plt.imshow(Image.open(imagefolder+'img_gen_2.png'))
    plt.axis('off')
    plt.subplot(5,4,8)
    plt.title('t')
    plt.imshow(Image.open(imagefolder+'img_gen_3.png'))
    plt.axis('off')
    plt.subplot(5,4,9)
    if mode=='f_At+b':
        At_b='At+b'
        At='At'
    elif mode=='f_At':
        At_b='At'
    elif mode=='f_At+t+b':
        At_b='At+t+b'
        At='At+t'
    else:
        At_b='t+b'
        
    plt.title(At_b)
    plt.imshow(Image.open(imagefolder+'img_gen_amp_0_0.png'))
    plt.axis('off')
    plt.subplot(5,4,10)
    plt.title(At_b)
    plt.imshow(Image.open(imagefolder+'img_gen_amp_0_1.png'))
    plt.axis('off')
    plt.subplot(5,4,11)
    plt.title(At_b)
    plt.imshow(Image.open(imagefolder+'img_gen_amp_0_2.png'))
    plt.axis('off')
    plt.subplot(5,4,12)
    plt.title(At_b)
    plt.imshow(Image.open(imagefolder+'img_gen_amp_0_3.png'))
    plt.axis('off')
    if mode=='f_At+t+b' or mode=='f_At+b':
        plt.subplot(5,4,13)
        plt.title('t+b')
        plt.imshow(Image.open(imagefolder+'img_gen_amp_b0_0.png'))
        plt.axis('off')
        plt.subplot(5,4,14)
        plt.title('t+b')
        plt.imshow(Image.open(imagefolder+'img_gen_amp_b0_1.png'))
        plt.axis('off')
        plt.subplot(5,4,15)
        plt.title('t+b')
        plt.imshow(Image.open(imagefolder+'img_gen_amp_b0_2.png'))
        plt.axis('off')
        plt.subplot(5,4,16)
        plt.title('t+b')
        plt.imshow(Image.open(imagefolder+'img_gen_amp_b0_3.png'))
        plt.axis('off')

        plt.subplot(5,4,17)
        plt.title(At)
        plt.imshow(Image.open(imagefolder+'img_gen_amp_aw0_0.png'))
        plt.axis('off')
        plt.subplot(5,4,18)
        plt.title(At)
        plt.imshow(Image.open(imagefolder+'img_gen_amp_aw0_1.png'))
        plt.axis('off')
        plt.subplot(5,4,19)
        plt.title(At)
        plt.imshow(Image.open(imagefolder+'img_gen_amp_aw0_2.png'))
        plt.axis('off')
        plt.subplot(5,4,20)
        plt.title(At)
        plt.imshow(Image.open(imagefolder+'img_gen_amp_aw0_3.png'))
        plt.axis('off')

    
    plt.savefig(imagefolder+''+tail+'.jpg')
    plt.show()