from utils import attention_unet, unet, segformer, swin_unet, unet3plus, UnetASPP, deeplabv3plus

def main():
    attention_unet.main()
    deeplabv3plus.main()
    unet.main()
    segformer.main()
    swin_unet.main()
    unet3plus.main()
    UnetASPP.main()

if __name__ == '__main__':
    main()