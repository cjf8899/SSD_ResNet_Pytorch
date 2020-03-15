




class Config:

	
    #voc root
    VOC_ROOT = '/SSD_ResNet_Pytorch'

    #class + 1
    num_classes = 21

    
    #learning rate
    lr = 0.001
    #ssd paper = 32
    batch_size = 32 
    
    momentum = 0.9
    weight_decay = 0.0005

    # 40k + 10k = 116 epock
    epoch = 116 


    #pre-train VGG root
    #The resnet pre-train model is in lib.res-model...
    save_folder = './weights/'
    basenet = 'vgg16_reducedfc.pth'

    log_fn = 10 

    neg_radio = 3
    

    
    #input-image size
    min_size = 300
    
    #boxe out image size
    grids = (38, 19, 10, 5, 3, 1)

    
    #boxe num
    anchor_num = [4, 6, 6, 6, 4, 4]
    
    #255 * R, G, B
    mean = (104, 117, 123)
    
    
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))

    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]

    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)] 
   

    
    
    variance = (0.1, 0.2)



opt = Config()
