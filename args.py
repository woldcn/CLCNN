class ArgsCLCNN:
    model_name = 'CLCNN'
    checkpoint = 'pth/vgg16-397923af.pth'
    device = 'cuda'
    batch_size = 10
    epochs = 200
    lr = 0.0001
    wd = 0.01
    rand_seed = 42
    T = 0.8
    useCL = True
    loss_name = 'ContrastiveLoss'
    train_path = 'aug/train'
    test_path = 'aug/test'
    log_path = 'result/CLCNN/log/log.txt'
    saved_model = 'result/CLCNN/saved_model/CLCNN.pth'

class ArgsVgg16:
    model_name = 'Vgg16'
    checkpoint = 'pth/vgg16-397923af.pth'
    device = 'cuda'
    batch_size = 10
    epochs = 200
    lr = 0.0001
    wd = 0.01
    rand_seed = 42
    loss_name = 'CrossEntropyLoss'
    train_path = 'aug/train'
    test_path = 'aug/test'
    log_path = 'result/vgg16/log/log.txt'
    saved_model = 'result/vgg16/saved_model/vgg16.pth'

ArgsVgg16 = ArgsVgg16()
ArgsCLCNN = ArgsCLCNN()

def getArgs(modelName):
    if modelName == 'CLCNN':
        return ArgsCLCNN
    if modelName == 'Vgg16':
        return ArgsVgg16
