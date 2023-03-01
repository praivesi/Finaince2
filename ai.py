from fastai.vision.all import *

print('a')
# 데이터 준비
path = untar_data(URLs.MNIST_SAMPLE)
print(path)
print('b')
dls = ImageDataLoaders.from_folder(path)

print('c')
# 모델 생성
learn = cnn_learner(dls, resnet18, pretrained=False, metrics=accuracy)

print('d')
# 모델 학습
learn.fit_one_cycle(1)

print('e')
# 모델 예측
x_test = dls.test_dl([path/'train'/'3'/'12.png'])
print('f')
y_pred = learn.get_preds(dl=x_test)
print('g')

print(y_pred)
