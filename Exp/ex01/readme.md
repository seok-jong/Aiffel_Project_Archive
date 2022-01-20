# Rock Scissor Paper Classification 
<br/><br/>

> 여러명의 데이터셋을 모으는 것이 번거로워 손쉽게 취합할 수 있도록 filename_editor.py를 만듦.  
이를 통해 파일 이름 중복문제 없이 모든 이미지들을 분류하여 취합 가능   
사용법은 `filename_editor.ipynb` 참고

<br/><br/>
## 1. **Import Library**
프로젝트 수행에 필요한 라이브러리들을 불러온다. 

- tensorflow
- keras 
- numpy 
- matplotlib 
- os
- PIL
- glob
- sklearn 

#

## 2. **Preprocess**

모델에 학습시키기위해 수집한 데이터셋(가위,바위,보 이미지)를 전처리한다.       


### 1) **Resize**


```python
def resize_images(img_path):
	images=glob.glob(img_path + "/*.jpg")  
    
	print(len(images), " images to be resized.")

    #resize to 28 x 28
	target_size=(28,28)
	for img in images:
		old_img=Image.open(img)
		new_img=old_img.resize(target_size,Image.ANTIALIAS)
		new_img.save(img, "JPEG")
    
	print(len(images), " images resized.")
```
#
## 3. Load Dataset


전처리한 이미지를 모델에 학습시키기 위한 형태로 load한다. 
```python
def resize_images(img_path):
	images=glob.glob(img_path + "/*.jpg")  
    
	print(len(images), " images to be resized.")

    #resize to 28 x 28
	target_size=(28,28)
	for img in images:
		old_img=Image.open(img)
		new_img=old_img.resize(target_size,Image.ANTIALIAS)
		new_img.save(img, "JPEG")
    
	print(len(images), " images resized.")
	
```

이미지가 로드되면 inference를 위한 데이터셋을 마련하기 위해 위의 학습데이터를 split한다. 
```python
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=777)
```
참고 : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


## 4. **Build Model**

```python
n_channel_1=16
n_channel_2=32
n_dense=32
n_train_epoch=20

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

```
모델을 설계할 때, keras의 models 메소드를 불러와 사용을 하고   Sequential API을 사용하여 설계한다.    

**Sequential API**을 사용하는 이유는 Task자체가 비교적 간단하고 데이터셋의 양이 적기 때문에 모델의 구조가 복잡할 필요가 없다.   
이처럼 단순한 모델을 설계하고자 할 때, Sequential API를 사용하면 간단하게 모델을 설계할 수 있다.    
<br/>

Sequential API를 통해 모델을 설계하는 방법은 아래 그림을 통해 대략적으로 이해할 수 있다. 
<br/>
<br/>

![model](https://d3s0tskafalll9.cloudfront.net/media/images/F-1-5.max-800x600.png)

위 그림을 참고하였을 때, 이 프로젝트에서 사용하는 데이터는 Color 이미지(3차원)이기 때문에 `input_shape`만 `(28,28,3)으로 수정해 준다. 
<br/><br/><br/><br/>
#

## 5. **Train and Inference**

`fit`을 이용하여 학습을 시키고 테스트까지 진행한 후, 결과를 출력해 본다. 

```python

# 모델 훈련
model.fit(x_train, y_train, epochs=n_train_epoch)

# 모델 시험
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
```



```33/33 [==============================] - 2s 3ms/step - loss: 5.8771 - accuracy: 0.3139
Epoch 2/20
33/33 [==============================] - 0s 2ms/step - loss: 1.2259 - accuracy: 0.4852
Epoch 3/20
33/33 [==============================] - 0s 2ms/step - loss: 0.8414 - accuracy: 0.6370
Epoch 4/20
33/33 [==============================] - 0s 2ms/step - loss: 0.5942 - accuracy: 0.7350
Epoch 5/20
33/33 [==============================] - 0s 2ms/step - loss: 0.5139 - accuracy: 0.8057
Epoch 6/20
33/33 [==============================] - 0s 2ms/step - loss: 0.4331 - accuracy: 0.8287
Epoch 7/20
33/33 [==============================] - 0s 2ms/step - loss: 0.4106 - accuracy: 0.8261
Epoch 8/20
33/33 [==============================] - 0s 2ms/step - loss: 0.3103 - accuracy: 0.8950
Epoch 9/20
33/33 [==============================] - 0s 2ms/step - loss: 0.2885 - accuracy: 0.8912
Epoch 10/20
33/33 [==============================] - 0s 2ms/step - loss: 0.2547 - accuracy: 0.9149
Epoch 11/20
33/33 [==============================] - 0s 2ms/step - loss: 0.2189 - accuracy: 0.9224
Epoch 12/20
33/33 [==============================] - 0s 2ms/step - loss: 0.2046 - accuracy: 0.9366
Epoch 13/20
33/33 [==============================] - 0s 2ms/step - loss: 0.2179 - accuracy: 0.9217
Epoch 20/20
33/33 [==============================] - 0s 2ms/step - loss: 0.1220 - accuracy: 0.9586
15/15 - 0s - loss: 0.1845 - accuracy: 0.9378
test_loss: 0.1845485121011734 
test_accuracy: 0.9377777576446533
```

데이터셋의 크기가 굉장히 작음에도 불구하고 높은 정확도를 도출해 낼 수 있었다. 

높은 accuracy가 나온 이유  
1) 적은 test data
2) 데이터를 확인해 보니 5명의 이미지 모두 비슷한 배경에 비슷한 위치에 손이 위치해 있음 - 모든 데이터에 대해 robust하지 못함. 
3) 비슷한 데이터에 대한 overfitting(?)
4) train data를 255로 나누어 Nomalization을 해주었는데 test data에도 똑같이 적용이 되었음



<br/>
<br/>
<br/><br/>

# reference

MNIST 데이터셋 공식 문서 
https://keras.io/api/datasets/mnist/#load_data-function

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
https://blog.naver.com/PostView.nhn?blogId=hankrah&logNo=221832296707&parentCategoryNo=&categoryNo=64&viewDate=&isShowPopularPosts=true&from=search
https://www.delftstack.com/ko/howto/python/rename-a-file-in-python/
https://www.delftstack.com/ko/howto/python/python-file-move/
https://wikidocs.net/80
https://code.tutsplus.com/ko/tutorials/compressing-and-extracting-files-in-python--cms-26816
https://ahnjg.tistory.com/88
https://frhyme.github.io/python-lib/matplotlib_extracting_color_from_cmap/