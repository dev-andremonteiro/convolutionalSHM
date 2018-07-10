################### Bibliotecas Utilizadas ##########################

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

####################################################################

#################### PARÂMETROS INICIAIS ###########################

PASTA_DE_DADOS = 'data'
SENSOR_NUMERO = 'S1'
REDUZIR_IMAGEM_PARA = 128           #Sempre o mesmo num para altura e largura
NUMERO_EPOCAS = 3

####################################################################


#################### CARREGANDO OS DADOS ###########################

data_path = os.getcwd() + '/' + PASTA_DE_DADOS + '/' + SENSOR_NUMERO
data_dir_list = os.listdir(data_path)
num_channel=1
linhas_img = REDUZIR_IMAGEM_PARA
colunas_img = REDUZIR_IMAGEM_PARA

lista_imgs=[]                                                                   #Lista que conterá todos os vetores caracteristicas de cada uma das imagens

def image_to_feature_vector(image, size=(linhas_img, colunas_img)):
    # Reduz a imagem para o tamanho especificado, depois achata a imagem
    # em uma lista de intensidade de pixels.
    return cv2.resize(image, size).flatten()

for classe in data_dir_list:
    classe_path = data_path+'/'+ classe
    img_list=os.listdir(classe_path)
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ classe + '/'+ img )               #Lê uma imagem com o caminho especificado.
        input_img = input_img[49:585, 114:792]                                  #Corta a borda branca.
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)                   #Conversão da imagem RGB em preto e branco.

        #cv2.imwrite('teste.jpg',input_img)

        input_img_flatten=image_to_feature_vector(input_img,(linhas_img,colunas_img))   #Conversão da imagem em PB para vetor caracteristica. (Resize e Flatten)
        lista_imgs.append(input_img_flatten)                                            #Adicionando a lista principal para guardar os vetores de característica.

####################################################################
#
#   A partir deste ponto temos as imagens em forma de vetor de caracteristica na
# variavel "lista_imgs". Todas imagens que estão nas subpastas do sensor escolhido
# já estão carregadas no programa.



#################### PRÉ PROCESSAMENTO ###########################

from sklearn.preprocessing import scale

np_lista_imgs = np.array(lista_imgs)                                            #Transformando a lista de imagens já vetorizadas para um array da biblioteca numpy.
np_lista_imgs = np_lista_imgs.astype('float32')                                 #Conversão em 'float32'
imgs_padronizadas = scale(np_lista_imgs)                                        #Padronização do conjunto de dados, centraliza os números para uma escala onde a média é 0.
imgs_padronizadas= imgs_padronizadas.reshape(np_lista_imgs.shape[0],num_channel,linhas_img,colunas_img)    #Reformatando a imagem para ficar no padrão da entrada.
np_lista_imgs = imgs_padronizadas

####################################################################



#################### CRIANDO AS CLASSES ###########################

num_classes = 4

num_amostras = np_lista_imgs.shape[0]
labels = np.ones((num_amostras,),dtype='int64')

labels[0:60]=0
labels[60:120]=1
labels[120:180]=2
labels[180:]=3

names = ['d1','d2','d3','integro']

Y = np_utils.to_categorical(labels, num_classes)                                #Hot-Encoding

####################################################################


################ DIFININDO O MODELO DA CNN #########################

input_shape=np_lista_imgs[0].shape

model = Sequential()

# 1 Camada #
model.add(Convolution2D(filters = 32,
                        kernel_size = (3,3),
                        padding = 'same',
                        input_shape = input_shape
                        #,activation='relu'
                        ))
model.add(Activation('relu'))
##########

# 2 Camada #
model.add(Convolution2D(filters = 32,
                        kernel_size = (3, 3)
                        ))
model.add(Activation(activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.5))
##########

# 3 Camada #
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
##########

# 4 Camada #
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
##########


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

####################################################################


############### VER A CONFIGURAÇÃO DO MODELO #######################
'''
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
'''
####################################################################


##################### TREINAR O MODELO #############################

x,y = shuffle(np_lista_imgs,Y, random_state=2)                                               #Embaralhando a lista de imagens
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=2)     #Separando as imagens para o treinamento ( TREINAMENTO | VALIDAÇÃO )

model.fit(x = X_train,                                                                       #Finalmente, treinando a rede!!
          y = y_train,
          batch_size=8,
          epochs=NUMERO_EPOCAS,
          verbose=1,
          validation_data=(X_test, y_test))

####################################################################


################ Grafico de Acertos e Perdas #######################
'''
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(train_loss,xc)
plt.plot(val_loss,xc)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(train_acc,xc)
plt.plot(val_acc,xc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
'''
####################################################################

#################### Avaliar uma imagem ############################

IMAGEM_TESTADA_PATH = 'data2/S2/Dano2_S2/d7.jpg'

''' Avaliar todas as imagens no bloco de validação.

score = model.evaluate(X_test, y_test, True, 0)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])

test_image = X_test
model.predict_classes(test_image, batch_size=8, verbose=1)
#print(y_test[0:1])

'''

test_image = cv2.imread(IMAGEM_TESTADA_PATH)
test_image = test_image[49:585, 114:792]

test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(linhas_img,colunas_img))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255


test_image= np.expand_dims(test_image, axis=3)
test_image= np.expand_dims(test_image, axis=0)


print((model.predict(test_image)))
print(model.predict_classes(test_image))


####################################################################



################### Visualizar img camada ##########################
# É preciso que esse bloco seja executado em conjunto com o bloco
# "Avaliar uma imagem" para que a variavel test_image seja iniciada.

NUM_CAMADA=2


def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

activations = get_featuremaps(model, int(NUM_CAMADA),test_image)

print (np.shape(activations))
feature_maps = activations[0][0]
print (np.shape(feature_maps))

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))
plt.title("featuremaps-layer-{}".format(NUM_CAMADA))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	ax.imshow(feature_maps[:,:,i],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()

####################################################################



###################### Matriz confusão #############################
# Printing the confusion matrix

from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score
import itertools

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ['class 0(d1)', 'class 1(d2)', 'class 2(d3)','class 3(integro)']

print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

print(cohen_kappa_score(np.argmax(y_test,axis=1), y_pred))



# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)
plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()

####################################################################


################### Salvando Pesos da CNN ##########################

from keras.models import load_model

model.save_weights("model.h5")
model.load_model('model.h5')

####################################################################
