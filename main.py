# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import random

# %%
data = []
labels = []
details = []
cur_path = os.getcwd()
classes = 43

# %%
for i in range(classes): 
    path = os.path.join(cur_path, 'dataset', 'train', str(i)) 
    images = os.listdir(path) 
    for a in images: 
        try: 
            image = Image.open(path+"\\"+a) 
            image = image.resize((30,30)) 
            image = np.array(image) 
            data.append(image) 
            labels.append(i)
            details.append({
                "full_path": path+"\\"+a,
                "image_name": a,
                "category": i
            })
        except: 
            print("Error loading image") 
data = np.array(data)
labels = np.array(labels)

# %%
print(data.shape, labels.shape)

# %%
detail_df = pd.DataFrame(details)
detail_df.head()

# %%
sign_df = pd.read_csv("./signnames.csv")
sign_dict = sign_df.to_dict()
sign_list = [item for item in sign_dict['SignName'].values()]

# %%
Image_dir = './dataset/Train/'

num_samples = 9
image_files = os.listdir(Image_dir)

# Randomly select num_samples images
rand_images = random.sample(image_files, num_samples)
fig, axes = plt.subplots(3, 3, figsize=(11, 11))

for i in range(num_samples):
    image = rand_images[i]
    sample_dir_path = os.path.join(Image_dir, image)
    sample_dir_list = os.listdir(sample_dir_path)
    random_image = random.choice(sample_dir_list)
    ax = axes[i // 3, i % 3]
    ax.imshow(plt.imread(os.path.join(sample_dir_path,random_image)))
    ax.set_title(sign_list[int(image)])
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
value_counts = detail_df['category'].value_counts()

plt.figure(figsize=(10,5))
# Plotting
plt.bar(value_counts.index, value_counts.values)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Count Bar Chart')
plt.show()

# %%
# Top 10 Category
display(value_counts.head(10))

# %% [markdown]
# - The count of id's 1,2,4,10,12,13 and 38 are high in number when compared to other sign id's.
# - The sign id's 0,19,31 and 38 are least in number.

# %%
#assigning class id's under the category mentioned in research paper.
prohibitory = [0,1,2,3,4,5,7,8,9,10,15,16]
danger = [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
mandatory = [33,34,35,36,37,38,39,40]
other = [6,12,13,14,17,32,41,42]

# %%
df = detail_df['category']
detail_df['Object Name'] = detail_df['category']
#assigning new labels, 1 implies prohibitory,2 implies danger,3 implies mandatory and 4 implies other.
for i in range(len(df)):
  if(df[i] in prohibitory):
    df.loc[i]=0
    detail_df['Object Name'].loc[i]='prohibitory'
  elif(df[i] in danger):
    df.loc[i]=1
    detail_df['Object Name'].loc[i]='danger'
  elif(df[i] in mandatory):
    df.loc[i]=2
    detail_df['Object Name'].loc[i]='mandatory'
  elif(df[i] in other):
    df.loc[i]=3
    detail_df['Object Name'].loc[i]='other'
  else:
    df.loc[i]=-1

# %%
# display(detail_df)
value_counts = detail_df['Object Name'].value_counts()

plt.figure(figsize=(8,4))
# Plotting
plt.bar(value_counts.index, value_counts.values)
plt.xlabel('Category Type')
plt.ylabel('Count')
plt.title('Count Object Chart')
plt.show()

# %%
detail_df.head()

# %%
#Splitting training and testing dataset
X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_t1.shape, X_t2.shape, y_t1.shape, y_t2.shape)

# %%
#Converting the labels into one hot encoding
y_t1 = to_categorical(y_t1, 43)
y_t2 = to_categorical(y_t2, 43)

# %%
#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape= X_t1.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# %%
#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
eps = 15
anc = model.fit(X_t1, y_t1, batch_size=32, epochs=eps, validation_data=(X_t2, y_t2))
model.save("./model/init_model.h5")

# %%
print(anc.history)

# %%
plt.figure(figsize=(12, 5))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(anc.history['accuracy'], label='Training Accuracy')
plt.plot(anc.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(anc.history['loss'], label='Training Loss')
plt.plot(anc.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


# %%
#testing accuracy on test dataset
from sklearn.metrics import accuracy_score
y_test = pd.read_csv('./dataset/Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
for img in imgs:
   image = Image.open("./dataset/" + img)
   image = image.resize((30,30))
   data.append(np.array(image))
X_test=np.array(data)
pred = model.predict(X_test)

# %%
#Accuracy with the test data
from sklearn.metrics import accuracy_score
# Convert probabilities to class labels by taking the argmax
pred_labels = np.argmax(pred, axis=1)
print("Accuracy:",accuracy_score(labels, pred_labels))
model.save('./model/traffic_classifier.h5')

# %%



