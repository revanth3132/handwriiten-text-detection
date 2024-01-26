import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Get the current user's home directory
user_home = os.path.expanduser("~")

# Define the relative path from the home directory to the Downloads folder
downloads_path = os.path.join(user_home, "Downloads")
# Load train and validation datasets
_train_data = pd.read_csv(downloads_path + "\\_AI_\\CSV\\written_name_train.csv")
_valid_data = pd.read_csv(downloads_path + "\\_AI_\\CSV\\written_name_validation.csv")
# Load the test dataset
_test_data = pd.read_csv(downloads_path + "\\_AI_\\CSV\\written_name_test.csv")
# Function to split the data equally for each alphabet
def split_data_equally(df, num_samples_per_alphabet):
    result_df = pd.DataFrame(columns=df.columns)
    df = df.dropna(subset=['IDENTITY'])  # Drop rows with NaN values in 'IDENTITY'

    unique_first_letters = df['IDENTITY'].str[0].unique()

    for letter in unique_first_letters:
        letter_data = df[df['IDENTITY'].str.startswith(letter)]
        sampled_data = letter_data.sample(min(num_samples_per_alphabet, len(letter_data)), random_state=42)
        result_df = pd.concat([result_df, sampled_data], ignore_index=True)

    return result_df

# Number of samples per alphabet for training and validation
num_samples_per_alphabet_train = 1500
num_samples_per_alphabet_valid = 100
samples_test = 15
# Split the training data
_train_data = split_data_equally(_train_data, num_samples_per_alphabet_train)

# Split the validation data
_valid_data = split_data_equally(_valid_data, num_samples_per_alphabet_valid)

#Split the test data
_test_data = split_data_equally(_test_data, samples_test)
# Data preprocessing and cleaning
print("Number of NaNs in train set      : ", _train_data['IDENTITY'].isnull().sum())
print("Number of NaNs in validation set : ", _valid_data['IDENTITY'].isnull().sum())
print("Number of NaNs in test set : ", _test_data['IDENTITY'].isnull().sum())

_train_data.dropna(axis=0, inplace=True)
_valid_data.dropna(axis=0, inplace=True)
_test_data.dropna(axis=0, inplace=True)

unreadable = _train_data[_train_data['IDENTITY'] == 'UNREADABLE']
unreadable.reset_index(inplace=True, drop=True)

_train_data = _train_data[_train_data['IDENTITY'] != 'UNREADABLE']
_valid_data = _valid_data[_valid_data['IDENTITY'] != 'UNREADABLE']
_test_data = _test_data[_valid_data['IDENTITY'] != 'UNREADABLE']

_train_data['IDENTITY'] = _train_data['IDENTITY'].str.upper()
_valid_data['IDENTITY'] = _valid_data['IDENTITY'].str.upper()
_test_data['IDENTITY'] = _test_data['IDENTITY'].str.upper()

_train_data.reset_index(inplace=True, drop=True)
_valid_data.reset_index(inplace=True, drop=True)
_test_data.reset_index(inplace=True, drop=True)

def preprocess(image):
    target_height, target_width = 64, 256
    height, width = image.shape

    # Crop or pad width to 256
    if width > target_width:
        image = image[:, :target_width]
    elif width < target_width:
        padding = np.ones((height, target_width - width)) * 255
        image = np.concatenate((image, padding), axis=1)

    # Crop or pad height to 64
    if height > target_height:
        image = image[:target_height, :]
    elif height < target_height:
        padding = np.ones((target_height - height, target_width)) * 255
        image = np.concatenate((image, padding), axis=0)

    # Rotate image clockwise by 90 degrees
    final_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return final_image

train_size = len(_train_data)
valid_size = len(_valid_data)
test_size = len(_test_data)

print("trainsize",train_size)
print("validsize", valid_size)
print("testsize", test_size)

train_x = []
valid_x = []
test_x = []

# Preprocess and normalize training images
for i in range(train_size):
    img_dir = downloads_path +"\\_AI_\\train_v2\\train\\" + _train_data.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image) / 255.
    train_x.append(image)

# Preprocess and normalize validation images
for i in range(valid_size):
    img_dir = downloads_path +"\\_AI_\\validation_v2\\validation\\" + _valid_data.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image) / 255.
    valid_x.append(image)

# Preprocess and normalize test images
for i in range(test_size):
    img_dir = downloads_path +"\\_AI_\\test_v2\\test\\" + _test_data.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image) / 255.
    test_x.append(image)

train_x = np.array(train_x).reshape(-1, 256, 64, 1)
valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)
test_x = np.array(test_x).reshape(-1, 256, 64, 1)

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24
num_of_characters = len(alphabets) + 1
num_of_timestamps = 64

# Convert labels to numeric format
def label_to_num(label):
    return np.array([alphabets.find(ch) for ch in label])

# Convert numeric labels back to string format
def num_to_label(num):
    return ''.join(alphabets[ch] for ch in num if ch != -1)


# Convert labels to numeric format
train_y = -np.ones((train_size, max_str_len))
train_label_len = np.zeros((train_size, 1))
train_input_len = (num_of_timestamps - 2) * np.ones((train_size, 1))
train_output = np.zeros(train_size)

for i in range(train_size):
    label_num = label_to_num(_train_data.loc[i, 'IDENTITY'])[:max_str_len]
    train_y[i, :len(label_num)] = label_num
    train_label_len[i] = len(label_num)

valid_y = -np.ones((valid_size, max_str_len))
valid_label_len = np.zeros((valid_size, 1))
valid_input_len = (num_of_timestamps - 2) * np.ones((valid_size, 1))
valid_output = np.zeros(valid_size)

for i in range(valid_size):
    label_num = label_to_num(_valid_data.loc[i, 'IDENTITY'])[:max_str_len]
    valid_y[i, :len(label_num)] = label_num
    valid_label_len[i] = len(label_num)

# Convert labels to numeric format for test
test_y = -np.ones((test_size, max_str_len))
test_label_len = np.zeros((test_size, 1))
test_input_len = (num_of_timestamps - 2) * np.ones((train_size, 1))
test_output = np.zeros(test_size)

for i in range(test_size):
    label_num = label_to_num(_test_data.loc[i, 'IDENTITY'])[:max_str_len]
    test_y[i, :len(label_num)] = label_num
    test_label_len[i] = len(label_num)

print(f"True label: {_train_data.loc[100, 'IDENTITY']}\n"
      f"train_y: {train_y[100]}\n"
      f"train_label_len: {train_label_len[100]}\n"
      f"train_input_len: {train_input_len[100]}")

input_data = Input(shape=(256, 64, 1), name='input')

inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
inner = Dropout(0.3)(inner)

# CNN to RNN
inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

## RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

## OUTPUT
inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

model = Model(inputs=input_data, outputs=y_pred)
model.summary()
# CTC loss function
def custom_ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = Lambda(lambda x: x[:, 2:, :])(y_pred)
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Define the inputs for the CTC loss
labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# Apply the custom CTC loss function
ctc_loss = Lambda(custom_ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

# Create the final model with the custom CTC loss
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

# Compile the final model with additional metrics
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                    optimizer=Adam(lr=0.0001),
                    metrics=['accuracy', Precision(), Recall(), AUC()])


# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("handwritten_text_predictor_weights.h5", save_best_only=True)
learning_rate_scheduler = LearningRateScheduler(lambda epoch: 0.0001 if epoch < 30 else 0.00001)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Train the model with additional metrics
history = model_final.fit(
    x=[train_x, train_y, train_input_len, train_label_len],
    y=train_output,
    validation_data=(
        [valid_x, valid_y, valid_input_len, valid_label_len],
        valid_output
    ),
    epochs=65,
    batch_size=128,
    callbacks=[early_stopping, model_checkpoint, learning_rate_scheduler, tensorboard]
)
# Evaluate the model on the validation set
preds = model.predict(valid_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1], greedy=True)[0][0])

prediction = [num_to_label(seq) for seq in decoded]

# Calculate and display standard metrics
validation_accuracy = accuracy_score(_valid_data['IDENTITY'], prediction)
validation_precision = precision_score(_valid_data['IDENTITY'], prediction, average='weighted')
validation_recall = recall_score(_valid_data['IDENTITY'], prediction, average='weighted')
validation_f1 = f1_score(_valid_data['IDENTITY'], prediction, average='weighted')

print('validation_Accuracy:', validation_accuracy)
print('validation_Precision:', validation_precision)
print('validation_Recall:', validation_recall)
print('validation_F1-Score:', validation_f1)


# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Calculate additional metrics
correct_char = sum([sum(1 for tr_char, pr_char in zip(tr, pr) if tr_char == pr_char) for tr, pr in zip(_valid_data['IDENTITY'], prediction)])
total_char = sum(len(tr) for tr in _valid_data['IDENTITY'])
correct_word = sum(pr == tr for pr, tr in zip(prediction, _valid_data['IDENTITY']))

print('Correct characters predicted : %.2f%%' % (correct_char * 100 / total_char))
print('Correct words predicted      : %.2f%%' % (correct_word * 100 / valid_size))

plt.show()

# Evaluate the model on the test set
preds = model.predict(test_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1], greedy=True)[0][0])

prediction = [num_to_label(seq) for seq in decoded]

# Calculate and display standard metrics
test_accuracy = accuracy_score(_test_data['IDENTITY'], prediction)
test_precision = precision_score(_test_data['IDENTITY'], prediction, average='weighted')
test_recall = recall_score(_test_data['IDENTITY'], prediction, average='weighted')
test_f1 = f1_score(_test_data['IDENTITY'], prediction, average='weighted')

print("For test data")
print('test_Accuracy:', test_accuracy)
print('test_Precision:', test_precision)
print('test_Recall:', test_recall)
print('test_F1-Score:', test_f1)


# Calculate additional metrics
correct_char = sum([sum(1 for tr_char, pr_char in zip(tr, pr) if tr_char == pr_char) for tr, pr in zip(_test_data['IDENTITY'], prediction)])
total_char = sum(len(tr) for tr in _test_data['IDENTITY'])
correct_word = sum(pr == tr for pr, tr in zip(prediction, _test_data['IDENTITY']))

print('Correct characters predicted : %.2f%%' % (correct_char * 100 / total_char))
print('Correct words predicted      : %.2f%%' % (correct_word * 100 / valid_size))

plt.show()


# Visualize predictions on test data for images 10 to 15
plt.figure(figsize=(15, 10))
for i in range(10, 16):  # Adjusted the range for images 10 to 15
    ax = plt.subplot(2, 3, i - 9)  # Adjusted the subplot index
    img_dir = downloads_path + "\\AI RUCHITH\\test_v2\\test\\" + _test_data.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')

    image = preprocess(image)
    image = image / 255.
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
    plt.title("Predicted: " + num_to_label(decoded[0]) + "\nActual: " + _test_data.loc[i, 'IDENTITY'], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)
plt.show()
# Save the model architecture as JSON
model_json = model.to_json()
with open("handwritten_text_predictor_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model_final.save_weights("handwritten_text_predictor_weights.h5")
