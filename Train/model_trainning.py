from Data import data
from Model import DL_model
import matplotlib.pyplot as plt

train, val = data.data_provider(200, 200, 32, 0.2, mpath='../Data/')
model = DL_model.face_model(200, 200, 3)

epochs = 2

history = model.fit(train, validation_data=val, epochs=epochs)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


model.save_weights('SavedModel/Weights/' + str(epochs) + '.hdf5')
print('the model was saved successfully')

model.save('SavedModel/Entire/')

model.evaluate(val)