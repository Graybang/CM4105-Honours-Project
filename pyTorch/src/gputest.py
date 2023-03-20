import tensorflow as tf
import torch

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Computation device: ', device)