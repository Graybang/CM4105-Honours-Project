import matplotlib.pyplot as plt

train_psnr = [25.725,
26.546,
26.685,
26.868,
27.109,
27.266,
27.371,
27.46,
27.54,
27.632]
val_psnr = [26.482,
26.632,
26.753,
26.951,
27.107,
27.249,
27.335,
27.44,
27.527,
27.567] 



plt.figure(figsize=(6, 4))
plt.xlim([-1, 10])
plt.plot(train_psnr, color='green', label='train PSNR dB')
plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.savefig('C:/Users/Percy/OneDrive - Robert Gordon University/CM4105-Honours/CM4105-Honours-Project/pyTorch/src/outputs_stereo/psnr_10.svg')
plt.show()