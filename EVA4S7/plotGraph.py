%matplotlib inline
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(df['WO_reg_loss'],label='without L1_L2 with BN')
ax.plot(df['M2_loss'],label='without L1_L2 with GBN')
ax.plot(df['M3_loss'],label='L1 with BN')
ax.plot(df['M4_loss'],label='L1 with GBN')
ax.plot(df['M5_loss'], label='L2 with BN')
ax.plot(df['M6_loss'], label='L2 with GBN')
ax.plot(df['M7_loss'], label='L1 and L2 with BN')
ax.plot(df['M8_loss'], label='L1 and L2 with GBN')
plt.autoscale(enable = True, axis = 'both',tight = 'true')
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
leg = ax.legend();

from google.colab import files
plt.savefig("validation_loss.png")
files.download("validation_loss.png")

fig, ax = plt.subplots()
ax.plot(df['WO_reg_Accuracy'],label='without L1_L2 with BN')
ax.plot(df['M2_Accuracy'],label='without L1_L2 with GBN')
ax.plot(df['M3_Accuracy'],label='L1 with BN')
ax.plot(df['M4_Accuracy'],label='L1 with GBN')
ax.plot(df['M5_Accuracy'], label='L2 with BN')
ax.plot(df['M6_Accuracy'], label='L2 with GBN')
ax.plot(df['M7_Accuracy'], label='L1 and L2 with BN')
ax.plot(df['M8_Accuracy'], label='L1 and L2 with GBN')
plt.autoscale(enable = True, axis = 'both',tight = 'true')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
leg = ax.legend();

from google.colab import files
plt.savefig("Validation_Accuracy.png")
files.download("Validation_Accuracy.png")