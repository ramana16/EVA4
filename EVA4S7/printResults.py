print(misc_im_l2.shape)
print(misc_im.shape)
print(misc_tr_l2.shape)
print(misc_tr.shape)
print(misc_pred_l2.shape)
print(misc_pred.shape)
%matplotlib inline
import matplotlib.pyplot as plt 

fig=plt.figure(figsize=(14, 16))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    img = misc_im[i-1].to(torch.device("cpu"))
    p = misc_pred[i-1].to(torch.device("cpu"))
    t = misc_tr[i-1].to(torch.device("cpu"))
    #img = img.permute(1, 2, 0)
    fig.add_subplot(rows, columns, i)
    #plt.imshow(img[:, :, 0].numpy().squeeze(),cmap='gray_r')
    plt.imshow(img.numpy().squeeze(),cmap='gray_r')
    plt.title("Predicted:"+str(p)[7:8]+"  Actual: "+str(t)[7:8])
#plt.show()
from google.colab import files
plt.savefig("L1_Misclassified_Images.png")
files.download("L1_Misclassified_Images.png")

fig=plt.figure(figsize=(14, 16))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    img = misc_im_l2[i-1].to(torch.device("cpu"))
    p = misc_pred_l2[i-1].to(torch.device("cpu"))
    t = misc_tr_l2[i-1].to(torch.device("cpu"))
    #img = img.permute(1, 2, 0)
    fig.add_subplot(rows, columns, i)
    #plt.imshow(img[:, :, 0].numpy().squeeze(),cmap='gray_r')
    plt.imshow(img.numpy().squeeze(),cmap='gray_r')
    plt.title("Predicted:"+str(p)[7:8]+"  Actual: "+str(t)[7:8])
#plt.show()

from google.colab import files
plt.savefig("L2_Misclassified_Images.png")
files.download("L2_Misclassified_Images.png")