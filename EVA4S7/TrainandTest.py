from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch, model_flag):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    if (model_flag != "M1") or (model_flag != "M2") or (model_flag != "M5")or (model_flag != "M6")  :        
        reg_loss = 0
        for param in model.parameters():                
                reg_loss += torch.sum(torch.abs(param))
        factor = 0.0005
        loss += factor * reg_loss
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader,model_flag):
    model.eval()
    test_loss = 0
    correct = 0
    flg = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Saving misclassified Images and their actual and pedicted labels
            tgt = target.view_as(pred)
            comp_df= pred.eq(tgt)
            mis_c = ~comp_df
            if flg == 0:
                misc_im = data[mis_c]
                misc_tr = tgt[mis_c]
                misc_pred = pred[mis_c]
                flg =1
            else:  
                misc_im = torch.cat((data[mis_c],misc_im))
                misc_tr = torch.cat((tgt[mis_c],misc_tr))
                misc_pred = torch.cat((pred[mis_c],misc_pred))

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    return misc_im,misc_tr,misc_pred

    from torch.optim.lr_scheduler import StepLR

model_types = ["M1","M2","M3","M4","M5","M6","M7","M8"]
#model_types = ["M1","M2","M3"]
EPOCHS = 25
import pandas as pd
import numpy as np

for model_flag in model_types:
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    #Without L1/L2 with BN
    if model_flag == "M1" :
      model =  Net().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    elif model_flag == "M2" :
      model =  NetGBN().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)      
      
    elif model_flag == "M3" :
      model =  Net().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
      
    elif model_flag == "M4" :     
      model =  NetGBN().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)      

    elif model_flag == "M5" :
      model =  Net().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.0005 )

    elif model_flag == "M6" :
      model =  NetGBN().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.0005 )      

    elif model_flag == "M7" :
      model =  Net().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.0005 )

    elif model_flag == "M8" :
      model =  NetGBN().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.0005 )      
    
        

    for epoch in range(EPOCHS):
      print("EPOCH:", epoch)
      if (model_flag == "M2") or (model_flag == "M4") or (model_flag == "M6") or (model_flag == "M8"): 
        train(model, device, train_loader2, optimizer, epoch,model_flag)
      else:
        train(model, device, train_loader, optimizer, epoch,model_flag)

      if model_flag == "M1":
        misc_im,misc_tr,misc_pred = test(model, device, test_loader,model_flag)
      elif model_flag == "M2":
        misc_im_l2,misc_tr_l2,misc_pred_l2 = test(model, device, test_loader,model_flag)
      else:  
        test(model, device, test_loader,model_flag)

    df_list = pd.DataFrame(np.column_stack([test_losses, test_acc]),columns=['test_losses', 'test_acc'])
    df_list['epoch'] = range(0,EPOCHS)    

    
    if (model_flag == "M1"):
      df_list = df_list.rename(columns = {"test_losses": "WO_reg_loss", "test_acc":"WO_reg_Accuracy"})
      df = df_list.copy(deep = True)
    else:
      df_list = df_list.rename(columns = {"test_losses": model_flag+"_loss", "test_acc":model_flag+"_Accuracy"})        
      df = pd.merge(left=df.copy(deep = True), right=df_list.copy(deep = True), how='inner')