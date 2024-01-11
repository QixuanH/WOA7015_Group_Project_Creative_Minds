import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import pandas as pd
import ssl
import seaborn as sns
ssl._create_default_https_context = ssl._create_unverified_context

# set path
data_path = "ssdb/output/"
save_model_path = "ResNetCRNN_ckpt/"

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.3       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 3             # number of target category
epochs = 10        # training epochs
batch_size = 16
learning_rate = 8e-5
log_interval = 10   # interval for displaying training info

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 0, 28, 1


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
        
        print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}")

    return losses, scores

def test(model, device, optimizer, test_loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1, )
            output = rnn_decoder(cnn_encoder(X))
            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()
            y_pred = output.max(1, keepdim=True)[1]
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    all_y = all_y.cpu().data.squeeze().numpy()
    all_y_pred = all_y_pred.cpu().data.squeeze().numpy()

    # 计算指标
    precision = precision_score(all_y, all_y_pred, average='weighted')
    recall = recall_score(all_y, all_y_pred, average='weighted')
    f1 = f1_score(all_y, all_y_pred, average='weighted')
    print('\n测试集: 平均损失: {:.4f}, 准确率: {:.2f}%, 精确率: {:.2f}%, 召回率: {:.2f}%, F1分数: {:.2f}%\n'.format(
        test_loss, 100 * accuracy_score(all_y, all_y_pred), 100 * precision, 100 * recall, 100 * f1))
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(all_y, all_y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap="Blues")
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.title('Confusion Martix')
    title = "./ResNetCRNN_ConfusionMatrix.png"
    plt.savefig(title, dpi=600)
    plt.close

    return test_loss, precision, recall, f1

def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score

if __name__ == '__main__':
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    # Data loading parameters
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4} 


    # load UCF101 actions names
    # with open(action_name_path, 'rb') as f:
    #     action_names = pickle.load(f)
    action_names = ['0','1', '2']
    # print(action_names)
    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)
    # show how many classes there are
    list(le.classes_)
    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)
    # # # # example
    # y = ['low', 'high']
    # y_onehot = labels2onehot(enc, le, y)
    # y2 = onehot2labels(le, y_onehot)
    # print(y2)


    # 读取 Excel 文件以获取视频路径
    excel_path = 'ssdb/filtered_output_armflapping.xlsx'  # 替换为您的 Excel 文件路径
    df = pd.read_excel(excel_path)

    all_X_list = df['frames_paths'].to_list()

    all_Y_list = df['Intensity'].tolist()
    all_Y_list = labels2cat(le, all_Y_list)

    import random

    def shuffle_lists(x_list, y_list):
        # 将两个列表合并为元组列表，然后一起打乱
        combined = list(zip(x_list, y_list))
        random.shuffle(combined)

        # 再次分离列表
        x_list_shuffled, y_list_shuffled = zip(*combined)
        return list(x_list_shuffled), list(y_list_shuffled)

    
    all_X_list, all_Y_list = shuffle_lists(all_X_list, all_Y_list)


    # train, test split
    train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_Y_list, test_size=0.25, random_state=42)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()


    train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                        Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)
    

    train_loader = data.DataLoader(train_set, **params)

    valid_loader = data.DataLoader(valid_set, **params)

    # Create model
    cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                            h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn_encoder = nn.DataParallel(cnn_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)

        # Combine all EncoderCNN + DecoderRNN parameters
        crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                    list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                    list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        # Combine all EncoderCNN + DecoderRNN parameters
        crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                    list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                    list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

    else: 
        crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                    list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                    list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

    optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)


    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    # start training
    for epoch in range(epochs):

        print(f"start training, epochs: {epoch}")
        # train, test model
        train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
        np.save('./CRNN_epoch_training_losses.npy', A)
        np.save('./CRNN_epoch_training_scores.npy', B)
        np.save('./CRNN_epoch_test_loss.npy', C)
        np.save('./CRNN_epoch_test_score.npy', D)

        print(f"Epoch {epoch+1}, train_losses: {train_losses}, train_scores: {train_scores}, test_loss: {epoch_test_loss}, test_score: {epoch_test_score}")
    
    
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    test_set = Dataset_CRNN(data_path, all_X_list, all_Y_list, selected_frames, transform=transform)
    
    test_loader = data.DataLoader(test_set, **params)

    test([cnn_encoder, rnn_decoder], device, optimizer, test_loader)

    # plot
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
    plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validate'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
    plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
    plt.title("training accuracy")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validate'], loc="upper left")
    title = "./ResNetCRNN.png"
    plt.savefig(title, dpi=600)
    plt.close(fig)