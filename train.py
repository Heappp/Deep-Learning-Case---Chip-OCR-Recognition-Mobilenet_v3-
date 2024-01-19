import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from data import MDataset
from mobilenet_v3_small import mobilenet_v3_small

net = mobilenet_v3_small()
loss_function = nn.CrossEntropyLoss(label_smoothing=0.2)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

epoch = 10
batch_size = 128


dataset = MDataset('dataset')
train_dataset, val_dataset = random_split(dataset, [2000, 254])


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)


for step in range(epoch):
    # 训练
    net.train()
    train_loss, train_acc = 0, 0
    for img, label in train_dataloader:

        optimizer.zero_grad()
        y = net.forward(img)
        loss = loss_function(y, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()
    scheduler.step()

    # 测试
    # net.eval()
    val_loss, val_acc = 0, 0
    for img, label in val_dataloader:

        y = net.forward(img)
        loss = loss_function(y, label)

        val_loss += loss.item()
        val_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()

    # 统计
    print("---------------", step + 1, "---------------")
    print("Loss:", train_loss / len(train_dataloader), val_loss / len(val_dataloader))
    print("Acc: ", train_acc / len(train_dataset), val_acc / len(val_dataset))
    print()

    # 保存模型
    torch.save(net.state_dict(), "mobilenet_v3_small.pt")
