## 前言

这是深度学习的一个入门小案例，非常适合用来参考和学习。在这里，我们会使用深度学习的网络来对芯片进行识别分类。其中我会介绍本次使用的数据集、数据集的预处理过程和网络的训练过程。另外如有问题的地方或者不理解的地方，可以及时联系我。另外我会非常乐意与大家交流分享一些技术上的知识，[此篇博客](http://echoself.com/index.php/2024/01/19/373/)完整代码和数据集我将会放在我的[Github](https://github.com/Heappp/Deep-Learning-Case---Chip-OCR-Recognition-Mobilenet_v3-)上，有需要请自取哈。**公式不能正常显示请刷新一次！！**

## 数据集

该数据集总共包含2254张灰度图像，其中被划分为训练集和测试集。在这2254张图片中，训练集包含了2000张，用于深度学习模型的训练过程。而测试集则包含了剩余的254张图片，被用于评估训练好的模型在未知数据上的性能表现。

另外该数据集有22个类别，分别为{'74LS00': 0, '74LS01': 1, '74LS02': 2, '74LS03': 3, '74LS05': 4, '74LS06': 5, '74LS09': 6, '74LS11': 7, '74LS112': 8, '74LS123': 9, '74LS14': 10, '74LS169': 11, '74LS175': 12, '74LS190': 13, '74LS191': 14, '74LS192': 15, '74LS193': 16, '74LS42': 17, '74LS47': 18, '74LS48': 19, '74LS83': 20, '74LS85': 21}。

在此数据集当中，其中每个类别的图片数目大致相同，并不需做数据平衡处理。每个类别的所有图片放在同一文件夹内，方便读取和处理。数据集如片示例如下，标签分别为'74LS00': 0和'74LS01': 1。
<table><tr>
<td><img src=image/1.bmp style="zoom:20%;" align="right"></td>
<td><img src=image/2.bmp style="zoom:21%;" align="left"></td>
</tr></table>

## 数据预处理

现在我们的任务是对该数据集进行分类，在识别的过程当中将芯片上面的字符作为一个整体进行识别，这是一个分类任务。在数据预处理的过程当中我们需要考虑芯片的角度问题、字符的清晰度以及芯片放缩的大小问题。这些我们都可以python中的opencv库来对数据进行预处理，使得在数据输入神经网络前不同类别的特征区分最为明显。预处理代码如下：

```python
    def pretreatment(image):
        # 二值化
        image = cv2.medianBlur(image, ksize=3)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -2)

        # 将矩形部分旋转为正
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(max_contour)
        angle = rect[2] if rect[1][0] > rect[1][1] else rect[2] - 90

        M = cv2.getRotationMatrix2D(rect[0], angle, 1)
        image = cv2.warpAffine(image, M, (1280, 960))

        # 截取矩形部分
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)

        w, h = max(rect[1][1], rect[1][0]), min(rect[1][1], rect[1][0])
        x, y = int(rect[0][0] - w / 2), int(rect[0][1] - h / 2)
        image = image[y:y + int(h), x:x + int(w)]

        # 开运算
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

        return image
```

#### 中值滤波

首先对图像进行中值滤波操作，中值滤波是一种常见的数字图像处理方法，它的作用是去除图像中的噪声，并尽量保留原始图像中的边缘和细节信息。中值滤波通过将每个像素周围的邻域像素排序并取中位数来进行滤波处理。由于中位数具有抗噪性能，因此中值滤波可以有效地去除图像中的椒盐噪声等离群点，同时又能够保留原始图像中的边缘和细节信息。

使用cv2.medianBlur(image, ksize=3)可以对原图像进行中值滤波操作，其中ksize取值为3。中值滤波操作将会使得图像更为平滑，可以有效去除图像中的噪声，使得在后续的二值化操作中找到一个更为合适的阈值。

#### 二值化

二值化方法有很多种，其中adaptiveThreshold算法是一种基于局部区域的自适应阈值法。其中将图像分成若干个小块，对每个小块单独计算一个阈值。由于不同区域的光照和对比度可能存在差异，因此使用相同的全局阈值不一定能得到理想的二值化结果，而自适应阈值化可以根据图像局部特点进行阈值计算，能够更准确地将图像进行分割。其中二值化的局部大小取值为41，阈值偏移量设置的-2，并采用高斯加权平均来计算每个小区域的阈值。处理前后对比如下：

<table><tr>
<td><img src=image/3.png style="zoom:20%;" align="right"></td>
<td><img src=image/4.png style="zoom:20%;" align="left"></td>
</tr></table>

#### 旋转芯片

数据集当中的芯片有很多不同的角度，现在我们将该图像进行旋转，使芯片能够旋转为正。旋转前后对比图像如下所示：

1. 调用cv2.findContours函数获取输入图像中的所有轮廓，cv2.RETR_EXTERNAL表示只检测最外层轮廓，返回值contours为检测到的所有轮廓。返回值contours记录了图像中所有矩形的轮廓点。
2. 我们现在需要最大的芯片矩形部分的轮廓，所以使用cv2.contourArea来作为max函数的key，找出面积最大的轮廓出来。
3. 使用cv2.minAreaRect函数获取最小外接矩形rect。最小外接矩形是能够包含轮廓的在最小矩形，其中包括矩形的中心点、宽度、高度和角度信息。具体而言，函数返回的对象是一个元组 (rect)，包含以下信息：
   - rect[0]：矩形的中心点坐标 (x, y)。
   - rect[1]：矩形的宽度和高度 (width, height)。
   - rect[2]：矩形的旋转角度，范围为 -90 度到 +90 度。
4. 使用cv2.getRotationMatrix2D函数生成一个旋转矩阵M，该矩阵用于对图像进行旋转操作，并传入cv2.warpAffine函数将输入图像image进行旋转操作，并将旋转后的图像大小设置为(1280, 960)。cv2.warpAffine函数可以根据指定的旋转矩阵对图像进行仿射变换，实现旋转、平移、缩放等操作。

<table><tr>
<td><img src=image/4.png style="zoom:20%;" align="right"></td>
<td><img src=image/5.png style="zoom:20%;" align="left"></td>
</tr></table>

#### 截取矩形部分

旋转之后即可进行图像裁剪，将芯片矩形部分给裁剪出来，使得网络能够更好识别。过程步骤和***旋转芯片***的过程类似，获得图像的元组 (rect)信息后，对需要部分直接切片即可，截取前后对比图像如下：

<table><tr>
<td><img src=image/5.png style="zoom:20%;" align="right"></td>
<td><img src=image/6.png style="zoom:50%;" align="left"></td>
</tr></table>

#### 开运算

开运算（Opening）是一种形态学图像处理操作，通常用于去除图像中的噪声、平滑边缘和消除小尺寸的物体。其过程一般为先腐蚀后膨胀。开运算可以平滑图像，减少噪声，保留图像的整体结构。它在图像处理中被广泛应用于去除小的斑点、细线、毛刺等干扰物。**注意开运算是针对白色部分即灰度值为255的部分（黑色为背景，白色为前景）**，对比图如下：

<table><tr>
<td><img src=image/6.png style="zoom:50%;" align="right"></td>
<td><img src=image/7.png style="zoom:50%;" align="left"></td>
</tr></table>

## 网络介绍

mobilenet_v3_small是一种轻量级的卷积神经网络架构，用于在资源受限的设备上进行高效的图像分类和目标检测任务。它是 Google 在2019年提出的MobileNet系列的最新版本。通过引入一些新的设计策略和技术来进一步提高模型的性能和效率。

#### SE注意力模型

如下图所示，SE注意力机制首先使用全局平均池化将$B\times C\times W\times H$的特征图转化为$B\times C\times 1\times 1$，之后经过全连接层学习到一组$B\times C\times 1\times 1$的权重，最后和原图像进行矩形的点乘操作。通过引入SE注意力模型，可以自适应的调整每个通道的权重，这也将通道维度的学习进行了学习。

<center>
<img src=image/8.png style="zoom:50%;">
</center>

在学习权重的过程当中，使用了Hardsigmoid激活函数将权重限制到[0, 1]之间。与标准的Sigmoid函数相比，Hardsigmoid在x接近0时具有更快的饱和度，因此在计算上更加高效。参数reduction（这里取值为4）控制连接层参数大小。

```python
class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

#### 倒残差结构

介绍倒残差结构之前先介绍深度可分离卷积。深度可分离卷积是一种轻量级卷积神经网络（CNN）中常用的卷积操作，它可以在保持较高的精度的同时显著减少模型的计算量和参数数量。

- 如下图（左）：在常规卷积的过程当中，我们一般是使用$out\_channels\times in\_channels\times ksize\times ksize$的卷积核去卷积原特征图，每一个$in\_channels\times ksize\times ksize$的卷积核都会对应的去卷积大小为$B\times in\_channels\times W\times H$的特征图，之后再将所有的通道维度相加得到一个通道的特征图。所以最后需要out\_channels个$in\_channels\times ksize\times ksize$的卷积核得到$B\times out\_channels\times W\times H$的特征图。
- 如下图（右）：在深度可分离卷积的过程当中并没有采取原来常规卷积的方法，只用一个$in\_channels\times ksize\times ksize$的卷积核去卷积原特征图，卷积完成后并不相加，而是直接输出$B\times in\_channels\times W\times H$的特征图。这样在卷积的过程当中不能改变通道数目的大小，也割裂通道之间特征的联系，也就是减弱了网络的学习能力。

但是，这样的设计使得网络的参数量从$out\_channels\times in\_channels\times ksize\times ksize$减少到$in\_channels\times ksize\times ksize$，运算速度也大为提高，是一种轻量化的设计。而前面的不能改变通道数目的大小和割裂通道之间特征的联系也有相对应的解决方法，即卷积前后使用$1\times 1$的常规卷积改变通道数目和使用前面提到的SE注意力机制学习通道之间的联系。

<table><tr>
<td><img src=image/9.png style="zoom:36%;" align="right"></td>
<td><img src=image/10.png style="zoom:50%;" align="left"></td>
</tr></table>

现在来介绍倒残差结构。其实倒残差结构就是在卷积之前先使用大小为$1\times 1$的卷积核先升维，后经过深度可分离卷积后再降维，最后再进行残差的过程。所谓的“倒”就是区别于原来的先降维后升维，原因是想提高深度可分离卷积带来的“参数量太小”的问题。另外在网络中有些使用Hardswish激活函数。Swish在计算上与ReLU一样高效，并且在更深的模型上表现出比ReLU更好的性能，而Hardswish则在Swish的基础上使用了一个硬切割（hard clip）操作，以减少计算复杂性。

在代码的实现过程当中，se代表是否使用SE注意力机制，nl代表使用的激活函数，stride传入卷积的步长，通过卷积的步长可以判断是否进行残差操作（不同长宽的特征图无法相加）。实现代码如下：

```py
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, exp_size, se=True, nl='RE'):
        super(InvertedResidual, self).__init__()
        self.use_res = True if stride == (1, 1) and in_channels == out_channels else False
        self.activation_layer = nn.ReLU() if nl == 'RE' else nn.Hardswish()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=(1, 1)),
            nn.BatchNorm2d(exp_size),
            self.activation_layer,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, padding=kernel_size[0] // 2, groups=exp_size),
            nn.BatchNorm2d(exp_size),
            self.activation_layer,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SELayer(out_channels, 4) if se else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.se:
            y = self.se(y)
        if self.use_res:
            y = x + y
        return y
```

#### mobilenet_v3_small的实现

最后我们根据mobilenet_v3_small的结构表来实现这个网络，结构表如下图所示：

<center>
<img src=image/11.png style="zoom:80%;">
</center>

```py
class mobilenet_v3_small(nn.Module):
    def __init__(self):
        super(mobilenet_v3_small, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
        )
        self.bottlenecks = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, exp_size, se=True, nl='RE'
            InvertedResidual(16, 16, (3, 3), (2, 2), 16, True, 'RE'),
            InvertedResidual(16, 24, (3, 3), (2, 2), 72, False, 'RE'),
            InvertedResidual(24, 24, (3, 3), (1, 1), 88, False, 'RE'),

            InvertedResidual(24, 40, (5, 5), (2, 2), 96, True, 'HS'),
            InvertedResidual(40, 40, (5, 5), (1, 1), 240, True, 'HS'),
            InvertedResidual(40, 40, (5, 5), (1, 1), 240, True, 'HS'),
            InvertedResidual(40, 48, (5, 5), (1, 1), 120, True, 'HS'),
            InvertedResidual(48, 48, (5, 5), (1, 1), 144, True, 'HS'),
            InvertedResidual(48, 96, (5, 5), (2, 2), 288, True, 'HS'),
            InvertedResidual(96, 96, (5, 5), (1, 1), 576, True, 'HS'),
            InvertedResidual(96, 96, (5, 5), (1, 1), 576, True, 'HS'),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=(1, 1)),
            nn.BatchNorm2d(576),
            nn.Hardswish(),
            SELayer(576, 4),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(576, 1024, kernel_size=(1, 1)),
            nn.Hardswish(),
            nn.Conv2d(1024, 22, kernel_size=(1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        return x
```

## 网络的训练

#### 封装数据为Dataset类型

这里将预处理函数作为封装类的静态函数，另外还定义了MyRotateTransform来实现图像随机180度旋转，因为预处理后的图像可能是正的，也可能是倒的，所以需要网络来学习判断图像的正倒。实现代码如下：

```py
class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


class MDataset(Dataset):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        MyRotateTransform([0, 180]),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    name2label = {'74LS00': 0, '74LS01': 1, '74LS02': 2, '74LS03': 3, '74LS05': 4, '74LS06': 5, '74LS09': 6, '74LS11': 7, '74LS112': 8, '74LS123': 9, '74LS14': 10, '74LS169': 11, '74LS175': 12, '74LS190': 13, '74LS191': 14, '74LS192': 15, '74LS193': 16, '74LS42': 17, '74LS47': 18, '74LS48': 19, '74LS83': 20, '74LS85': 21}
    label2name = {0: '74LS00', 1: '74LS01', 2: '74LS02', 3: '74LS03', 4: '74LS05', 5: '74LS06', 6: '74LS09', 7: '74LS11', 8: '74LS112', 9: '74LS123', 10: '74LS14', 11: '74LS169', 12: '74LS175', 13: '74LS190', 14: '74LS191', 15: '74LS192', 16: '74LS193', 17: '74LS42', 18: '74LS47', 19: '74LS48', 20: '74LS83', 21: '74LS85'}

    def __init__(self, url):
        self.data, self.label = [], []
        for folder in MDataset.name2label.keys():
            filenames = os.listdir(url + '/' + folder)
            for filename in filenames:
                image = cv2.imread(url + '/' + folder + '/' + filename, cv2.IMREAD_GRAYSCALE)
                image = MDataset.pretreatment(image)

                self.data.append(image)
                self.label.append(MDataset.name2label[folder])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return MDataset.transform(self.data[index]), self.label[index]

    @staticmethod
    def pretreatment(image):
        # 二值化
        image = cv2.medianBlur(image, ksize=3)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -2)

        # 将矩形部分旋转为正
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(max_contour)
        angle = rect[2] if rect[1][0] > rect[1][1] else rect[2] - 90

        M = cv2.getRotationMatrix2D(rect[0], angle, 1)
        image = cv2.warpAffine(image, M, (1280, 960))

        # 截取矩形部分
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)

        w, h = max(rect[1][1], rect[1][0]), min(rect[1][1], rect[1][0])
        x, y = int(rect[0][0] - w / 2), int(rect[0][1] - h / 2)
        image = image[y:y + int(h), x:x + int(w)]

        # 开运算
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

        return image
```

#### 训练过程

训练的代码就不过多介绍了，这里直接贴代码，需要注意的是使用了random_split函数来划分训练集和验证集，我是使用cpu进行训练的，因为该数据集很小且mobilenet_v3_small网络非常的轻量化，训练所需要的时间可以接受。

```py
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
```

#### 训练结果

我这里自己电脑上训练了20代，验证集准确率达到了99%。当然可以使用torch自带的mobilenet_v3_small模型，在采用预训练的条件下可以到达1，有兴趣可以自己尝试一下。

## 结语

这篇博客的重点主要集中在数据集的处理和mobilenet_v3_small的搭建上，针对于芯片数据集识别是一个很好的学习小项目。现在卷积神经网络已经能够学习到很多的特征，有非常强大的学习能力，但有时我们却忽略了最重要的东西即数据本身。数据的预处理的好坏能够很大程度上提高网络的收敛速度和识别能力，也是深度学习过程中重要的一环。
