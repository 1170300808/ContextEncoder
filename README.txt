实验有两个部分，一个是实现Context Encoders: Feature Learning by Inpainting，包括的程序文件有Net中的CoderGan、Discriminator类，以及Train，运行Train即可训练
另一个是实现了High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis
包含的程序文件有Net中的ContentNet类，TextureTrain，TextureNet, ContentTrain, StyleLoss, ContentLoss, TvLoss, Run, 运行Run即可，其中ContenTrain 和 ContentNet为第一个部分的Context Encoders修改leakyRelu为ELU后的网络
最终结果在result的根目录下
