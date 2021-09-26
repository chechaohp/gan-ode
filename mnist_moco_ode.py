import torch
import torch.nn as nn
import numpy as np
from dataset import MNISTRotationVideo, MNISTRotationImage
from on_dev.mocogan import VideoDiscriminator, PatchImageDiscriminator
from on_dev.mocogan_ode import VideoGeneratorMNIST
# from on_dev.evaluation_metrics import calculate_inception_score
from tqdm import tqdm
from skvideo import io
from pathlib import Path

epochs = 100000
batch_size = 32
start_epoch = 0
path = '../drive/MyDrive/moco_ode/mnist'

Path('../drive/MyDrive/moco_ode/checkpoints/'+path).mkdir(parents=True, exist_ok=True)
Path('../drive/MyDrive/moco_ode/video_samples/'+path).mkdir(parents=True, exist_ok=True)

path_to_mnist_rot = '../drive/MyDrive/MNIST/rot-mnist.mat'


def genSamples(g, n=8, e=1):
    g.eval()
    with torch.no_grad():
        s = g.sample_videos(n**2)[0].cpu().detach().numpy()
    g.train()

    out = np.zeros((3, 16, 64*n, 64*n))

    for j in range(n):
        for k in range(n):
            out[:, :, 64*j:64*(j+1), 64*k:64*(k+1)] = s[j*n + k, :, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (out + 1) / 2 * 255
    io.vwrite(
        f'../drive/MyDrive/moco_ode/video_samples/gensamples_id{e}.gif',
        out
    )


def train():
    # data
    print("Read MNIST Rotation data")
    videoDataset = MNISTRotationVideo(path_to_mnist_rot)
    imgDataset = MNISTRotationImage(path_to_mnist_rot)
    videoLoader = torch.utils.data.DataLoader(videoDataset, batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)
    imgLoader = torch.utils.data.DataLoader(imgDataset, batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)

    use_cuda = torch.cuda.is_available()


    def dataGen(loader):
        while True:
            for d in loader:
                yield d

    vidGen = dataGen(videoLoader)
    imgGen = dataGen(imgLoader)
    # gen model
    # number of channel in dataset
    n_channels = 1
    disVid = VideoDiscriminator(n_channels)
    disImg = PatchImageDiscriminator(n_channels)
    gen = VideoGeneratorMNIST(n_channels, 50, 0, 16, 16)

    if use_cuda:
        disVid.cuda()
        disImg.cuda()
        gen.cuda()

    # init optimizers and loss
    disVidOpt = torch.optim.Adam(disVid.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    disImgOpt = torch.optim.Adam(disImg.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    genOpt = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    loss = nn.BCEWithLogitsLoss()

    # resume training
    resume = True
    start_epoch = 0
    if resume:
        state_dicts = torch.load(f'{path}/state_normal1000.ckpt')
        start_epoch = state_dicts['epoch'] + 1

        gen.load_state_dict(state_dicts['model_state_dict'][0])
        disVid.load_state_dict(state_dicts['model_state_dict'][1])
        disImg.load_state_dict(state_dicts['model_state_dict'][2])
        genOpt.load_state_dict(state_dicts['optimizer_state_dict'][0])
        disVidOpt.load_state_dict(state_dicts['optimizer_state_dict'][1])
        disImgOpt.load_state_dict(state_dicts['optimizer_state_dict'][2])

    # train
    # isScores = []
    # if resume:
    #     isScores = list(np.load('epoch_is/mocogan_ode_inception.npy'))
    # else:
    #     isScores = []
    d_iters = 2

    for epoch in tqdm(range(start_epoch, epochs)):
        for i in range(d_iters):
            # image discriminator
            disImgOpt.zero_grad()
            real, _ = next(imgGen)

            if use_cuda:
                real = real.cuda()

            pr, _ = disImg(real)
            with torch.no_grad():
                fake, _ = gen.sample_images(batch_size)
            pf, _ = disImg(fake)
            pr_labels = torch.ones_like(pr)
            pf_labels = torch.zeros_like(pf)
            dis_img_loss = loss(pr, pr_labels) + loss(pf, pf_labels)
            # dis_img_loss_val = dis_img_loss.item()
            dis_img_loss.backward()
            disImgOpt.step()

            # video discriminator
            disVidOpt.zero_grad()
            real,_ = next(vidGen)
            if use_cuda:
                real = real.cuda().transpose(1, 2)
            else:
                real = real.transpose(1, 2)

            pr, _ = disVid(real)
            with torch.no_grad():
                fake, _ = gen.sample_videos(batch_size)
            pf, _ = disVid(fake)
            pr_labels = torch.ones_like(pr)
            pf_labels = torch.zeros_like(pf)
            dis_vid_loss = loss(pr, pr_labels) + loss(pf, pf_labels)
            # dis_vid_loss_val = dis_vid_loss.item()
            dis_vid_loss.backward()
            disVidOpt.step()

        # generator
        genOpt.zero_grad()
        fakeVid, _ = gen.sample_videos(batch_size)
        fakeImg, _ = gen.sample_images(batch_size)
        pf_vid, _ = disVid(fakeVid)
        pf_img, _ = disImg(fakeImg)
        pf_vid_labels = torch.ones_like(pf_vid)
        pf_img_labels = torch.ones_like(pf_img)
        gen_loss = loss(pf_vid, pf_vid_labels) + loss(pf_img, pf_img_labels)
        # gen_loss_val = gen_loss.item()
        gen_loss.backward()
        genOpt.step()
        print('Epoch', epoch, 'DisImg', dis_img_loss.item(), 'DisVid', dis_vid_loss.item(), 'Gen', gen_loss.item())
        if epoch % 1000 == 0:
            genSamples(gen, e=epoch)
            if epoch % 1000 == 0:
                # gen.cpu()
                # isScores.append(calculate_inception_score(gen, test=False,
                #                                           moco=True))
                # print(isScores[-1])
                # np.save('epoch_is/mocogan_ode_inception.npy', isScores)
                gen.cuda()
                torch.save({'epoch': epoch,
                            'model_state_dict': [gen.state_dict(),
                                                disVid.state_dict(),
                                                disImg.state_dict()],
                            'optimizer_state_dict': [genOpt.state_dict(),
                                                    disVidOpt.state_dict(),
                                                    disImgOpt.state_dict()]},
                        f'{path}/state_normal{epoch}.ckpt')
    torch.save({'epoch': epoch,
                'model_state_dict': [gen.state_dict(),
                                     disVid.state_dict(),
                                     disImg.state_dict()],
                'optimizer_state_dict': [genOpt.state_dict(),
                                         disVidOpt.state_dict(),
                                         disImgOpt.state_dict()]},
               f'{path}/state_normal{epoch}.ckpt')
    # isScores.append(calculate_inception_score(gen, test=False,
    #                                           moco=True))
    # np.save('epcoh_is/mocogan_ode_inception.npy', isScores)
    # print(isScores[-1])



if __name__ == '__main__':
    train()