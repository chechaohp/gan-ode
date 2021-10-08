import torch
import torch.nn as nn
import numpy as np
from dataset import MNISTRotationVideo, MNISTRotationImage
from on_dev.mocogan import VideoDiscriminator, PatchImageDiscriminator
from on_dev.mocogan_ode import VideoGeneratorMNISTODE as VideoGeneratorMNIST
# from on_dev.evaluation_metrics import calculate_inception_score
from on_dev.ode_training import GANODETrainer
import functools
from tqdm import tqdm
from skvideo import io
import os

epochs = 100000
batch_size = 32
start_epoch = 0
path = 'mnist-rand'
d_iters = 2

checkpoint_path = '../drive/MyDrive/moco_ode/checkpoints/'+path
video_sample_path = '../drive/MyDrive/moco_ode/video_samples/'+path

if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

if not os.path.exists(video_sample_path):
    os.mkdir(video_sample_path)


path_to_mnist_rot = '../drive/MyDrive/MNIST/rot-mnist_rand.mat'


def genSamples(g, n=8, e=1, size = 64):
    g.eval()
    with torch.no_grad():
        s = g.sample_videos(n**2)[0].cpu().detach().numpy()
    g.train()

    out = np.zeros((3, 16, size*n, size*n))

    for j in range(n):
        for k in range(n):
            out[:, :, size*j:size*(j+1), size*k:size*(k+1)] = s[j*n + k, :, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (out + 1) / 2 * 255
    io.vwrite(
        f'{video_sample_path}/gensamples_id{e}.gif',
        out
    )

def discriminator_train(discriminator, generator, real, loss_fn, use_cuda = True):
    if use_cuda:
        real = real.cuda()
    predict_real, _ = discriminator(real)
    with torch.no_grad():
        fake, _ = generator.sample_images(batch_size)
    predict_fake, _ = discriminator(fake)
    pr_labels = torch.ones_like(predict_real)
    pf_labels = torch.zeros_like(predict_fake)
    dis_loss = loss_fn(predict_real, pr_labels) + loss_fn(predict_fake, pf_labels)
    # dis_img_loss_val = dis_img_loss.item()
    return dis_loss

def generator_train(image_discriminator, video_discriminator, generator, loss_fn):
    # generator
    fakeVid, _ = generator.sample_videos(batch_size)
    fakeImg, _ = generator.sample_images(batch_size)
    predict_fake_vid, _ = video_discriminator(fakeVid)
    predict_fake_img, _ = image_discriminator(fakeImg)
    pf_vid_labels = torch.ones_like(predict_fake_vid)
    pf_img_labels = torch.ones_like(predict_fake_img)
    gen_loss = loss_fn(predict_fake_vid, pf_vid_labels) + loss_fn(predict_fake_img, pf_img_labels)
    # gen_loss_val = gen_loss.item()
    return gen_loss


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
    disVid = VideoDiscriminator(n_channels,ksize=2)
    disImg = PatchImageDiscriminator(n_channels)
    gen = VideoGeneratorMNIST(n_channels, 50, 0, 16, 16, linear=False)

    if use_cuda:
        disVid.cuda()
        disImg.cuda()
        gen.cuda()

    # init optimizers and loss
    loss = nn.BCEWithLogitsLoss()

    # resume training
    resume = False
    start_epoch = 0
    if resume:
        state_dicts = torch.load(f'{checkpoint_path}/state_normal1000.ckpt')
        start_epoch = state_dicts['epoch'] + 1

        gen.load_state_dict(state_dicts['model_state_dict'][0])
        disVid.load_state_dict(state_dicts['model_state_dict'][1])
        disImg.load_state_dict(state_dicts['model_state_dict'][2])

    # train
    # isScores = []
    # if resume:
    #     isScores = list(np.load('epoch_is/mocogan_ode_inception.npy'))
    # else:
    #     isScores = []
    disImg_train = functools.partial(discriminator_train, discriminator=disImg, generator=gen, loss_fn=loss, use_cuda = use_cuda)
    disVid_train = functools.partial(discriminator_train, discriminator=disVid, generator=gen, loss_fn=loss, use_cuda = use_cuda)
    gen_train = functools.partial(generator_train, image_discriminator=disImg, video_discriminator=disVid, generator=gen, loss_fn=loss)
    trainer = GANODETrainer(gen.parameters(), disImg.parameters(), disVid.parameters(),
                            gen_train, 
                            disImg_train, 
                            disVid_train,
                            method='rk4')

    for epoch in tqdm(range(start_epoch, epochs)):
        for i in range(d_iters):
            # image discriminator
            real, _ = next(imgGen)
            dis_img_loss = trainer.step(real, model='dis_img')
            # video discriminator

            real,_ = next(vidGen)
            dis_vid_loss = trainer.step(real, model='dis_vid')

        # generator
        gen_loss = trainer.step(model='gen')

        if epoch % 100 == 0:
            print('Epoch', epoch, 'DisImg', dis_img_loss.item(), 'DisVid', dis_vid_loss.item(), 'Gen', gen_loss.item())
        if epoch % 1000 == 0:
            genSamples(gen, e=epoch,size=28)
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
                                                disImg.state_dict()]},
                        f'{checkpoint_path}/state_normal{epoch}.ckpt')
    torch.save({'epoch': epoch,
                'model_state_dict': [gen.state_dict(),
                                     disVid.state_dict(),
                                     disImg.state_dict()]},
               f'{checkpoint_path}/state_normal{epoch}.ckpt')
    # isScores.append(calculate_inception_score(gen, test=False,
    #                                           moco=True))
    # np.save('epcoh_is/mocogan_ode_inception.npy', isScores)
    # print(isScores[-1])



if __name__ == '__main__':
    train()