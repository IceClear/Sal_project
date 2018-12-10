from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm
import torch.autograd as autograd
from torch.autograd import Variable
from logger import Logger

from data_loader import DataLoader
from discriminator import Discriminator
from generator import Generator
from utils import *

if_use_wgan_gp = False
if_vis = True

logger = Logger('./logs')
batch_size = 40
lr = 0.0003

discriminator = Discriminator()
generator = Generator()
one = torch.FloatTensor([1])
mone = one * -1
LAMBDA = 10

try:
    discriminator.load_state_dict(torch.load("./discriminator.pkl"))
    generator.load_state_dict(torch.load("./generator.pkl"))
    print('Load learner previous point: Successed')
except Exception as e:
    print('Load learner previous point: Failed')

if if_vis:
    TMUX = 'TMUX 0'
    port = 8097
    from visdom import Visdom
    viz = Visdom(port=port)
    win = None
    win_dic = {}
    recorder = {
        'plot':{},
        'line':{},
        'scatter':{},
        'image':{},
        'text':{},
        'fixations':{},
    }
    def record(name,value,data_type='plot'):
        if data_type in ['plot','scatter']:
            try:
                # try expend
                recorder[data_type][name] += [value]
            except Exception as e:
                # else, initialize
                recorder[data_type][name] = [value]
        else:
            recorder[data_type][name] = value

    def log_visdom():
        '''push everything to the visdom server'''

        # plot lines
        for plot_name in recorder['plot'].keys():
            if plot_name in win_dic.keys():
                if len(recorder['plot'][plot_name]) > 0:
                    win_dic[plot_name] = viz.line(
                        torch.from_numpy(np.asarray(recorder['plot'][plot_name])),
                        win=win_dic[plot_name],
                        opts=dict(title=TMUX+'\n'+plot_name)
                    )
            else:
                win_dic[plot_name] = None

        for plot_name in recorder['scatter'].keys():
            if plot_name in win_dic.keys():
                if len(recorder['scatter'][plot_name]) > 0:
                    win_dic[plot_name] = viz.scatter(
                        torch.from_numpy(
                            np.asarray(recorder['scatter'][plot_name])
                        ),
                        win=win_dic[plot_name],
                        opts=dict(title=TMUX+'\n'+plot_name)
                    )
            else:
                win_dic[plot_name] = None

        for plot_name in recorder['fixations'].keys():
            if plot_name in win_dic.keys():
                if len(recorder['fixations'][plot_name]) > 0:
                    win_dic[plot_name] = viz.scatter(
                        torch.from_numpy(
                            np.asarray(recorder['fixations'][plot_name])
                        ),
                        win=win_dic[plot_name],
                        opts=dict(title=TMUX+'\n'+plot_name)
                    )
            else:
                win_dic[plot_name] = None

        # log images
        for images_name in recorder['image'].keys():
            if images_name in win_dic.keys():
                win_dic[images_name] = viz.images(
                    recorder['image'][images_name],
                    win=win_dic[images_name],
                    opts=dict(title=TMUX+'\n'+images_name)
                )
            else:
                win_dic[images_name] = None

        # log text
        for text_name in recorder['text'].keys():
            if text_name in win_dic.keys():
                win_dic[text_name] = viz.text(
                    recorder['text'][text_name],
                    win=win_dic[text_name],
                    opts=dict(title=TMUX+'\n'+text_name)
                )
            else:
                win_dic[text_name] = None

'''record basic information'''
record(
    'basic info',
    (TMUX +'<br>'+'train'+'<br>'+'wgan-gp: '+ str(if_use_wgan_gp)),
    'text'
    )


if torch.cuda.is_available():
    use_cuda = True
    discriminator.cuda()
    generator.cuda()
    one = one.cuda()
    mone = mone.cuda()

loss_function = nn.BCELoss()

if if_use_wgan_gp:
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    g_optim = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
else:
    d_optim = torch.optim.Adagrad(discriminator.parameters(), lr=lr)
    g_optim = torch.optim.Adagrad(generator.parameters(), lr=lr)

num_epoch = 120
dataloader = DataLoader(batch_size)
num_batch = int(dataloader.num_batches)# length of data / batch_size
print(num_batch)

def to_variable(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.unsqueeze(1)
    alpha = alpha.unsqueeze(1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

counter = 0
start_time = time.time()
DIR_TO_SAVE = "./generator_output/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)
validation_sample = cv2.imread("COCO_val2014_000000143859.png")

for current_epoch in tqdm(range(1,num_epoch+1)):
    n_updates = 1

    d_cost_avg = 0
    g_cost_avg = 0
    for idx in range(num_batch):

        (batch_img, batch_map) = dataloader.get_batch()
        batch_img = to_variable(batch_img,requires_grad=False)
        batch_map = to_variable(batch_map,requires_grad=False)
        real_labels = to_variable(torch.FloatTensor(np.ones(batch_size, dtype = float)),requires_grad=False)
        fake_labels = to_variable(torch.FloatTensor(np.zeros(batch_size, dtype = float)),requires_grad=False)

        if n_updates % 2 == 1:
            #print('Training Discriminator...')
            #discriminator.zero_grad()
            d_optim.zero_grad()

            if if_use_wgan_gp:
                for p in discriminator.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                discriminator.zero_grad()
                inp_d = torch.cat((batch_img,batch_map),1)

                # train with real
                D_real = discriminator(inp_d)
                D_real = D_real.mean()
                D_real = D_real.unsqueeze(0)
                # print D_real
                D_real.backward(mone)

                with torch.no_grad():
                    fake_map = generator(batch_img)

                inp_d_fake = torch.cat((batch_img,fake_map),1)
                D_fake = discriminator(inp_d_fake)

                D_fake = D_fake.mean()
                D_fake = D_fake.unsqueeze(0)
                D_fake.backward(one)

                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(discriminator, inp_d.data, inp_d_fake.data)
                gradient_penalty.backward()

                d_loss = D_fake - D_real + gradient_penalty
                d_loss.register_hook(print)
                Wasserstein_D = D_real - D_fake
                real_score = Wasserstein_D
                d_cost_avg += d_loss.data[0]
                d_optim.step()

            else:
                inp_d = torch.cat((batch_img,batch_map),1)
                #print(inp_d.size())
                outputs = discriminator(inp_d).squeeze()


                d_real_loss = loss_function(outputs,real_labels)
                d_real_loss = loss_function(outputs,real_labels)
                #print('D_real_loss = ', d_real_loss.data[0])

                #print(outputs)
                real_score = outputs.data.mean()

    #            fake_map = generator(batch_img)
    #            inp_d = torch.cat((batch_img,fake_map),1)
    #            outputs = discriminator(inp_d)
    #            d_fake_loss = loss_function(outputs, fake_labels)
    #            print('D_fake_loss = ', d_fake_loss.data[0])
                d_loss = torch.sum(torch.log(outputs))
                d_cost_avg += d_loss.data[0]

                d_loss.backward()
                d_loss.register_hook(print)
                d_optim.step()

            info = {
                 'd_loss' : d_loss.data[0],
                 'real_score_mean/Wasserstein_D' : real_score,
            }
            # for tag,value in info.items():
            #     logger.scalar_summary(tag, value, counter)
        else:
            #print('Training Generator...')
            #generator.zero_grad()
            if if_use_wgan_gp:
                for p in discriminator.parameters():
                    p.requires_grad = False  # to avoid computation

            g_optim.zero_grad()

            if if_use_wgan_gp:
                fake_map = generator(batch_img)
                inp_d = torch.cat((batch_img,fake_map),1)

                fake_score = discriminator(inp_d)
                fake_score = fake_score.mean()
                fake_score = fake_score.unsqueeze(0)
                fake_score.backward(mone)
                g_loss = -fake_score
                g_cost_avg += g_loss.data[0]
                g_optim.step()
                # print(max(fake_map.data.))
                # print(s)

                if (idx+1)%100 == 0:
                    record(
                        name = 'img',
                        value = batch_img[0:1],
                        data_type = 'image',
                    )

                    record(
                        name = 'sal_map',
                        value = fake_map[0:1]*255,
                        data_type = 'image',
                    )

            else:
                fake_map = generator(batch_img)
                inp_d = torch.cat((batch_img,fake_map),1)
                outputs = discriminator(inp_d)
                fake_score = outputs.data.mean()

                g_gen_loss = loss_function(fake_map,batch_map)
                g_dis_loss = -torch.log(outputs)
                alpha = 0.05
                g_loss = torch.sum(g_dis_loss + alpha * g_gen_loss)

                g_cost_avg += g_loss.data[0]

                g_loss.backward()
                g_optim.step()

                if (idx+1)%100 == 0:
                    record(
                        name = 'img',
                        value = batch_img[0:1],
                        data_type = 'image',
                    )

                    record(
                        name = 'sal_map',
                        value = fake_map[0:1]*255,
                        data_type = 'image',
                    )


            info = {
                  'g_loss' : g_loss.data[0],
                  'fake_score_mean' : fake_score,
            }
            # for tag,value in info.items():
            #     logger.scalar_summary(tag, value, counter)

        n_updates += 1

        if (idx+1)%100 == 0:
            print("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %2.f, D(G(x)): %.2f, time: %4.4f"
		        % (current_epoch, num_epoch, idx+1, num_batch, d_loss.data[0], g_loss.data[0],
		        real_score, fake_score, time.time()-start_time))

            record(
                    name = 'loss_d',
                    value = d_loss.data.cpu().numpy(),
                    data_type = 'plot',
                )

            record(
                    name = 'loss_g',
                    value = g_loss.data.cpu().numpy(),
                    data_type = 'plot',
                )
            log_visdom()

        counter += 1
    d_cost_avg /= num_batch
    g_cost_avg /= num_batch

    # Save weights every 3 epoch
    if (current_epoch + 1) % 3 == 0:
        print('Epoch:', current_epoch, ' train_loss->', (d_cost_avg, g_cost_avg))
        torch.save(generator.state_dict(), './generator.pkl')
        torch.save(discriminator.state_dict(), './discriminator.pkl')
    # predict(generator, validation_sample, current_epoch, DIR_TO_SAVE)
torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')
print('Done')
