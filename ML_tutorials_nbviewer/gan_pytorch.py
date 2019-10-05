import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
#Parameters

batchSize = 64
imageSize = 64

path = os.getcwd() 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)
#Image transformations
transform = transforms.Compose([transforms.Scale(imageSize),
transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),])

# Loading the dataset
#dataset = dset.CIFAR10(root=path, download = True, transform = transform)
dataset = dset.ImageFolder(root="/home/rubencr/GAN/faces/lfw", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers = 2)


def weights_init(m):

	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find("BatchNorm") != -1:
		m.weigth.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)



class DiscriminatorNet(torch.nn.Module):

	def __init__(self):
		super(DiscriminatorNet, self).__init__()
		self.model = nn.Sequential(

			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias = False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias = False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias = False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias = False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias = False),
			nn.Sigmoid()

			)



	def forward(self, input):

		output = self.model(input)
		return output.view(-1)




class GeneratorNet(torch.nn.Module):

	def __init__(self):
		super(GeneratorNet, self).__init__()


		self.model = nn.Sequential(

			nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias = False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
			nn.Tanh()

			)



	def forward(self, input):

		output = self.model(input)
		return output


discriminator = DiscriminatorNet()
discriminator.cuda()
generator = GeneratorNet()
generator.cuda()
criterion = nn.BCELoss()
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))



for epoch in range(400):

	for i, data in enumerate(dataloader,0):

		#Now we train the discrimnator which are the real data and the classifier label for
		#real data is one, this is why the target is built using a torch.ones which builds a matrix
		#in which all the values equals one, and of course the size of the squared-matrix is the input size!

		discriminator.zero_grad() #Gradients set to zero!
		real, _ = data
		inp = Variable(real)
		target = Variable(torch.ones(inp.size()[0]))
		output = discriminator(inp.cuda())
		errD_real = criterion(output.cuda(), target.cuda())


		# NOw we train the discriminator with the fake data which labels are 0 and thus, we use the torch.zeros function!

		noise = Variable(torch.randn(inp.size()[0], 100, 1, 1))
		fake = generator(noise.cuda())
		target = Variable(torch.zeros(inp.size()[0]))
		output = discriminator(fake.detach())
		errD_fake = criterion(output.cuda(), target.cuda())

		errD = errD_real + errD_fake
		errD.backward()
		optimizer_d.step()


		#NOW THE GENERATOR IS TRAINED!!

		generator.zero_grad()
		target = Variable(torch.ones(inp.size()[0]))
		output = discriminator(fake.cuda())
		errG = criterion(output.cuda(), target.cuda())
		errG.backward()
		optimizer_g.step()

		#print("loss_g {} loss_d {}".format(errG.data, errD.data))
		print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 50, i, len(dataloader), errD.data, errG.data))
                if i % 100 == 0:
                    vutils.save_image(real, '%s/real_samples.png' % path, normalize = True)
                    fake = generator(noise.cuda())
                    vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (path, epoch), normalize = True)
