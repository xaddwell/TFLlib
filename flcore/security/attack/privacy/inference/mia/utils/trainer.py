import torch
from torch.optim import lr_scheduler

'''
code from: DPSUR: Accelerating Differentially Private Stochastic Gradient Descent Using Selective Update and Release
'''
class shadow():
    def __init__(self, trainloader, testloader, model, optimizer, criterion, epoch, device):
        self.epoch = epoch
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)


    # Training
    def train(self):
        self.model.train()

        train_loss = 0
        correct = 0
        total = 0
        for i in range(self.epoch):
            print("<======================= Epoch " + str(i+1) + " =======================>")
            for batch in self.trainloader:
                self.optimizer.zero_grad()
                if(len(batch) == 3):
                    inputs, targets, members = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                else:
                    inputs, masks, targets, members = batch
                    inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs, masks)

                loss = self.criterion(outputs, targets.squeeze())
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            self.scheduler.step()

            print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/((i+1)*len(self.trainloader))))
            self.test()

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.testloader:
                self.optimizer.zero_grad()
                if(len(batch) == 3):
                    inputs, targets, members = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                else:
                    inputs, masks, targets, members = batch
                    inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs, masks)

                loss = self.criterion(outputs, targets.squeeze())

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total
