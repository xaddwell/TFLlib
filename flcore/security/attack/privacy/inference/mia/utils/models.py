import torch
import torch.nn as nn
import torch.nn.functional as F

class attrinf_attack_model(nn.Module):
    def __init__(self, inputs, outputs):
        super(attrinf_attack_model, self).__init__()
        self.classifier = nn.Linear(inputs, outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CNN_MIA(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN_MIA, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Tanh(),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Conv2d(64, 128, kernel_size=3),
            #nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*6*6, 64),
            nn.Tanh(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ShadowAttackModel(nn.Module):
	def __init__(self, class_num, n_in=None):
		super(ShadowAttackModel, self).__init__()
		self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			# print("---class_num",len(class_num))
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(128, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, prediction):
		# print("Output_Component_result = self.Output_Component(output)")
		Output_Component_result = self.Output_Component(output)
		Prediction_Component_result = self.Prediction_Component(prediction.reshape(-1,1))
		# print("Prediction_Component_result: ", Prediction_Component_result.shape)
		# print("***********")
		final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result

class PartialAttackModel(nn.Module):
	def __init__(self, class_num):
		super(PartialAttackModel, self).__init__()
		self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(128, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, prediction):
		Output_Component_result = self.Output_Component(output)
		Prediction_Component_result = self.Prediction_Component(prediction)
		
		final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result


class WhiteBoxAttackModel(nn.Module):
	def __init__(self, class_num, total):
		super(WhiteBoxAttackModel, self).__init__()
		self.Output_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Loss_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Gradient_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Conv2d(1, 1, kernel_size=5, padding=2),
			nn.BatchNorm2d(1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Flatten(),
			nn.Dropout(p=0.2),
			nn.Linear(total, 256),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Label_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, loss, gradient, label):
		Output_Component_result = self.Output_Component(output)
		Loss_Component_result = self.Loss_Component(loss)
		Gradient_Component_result = self.Gradient_Component(gradient)
		Label_Component_result = self.Label_Component(label)

		# Loss_Component_result = F.softmax(Loss_Component_result, dim=1)
		# Gradient_Component_result = F.softmax(Gradient_Component_result, dim=1)

		# final_inputs = Output_Component_result
		# final_inputs = Loss_Component_result
		# final_inputs = Gradient_Component_result
		# final_inputs = Label_Component_result
		
		final_inputs = torch.cat((Output_Component_result, Loss_Component_result, Gradient_Component_result, Label_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result
	
class HZ_WhiteBoxAttackModel_1(nn.Module):
    # for grads
    def __init__(self,num_classes,num_layers=1):
        self.num_layers=num_layers
        self.num_classes=num_classes
        super(HZ_WhiteBoxAttackModel_1, self).__init__()
        
        self.grads_conv=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1,100,kernel_size=(1,100),stride=1),
            nn.ReLU(),
            
            )
        self.grads_linear = nn.Sequential(
        
            nn.Dropout(p=0.2),
            nn.Linear(256*100,2024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
        )
        
        
        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.preds=nn.Sequential(
           nn.Linear(num_classes,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.preds2=nn.Sequential(
           nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.preds3=nn.Sequential(
           nn.Linear(1024,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.correct=nn.Sequential(
           nn.Linear(1,num_classes),
            nn.ReLU(),
            nn.Linear(num_classes,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*(6+num_layers),256),
            
            nn.ReLU(),
            nn.Linear(256,128),
            
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            print (key)
            if key.split('.')[-1] == 'weight':    
                nn.init.normal(self.state_dict()[key], std=0.01)
                print (key)
                
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
        
    def forward(self,g,l,c,o,l1,l2):
        print(g.shape)
        out_g = self.grads_conv(g)
        print("After grads_conv:", out_g.shape)
        out_g = out_g.view([g.size()[0],-1])
        print(out_g.shape)
        out_g = self.grads_linear(out_g)
        print(out_g.shape)
        out_l = self.labels(l)
        print(out_l.shape)
        out_c = self.correct(c)
        print(out_c.shape)
        out_o = self.preds(o)
        print(out_o.shape)
        #out_g1 = self.preds2(l1)
        #out_g2 = self.preds3(l2)
        
        _outs= torch.cat((out_g,out_c,out_l),1)
        
        if self.num_layers>0:
            _outs= torch.cat((_outs,out_o),1)
#         if self.num_layers>1:
#             _outs= torch.cat((_outs,out_l1),1)
            
#         if self.num_layers>2:
#             _outs= torch.cat((_outs,out_l2),1)
    
        is_member =self.combine(_outs )
        
        
        return self.output(is_member)
    
class HZ_WhiteBoxAttackModel_2(nn.Module):
	# for outputs
    def __init__(self,num_classes,num_layers=1):
        self.num_layers=num_layers
        self.num_classes=num_classes
        super(HZ_WhiteBoxAttackModel_2, self).__init__()
        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.preds=nn.Sequential(
           nn.Linear(num_classes,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.preds2=nn.Sequential(
           nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.preds3=nn.Sequential(
           nn.Linear(1024,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.correct=nn.Sequential(
           nn.Linear(1,num_classes),
            nn.ReLU(),
            nn.Linear(num_classes,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*(2+num_layers),256),
            
            nn.ReLU(),
            nn.Linear(256,128),
            
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            print (key)
            if key.split('.')[-1] == 'weight':    
                nn.init.normal(self.state_dict()[key], std=0.01)
                print (key)
                
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
    def forward(self,l,c,o,l1,l2):
        
        out_l = self.labels(l)
        out_c = self.correct(c)
        out_o = self.preds(o)
        out_l1 = self.preds2(l1)
        out_l2 = self.preds3(l2)
        
        _outs= torch.cat((out_c,out_l),1)
        
        if self.num_layers>0:
            _outs= torch.cat((_outs,out_o),1)
        if self.num_layers>1:
            _outs= torch.cat((_outs,out_l1),1)
            
        if self.num_layers>2:
            _outs= torch.cat((_outs,out_l2),1)
    
        is_member =self.combine(_outs )
        
        
        return self.output(is_member)

class HZ_WhiteBoxAttackModel_3(nn.Module):
    def __init__(self,num_classes,num_feds):
        self.num_classes=num_classes
        self.num_feds=num_feds
        super(HZ_WhiteBoxAttackModel_3, self).__init__()
        self.grads_conv=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1,1000,kernel_size=(1,100),stride=1),
            nn.ReLU(),
            
            )
        self.grads_linear = nn.Sequential(
        
            nn.Dropout(p=0.2),
            nn.Linear(256*1000,1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
        )
        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.preds=nn.Sequential(
           nn.Linear(num_classes,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.correct=nn.Sequential(
           nn.Linear(1,num_classes),
            nn.ReLU(),
            nn.Linear(num_classes,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*4*self.num_feds,256),
            
            nn.ReLU(),
            nn.Linear(256,128),
            
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            print (key)
            if key.split('.')[-1] == 'weight':    
                nn.init.normal(self.state_dict()[key], std=0.01)
                print (key)
                
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
    def forward(self,gs,ls,cs,os):
        
        
        for i in range(self.num_feds):
            out_g = self.grads_conv(gs[i]).view([gs[i].size()[0],-1])
            out_g = self.grads_linear(out_g)
            out_l = self.labels(ls[i])
            out_c = self.correct(cs[i])
            out_o = self.preds(os[i])
            if i==0:
                com_inp = torch.cat((out_g,out_c,out_o),1)
            else:
                com_inp= torch.cat((out_g,out_c,out_o,com_inp),1)
                    
        is_member =self.combine( com_inp)
        
        
        return self.output(is_member)

class ML_CNN(nn.Module):
    def __init__(self, n_out, n_in, n_hidden = 50):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding='same')
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(32, n_hidden)  
        self.output = nn.Linear(n_hidden, n_out)

    def forward(self, x, no_use):
        x = torch.unsqueeze(x, 0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc(x))
        x = self.output(x)
        return F.softmax(x, dim=1)

class ML_NN(nn.Module):
    def __init__(self, n_out, n_in, n_hidden = 50):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.output = nn.Linear(n_hidden, n_out)

    def forward(self, x, no_use):
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return F.softmax(x, dim=0)
    
class ML_Softmax(nn.Module):
    def __init__(self, n_out, n_in):
        super().__init__()
        self.output = nn.Linear(n_in, n_out)

    def forward(self, x, no_use):
        return F.softmax(self.output(x), dim=0)
