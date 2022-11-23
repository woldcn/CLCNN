import torch
import torch.nn.functional as F
import shutil
import os

def train(model, loss_function, optimizer, train_loader, val_loader, epochs, device, useCL, T, log):
	best_acc = 0
	best_epoch = 0
	# 迭代
	for epoch in range(epochs):
		train_loss = 0.0
		val_loss = 0.0

		# 训练
		model.train()
		num_correct = 0
		num_examples = 0
		for batch in train_loader:
			optimizer.zero_grad()
			inputs, targets = batch
			inputs = inputs.to(device)
			targets = targets.to(device)
			# outputs, repr, plt = model(inputs)
			outputs, repr = model(inputs)
			loss = loss_function(outputs, repr, targets, useCL, T)
			loss.backward()
			optimizer.step()
			train_loss += loss.data.item()
			correct = torch.eq(torch.max(F.softmax(outputs, dim=1), dim=1)[1], targets)
			num_correct += torch.sum(correct).item()
			num_examples += correct.shape[0]

		train_loss /= len(train_loader.dataset)
		train_acc = num_correct / num_examples

		# 评估
		model.eval()
		num_correct = 0
		num_examples = 0
		for batch in val_loader:
			inputs, targets = batch
			inputs = inputs.to(device)
			targets = targets.to(device)
			# outputs, repr, plt = model(inputs)
			outputs, repr = model(inputs)
			loss = loss_function(outputs, repr, targets, useCL, T)
			val_loss += loss.data.item()
			correct = torch.eq(torch.max(F.softmax(outputs, dim = 1), dim = 1)[1], targets)
			num_correct += torch.sum(correct).item()
			num_examples += correct.shape[0]

		val_loss /= len(val_loader.dataset)
		val_acc = num_correct / num_examples
		log.print('Epoch: {}, train_loss: {:.4f}, test_loss: {:.4f}, train_acc: {:.4f}, val_acc: {:.4f}'.format(epoch, train_loss, val_loss, train_acc, val_acc))

		if val_acc>best_acc:
			best_acc = val_acc
			best_epoch = epoch

			shutil.rmtree('./saved_model/CLCNN')
			os.mkdir('./saved_model/CLCNN')
			torch.save(model.state_dict(), './saved_model/CLCNN/best_epoch_' + str(epoch) + '_' + str(round(val_acc,4)) + '.pth')

	log.print('best val_acc: {:.4f} at epoch {}'.format(best_acc, best_epoch))
	log.save
