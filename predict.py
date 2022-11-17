import torch
import torch.nn.functional as F

def predict(model, val_loader, device, checkpoint):
	model.load_state_dict(torch.load(checkpoint))
	val_loss = 0.0
	# 评估
	model.eval()
	num_correct = 0
	num_examples = 0

	n_classes = 8
	# nums = [0] * batch_size
	# pre = [0] * batch_size
	# ever_acc = [0] * batch_size

	target_num = torch.zeros((1, n_classes))  # n_classes为分类任务类别数量
	predict_num = torch.zeros((1, n_classes))
	acc_num = torch.zeros((1, n_classes))

	# 实例化混淆矩阵，这里NUM_CLASSES = 8
	class_indict = {'0':'early_1', '1':'early_2', '2':'early_3', '3':'early_4', '4':'late_1', '5':'late_2', '6':'late_3', '7':'late_4'}
	label = [label for _, label in class_indict.items()]
	confusion = ConfusionMatrix(num_classes=n_classes, labels=label)

	for batch in val_loader:
		inputs, targets = batch
		inputs = inputs.to(device)
		targets = targets.to(device)
		# outputs, repr, plt = model(inputs)
		outputs, repr = model(inputs)
		# loss = loss_function(outputs, targets)
		# con_loss = con_loss_func(repr, targets)
		# if con_loss != 0:
		# 	loss += con_loss
		# val_loss += loss.data.item()

		# 计算每一类的准确率
		# a =  targets
		# b = torch.max(F.softmax(outputs, dim=1), dim=1)[1]
		# for i in a:
		# 	nums[i] += 1
		# for i in range(len(pre)):
		# 	if b[i] == a[i]:
		# 		pre[b[i]] += 1

		# print(torch.max(F.softmax(outputs, dim=1), dim=1)[1])
		# print(targets)

		predicted = torch.max(F.softmax(outputs, dim=1), dim=1)[1]

		# 4个指标计算
		pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
		predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
		tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
		target_num += tar_mask.sum(0)  # 得到数据中每类的数量
		acc_mask = pre_mask * tar_mask
		acc_num += acc_mask.sum(0)  # 得到各类别分类正确的样本数量

		# confusion_matrix
		confusion.update(predicted.cpu().numpy(), targets.cpu().numpy())



		correct = torch.eq(torch.max(F.softmax(outputs, dim=1), dim=1)[1], targets)
		num_correct += torch.sum(correct).item()
		num_examples += correct.shape[0]

	# val_loss /= len(val_loader.dataset)
	acc = num_correct / num_examples

	# print('Epoch: {}, train_loss: {:.2f}, val_loss: {:.2f}, acc: {:.2f}'.format(epoch, train_loss, val_loss, acc))
	print('acc: {:.4f}'.format(acc))


	# for i in range(len(pre)):
	# 	ever_acc[i] = pre[i] / nums[i]
	# print(nums)
	# print(ever_acc)

	recall = acc_num / target_num
	precision = acc_num / predict_num
	F1 = 2 * recall * precision / (recall + precision)
	accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
	print('Test Acc {}, recal {}, precision {}, F1-score {}'.format(accuracy, recall, precision, F1))

	# 绘制混淆矩阵
	confusion.plot()
	confusion.summary()

import matplotlib.pyplot as plt
import numpy as np
import prettytable
class ConfusionMatrix(object):

	def __init__(self, num_classes: int, labels: list):
		self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
		self.num_classes = num_classes  # 类别数量，本例数据集类别为5
		self.labels = labels  # 类别标签

	def update(self, preds, labels):
		for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
			self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

	def summary(self):  # 计算指标函数
		# calculate accuracy
		sum_TP = 0
		n = np.sum(self.matrix)
		for i in range(self.num_classes):
			sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
		acc = sum_TP / n  # 总体准确率
		print("the model accuracy is ", acc)

		# kappa
		sum_po = 0
		sum_pe = 0
		for i in range(len(self.matrix[0])):
			sum_po += self.matrix[i][i]
			row = np.sum(self.matrix[i, :])
			col = np.sum(self.matrix[:, i])
			sum_pe += row * col
		po = sum_po / n
		pe = sum_pe / (n * n)
		# print(po, pe)
		kappa = round((po - pe) / (1 - pe), 3)
		# print("the model kappa is ", kappa)

		# precision, recall, specificity
		table = prettytable.PrettyTable()  # 创建一个表格
		table.field_names = ["", "Precision", "Recall", "Specificity"]
		for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
			TP = self.matrix[i, i]
			FP = np.sum(self.matrix[i, :]) - TP
			FN = np.sum(self.matrix[:, i]) - TP
			TN = np.sum(self.matrix) - TP - FP - FN

			Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
			Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
			Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

			table.add_row([self.labels[i], Precision, Recall, Specificity])
		print(table)
		return str(acc)[:6]

	def plot(self):  # 绘制混淆矩阵
		matrix = self.matrix
		print(matrix)
		plt.imshow(matrix, cmap=plt.cm.Blues)

		# 设置x轴坐标label
		plt.xticks(range(self.num_classes), self.labels, rotation=45)
		# 设置y轴坐标label
		plt.yticks(range(self.num_classes), self.labels)
		# 显示colorbar
		plt.colorbar()
		plt.xlabel('True Labels')
		plt.ylabel('Predicted Labels')
		plt.title('Confusion matrix (acc=' + self.summary() + ')')

		# 在图中标注数量/概率信息
		thresh = matrix.max() / 2
		for x in range(self.num_classes):
			for y in range(self.num_classes):
				# 注意这里的matrix[y, x]不是matrix[x, y]
				info = int(matrix[y, x])
				plt.text(x, y, info,
						 verticalalignment='center',
						 horizontalalignment='center',
						 color="white" if info > thresh else "black")
		plt.tight_layout()
		plt.show()

