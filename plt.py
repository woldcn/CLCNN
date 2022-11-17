import numpy
import torch
import numpy as np
import pandas as pd

def plt(model, plt_loader, checkpoint, device):
	model.load_state_dict(torch.load(checkpoint))
	model.to(device)
	result = numpy.empty((1,3))
	for batch in plt_loader:
		inputs, targets = batch
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs, repr, plt = model(inputs)



		plt = plt.cpu()
		targets = targets.cpu()
		plt = plt.detach().numpy()
		targets = targets.detach().numpy()
		# for i in range(len(plt)):
		# 	print(str(plt[i][0]) + '\t' + str(plt[i][1]))

		# print(plt)
		# print(targets)

		data = np.insert(plt, 0, targets, axis=1)

		result = np.concatenate((result, data), axis=0)


	data = pd.DataFrame(result)
	writer = pd.ExcelWriter('plot.xlsx')		# 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
