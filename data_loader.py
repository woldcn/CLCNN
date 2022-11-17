from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data(train_path, val_path, batch_size):
	# 图片转换
	trans = transforms.Compose([
		# 转换图片大小
		transforms.Resize([650, 650]),
		# 转为Tensor格式
		transforms.ToTensor(),
	])

	train_data = datasets.ImageFolder(train_path, transform = trans)
	val_data = datasets.ImageFolder(val_path, transform = trans)

	train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last=True)
	val_loader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle = False, drop_last=True)

	return train_loader, val_loader, len(train_data), len(val_data)

