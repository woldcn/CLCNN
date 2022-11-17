import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_func(outputs, repr, label, useCL, T):
	loss_f = nn.CrossEntropyLoss().cuda()
	loss =loss_f(outputs, label)
	if useCL==False:
		return loss
	# 对比损失
	# 将标签转为 1 维, 再转 int
	con_label = label.view(-1).int()
	label_lists = con_label.tolist()
	entity_count_list = [0] * 9  # 类别数    #########################################
	for i in label_lists:
		entity_count_list[i] += 1
	# 统计数量最多的类型
	most_entity_count = -1
	most_entity_type = -1
	for i, entity_num in enumerate(entity_count_list):
		if entity_num > most_entity_count and i != 0:
			most_entity_count = entity_num
			most_entity_type = i
	# 将类型最多的，作为正样例，其余作为负样例
	entity_is_pos = [0] * con_label.shape[0]
	for i, type in enumerate(con_label):
		if type == most_entity_type:
			entity_is_pos[i] = 1

	# 将表示转为 2 维
	# T = 0.5  # 温度参数 T
	entity_is_pos = torch.tensor(entity_is_pos).cuda()
	n = entity_is_pos.shape[0]
	repr = repr.view(-1, repr.shape[-1])
	similarity_matrix = F.cosine_similarity(repr.unsqueeze(1), repr.unsqueeze(0), dim=2)
	similarity_matrix = torch.where(torch.isnan(similarity_matrix), torch.full_like(similarity_matrix, 0),
									similarity_matrix)
	mask = torch.ones_like(similarity_matrix) * (entity_is_pos.expand(n, n).eq(entity_is_pos.expand(n, n).t()))
	mask_no_sim = torch.ones_like(mask) - mask
	mask_dui_jiao_0 = torch.ones(n, n) - torch.eye(n, n)
	mask_dui_jiao_0 = mask_dui_jiao_0.cuda()
	similarity_matrix = torch.exp(similarity_matrix / T)
	similarity_matrix = similarity_matrix * mask_dui_jiao_0
	sim = mask * similarity_matrix
	no_sim = similarity_matrix - sim
	no_sim_sum = torch.sum(no_sim, dim=1)
	no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
	sim_sum = sim + no_sim_sum_expend
	con_loss = torch.div(sim, sim_sum)
	con_loss = mask_no_sim.cuda() + con_loss + torch.eye(n, n).cuda()
	con_loss = -torch.log(con_loss)
	# con_loss = con_loss.sigmoid()
	con_loss = torch.sum(torch.sum(con_loss, dim=1)) / (n * n) / 10  #10
	con_loss = torch.where(torch.isnan(con_loss), torch.full_like(con_loss, 0), con_loss)
	if con_loss!=0:
		loss = loss + con_loss
	return loss