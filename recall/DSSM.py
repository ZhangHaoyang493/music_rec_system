import torch
import torch.nn as nn



import torch
import torch.nn as nn



class DSSM(nn.Module):
    """
    user_fea写法说明：（一个字典）
    key：特征名称（用户自定义）
    value：
    一个列表，包含若干个元素：
    + 0: 对于每个变量，标注这是一个embedding变量还是  数值型变量（numerical）
        list_embedding(一个列表，全部是embedding特征，例如用户的听歌历史)，list_numerical(一个列表，全部是numerical特征)
    + 1: 这个特征的长度，不是list的长度都是1，是的话长度为list的长度
    + 2: 如果是embedding，指出embedding表的数量，也就是这个特征有几个值 ｜ 如果是list_embedding，指出这个特征和哪个特征共用embedding
    + 3: 如果是embedding，那么指出embedding的维度
    """
    def __init__(self, user_fea: dict, item_fea: dict):
        self.user_dim = 0
        self.item_dim = 0

        self.user_fea_name = {}
        self.item_fea_name = {}
        
        fea_index = 0
        for fea_name, fea_info in user_fea.items():
            assert fea_info[0] in ['embedding', 'numerical', 'list_embedding', 'list_numerical'], \
                '特征类型必须是embedding或者numerical或者list_embedding或者list_numerical'

            

            if fea_info[0] == 'embedding':
                setattr(self, fea_name, nn.Embedding(fea_info[2], fea_info[3]))
                self.user_dim += fea_info[3] * fea_info[1]
                # 注册这个特征                      特征名       特征类型 特征长度  特征维度
                self.user_fea_name[fea_index] = (fea_name, fea_info[0], 1, fea_info[3])
            elif fea_info[0] == 'numerical':
                self.user_dim += 1
                # 注册这个特征                      特征名       特征类型 特征长度  特征维度
                self.user_fea_name[fea_index] = (fea_name, fea_info[0], 1, 1)
            elif fea_info[0] == 'list_embedding':
                setattr(self, fea_name, getattr(self, fea_info[1]))
                self.user_dim += getattr(self, fea_name).embedding_dim * fea_info[1]
                # 注册这个特征                      特征名       特征类型 特征长度  特征维度
                self.user_fea_name[fea_index] = (fea_name, fea_info[0], fea_info[1], getattr(self, fea_name).embedding_dim)
            elif fea_info[0] == 'list_numerical':
                self.user_dim += 1 * fea_info[1]
                # 注册这个特征                      特征名       特征类型 特征长度  特征维度
                self.user_fea_name[fea_index] = (fea_name, fea_info[0], fea_info[1], fea_info[1])
            fea_index += 1
        
        for fea_name, fea_info in item_fea.items():
            assert fea_info[0] in ['embedding', 'numerical', 'list_embedding', 'list_numerical'], \
                '特征类型必须是embedding或者numerical或者list_embedding或者list_numerical'

            if fea_info[0] == 'embedding':
                setattr(self, fea_name, nn.Embedding(fea_info[2], fea_info[3]))
                self.user_dim += fea_info[3] * fea_info[1]
                # 注册这个特征                      特征名       特征类型 特征长度  特征维度
                self.item_fea_name[fea_index] = (fea_name, fea_info[0], 1, fea_info[3])
            elif fea_info[0] == 'numerical':
                self.user_dim += 1
                # 注册这个特征                      特征名       特征类型 特征长度  特征维度
                self.item_fea_name[fea_index] = (fea_name, fea_info[0], 1, 1)
            elif fea_info[0] == 'list_embedding':
                setattr(self, fea_name, getattr(self, fea_info[1]))
                self.user_dim += getattr(self, fea_name).embedding_dim * fea_info[1]
                # 注册这个特征                      特征名       特征类型 特征长度  特征维度
                self.item_fea_name[fea_index] = (fea_name, fea_info[0], fea_info[1], getattr(self, fea_name).embedding_dim)
            elif fea_info[0] == 'list_numerical':
                self.user_dim += 1 * fea_info[1]
                # 注册这个特征                      特征名       特征类型 特征长度  特征维度
                self.item_fea_name[fea_index] = (fea_name, fea_info[0], fea_info[1], fea_info[1])
            fea_index += 1
            
        self.user_tower = nn.Sequential(
            nn.Linear(self.user_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

        self.item_tower = nn.Sequential(
            nn.Linear(self.user_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

    def _parse_data_to_feature(self, data, tower='user'):
        """
            根据self.user_fea_name或者self.item_fea_name里面存储的特征信息对输入的数据进行处理
        """
        assert tower in ['user', 'item'], 'tower必须是user或者item'
        fea_dict = self.user_fea_name if tower == 'user' else self.item_fea_name

        fea = []
        index = 0
        # 特征名       特征类型 特征长度  特征维度
        for k, v in fea_dict.items():
            fea_name, fea_kind, fea_len, fea_dim = v
            if fea_kind == 'embedding':
                fea.append(getattr(self, fea_name)(data[:, :, index:index+1]))
                index += 1
            elif fea_kind == 'numerical':
                fea.append(data[:, :, index:index+1])
                index += 1
            elif fea_kind == 'list_embedding':
                for i in range(fea_len):
                    fea.append(getattr(self, fea_name)(data[:, :, index:index+1]))
                    index += 1
            elif fea_kind == 'list_numerical':
                for i in range(fea_len):
                    fea.append(data[:, :, index:index+1])
                    index += 1
        
        return torch.concat(fea, dim=-1)


    # 每一个data的batch内，第一个为正样本，其他的为负样本，正样本和负样本的和未sample_num
    # data: {'user': b * sample_num * self.user_dim, 'item': b * sample_num * self.item_dim}
    def forward(self, data):
        user_data = data['user']
        item_data = data['item']

        user_feature = self._parse_data_to_feature(user_data, tower='user')
        item_feature = self._parse_data_to_feature(item_data, tower='item')

        user_feature = self.user_tower(user_feature)
        item_feature = self.item_tower(item_feature)

        dot_product = torch.sum(user_feature, item_feature, dim=-1, keepdim=True)  # bxsample_num*1
        return dot_product

