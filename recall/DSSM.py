import torch
import torch.nn as nn



import torch
import torch.nn as nn


# class DynamicAttributeClass:
#     def __init__(self, attribute_name, value):
#         setattr(self, attribute_name, value)

# # 使用示例
# obj = DynamicAttributeClass("dynamic_attr", 42)
# print(obj.dynamic_attr)  # 输出: 42


# class DSSM(nn.Module):
#     def __init__(self, user_fea, item_dim):
#         super(DSSM, self).__init__()
#         self.user_tower = nn.Sequential(
#             nn.Linear(user_fea, 128),
#             nn.ReLU(),
#             nn.Linear(128, item_dim)
#         )
#         self.item_tower = nn.Sequential(
#             nn.Linear(item_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, item_dim)
#         )

#     def get_attribute(self, attr_name):
#         if hasattr(self, attr_name):
#             return getattr(self, attr_name)
#         else:
#             raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")

# # 使用示例
# dssm = DSSM(user_fea=256, item_dim=128)
# print(dssm.get_attribute("user_tower"))  # 输出: user_tower的值



class DSSM(nn.Module):
    def __init__(self, user_fea, item_dim):
        self.user_tower = nn.Sequential(
            nn.Linear()
        )