from common import visa_obj_list, anomaly_detail_gpt, status_abnormal_winclip, mvtec_obj_list
from common import PromptTemplate
import re
# abnormal visa
prompt = PromptTemplate(dataset='visa', abnormal=True, position=True)
prompt_template = prompt.prompt_template

cls_map = {cls_name: re.sub(r'\d+', '', cls_name) for cls_name in visa_obj_list}
detail = anomaly_detail_gpt['visa']
p_num = {}
for class_name in visa_obj_list:
    cls_name = cls_map[class_name]
    if cls_name not in p_num:
        p_num[cls_name] = 0
        len_status = len(detail[class_name].split(',')) + len(status_abnormal_winclip)
        p_num[cls_name] += len_status

for cls_name in prompt_template.keys():
    if len(prompt_template[cls_name]) == p_num[cls_name]*9:
        # print("True")
        pass
    else:
        print(cls_name)
# abnormal mvtec
prompt = PromptTemplate(dataset='mvtec', abnormal=True, position=True)
prompt_template = prompt.prompt_template

cls_map = {cls_name: re.sub(r'\d+', '', cls_name) for cls_name in mvtec_obj_list}
detail = anomaly_detail_gpt['mvtec']
p_num = {}
for class_name in mvtec_obj_list:
    cls_name = cls_map[class_name]
    if cls_name not in p_num:
        p_num[cls_name] = 0
        len_status = len(detail[class_name].split(',')) + len(status_abnormal_winclip)
        p_num[cls_name] += len_status

for cls_name in prompt_template.keys():
    if len(prompt_template[cls_name]) == p_num[cls_name]*9:
        # print("True")
        pass
    else:
        print(cls_name)
# normal visa
prompt = PromptTemplate(dataset='visa', abnormal=False, position=False)
prompt_template = prompt.prompt_template
# print(prompt_template)
cls_map = {cls_name: re.sub(r'\d+', '', cls_name) for cls_name in visa_obj_list}
if set(prompt_template.keys()) == set(cls_map.values()):
    print("True")
else:
    print(list(prompt_template.keys()))
    print(list(set(cls_map.values())))
    print("False")
for p in prompt_template.values():
    # print(p)
    print(len(p) == 7)
# normal mvtec
prompt = PromptTemplate(dataset='mvtec', abnormal=False, position=False)
prompt_template = prompt.prompt_template
# print(prompt_template)
cls_map = {cls_name: re.sub(r'\d+', '', cls_name) for cls_name in mvtec_obj_list}
if set(prompt_template.keys()) == set(cls_map.values()):
    print("True")
else:
    # print(list(prompt_template.keys()))
    # print(list(set(cls_map.values())))
    print("False")
for p in prompt_template.values():
    # print(p)
    print(len(p) == 7)