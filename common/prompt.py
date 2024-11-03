import re

from .prompt_data import object_list, anomaly_detail_gpt
from .prompt_data import status_normal, status_abnormal_winclip
from .prompt_data import positions_list


class PromptTemplate:
    def __init__(self,
                 dataset: str = 'visa',
                 abnormal: bool = False,
                 position: bool = False
                 ):
        self.oringal_class_names = object_list[dataset]
        self.is_abnormal = abnormal
        self.is_position = position

        self.cls_map = {cls_name: re.sub(r'\d+', '', cls_name) for cls_name in self.oringal_class_names}  # pcb1 -> pcb
        self.anomaly_detail_gpt = anomaly_detail_gpt[dataset]
        self.status = {}
        if abnormal:
            for class_name in self.anomaly_detail_gpt.keys():
                self.status[class_name] = ['abnormal {} ' + 'with {}'.format(x) for x in
                                           self.anomaly_detail_gpt[class_name].split(',')] + status_abnormal_winclip
        else:
            self.status = status_normal
        self.positions = positions_list
        self.prompt_template = self.gen_prompt()

    def gen_prompt(self):
        prompt_template = {}
        for class_name in self.oringal_class_names:
            cls_name = self.cls_map[class_name]

            # FIXME: 对于异常描述不加位置，不会出现.结尾的情况，没有修改
            if self.is_abnormal:
                # [w_1][w_2]...[w_{n_ctx}][STATE][CLASS] with [ANOMALY CLASS]
                p = [
                    status_i.format(class_name)
                    for status_i in self.status[class_name]
                ]
            else:
                # [v_1][v_2]...[v_{n_ctx}][STATE][CLASS]
                p = [status_i.format(cls_name) + "." for status_i in self.status]
            if self.is_position:
                p = [
                    p_i + " at " + position + "."
                    for p_i in p
                    for position in self.positions
                ]

            prompt_template[cls_name] = p

        return prompt_template

    def get_prompt(self):
        return self.prompt_template
