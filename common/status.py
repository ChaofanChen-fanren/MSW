# normal status 
status_normal = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]


# abnormal status
from common.data_anomaly_detail_gpt import (
    mvtec_anomaly_detail_gpt,
    visa_anomaly_detail_gpt
)

status_abnormal_winclip = [
    "damaged {}",
    "broken {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

status_abnormal = {}

for cls_name in mvtec_anomaly_detail_gpt.keys():
    status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in mvtec_anomaly_detail_gpt[cls_name]] + status_abnormal_winclip

for cls_name in visa_anomaly_detail_gpt.keys():
    status_abnormal[cls_name]= ['abnormal {} ' + 'with {}'.format(x) for x in visa_anomaly_detail_gpt[cls_name]] + status_abnormal_winclip 


if __name__ == '__main__':
    print(status_normal)
    print(status_abnormal)