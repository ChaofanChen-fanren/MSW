import re
# mvtec dataset object list
mvtec_obj_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
# visa dataset object list
visa_obj_list = [
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "capsules",
]

# all object list aggregation
cls_map = {}

for cls_name in mvtec_obj_list:
    cls_map[cls_name] = cls_name
    
for cls_name in visa_obj_list:
    cls_map[cls_name] = cls_name

# Remove the numbers from the object names with numeric suffixes  eg "pcb1 -> pcb"
for key, value in cls_map.items():
    value = re.sub(r'\d+', '', value)  # remove digits
    cls_map[key] = value    




if __name__ == "__main__":
    print(len(cls_map))
    print(cls_map)