from GSA import grounded_sam_simple_demo
from GSA import automatic_label_simple_demo
import gpt4v
import json
if __name__ == "__main__":
    name_img='44.jpg'
    # contents=gpt4v.llm_getobj(name_img)
    # content = json.dump(contents)
    # 访问字段并打印
    # object = content["Type"]
    # print(object)
    # automatic_label_simple_demo.auto_label(name_img)
    # obj_list=["mustard bottle", "box", "scissors", "clamp", "wrench", "apple"]
    # obj_list=["banana", "box", "can", "can", "can", "bottle", "clamp", "scissors", "wrench", "key"]
    # obj_list=["smooth metal surface"]

    obj_list = ['silver wrench', 'silver key','gray and orange scissors','silver and black scissors','silver nail']
    grounded_sam_simple_demo.ground_sam(name_img,obj_list)