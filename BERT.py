from bert_serving.client import BertClient
import json


bc = BertClient()
print(2)
json.dumps(bc.server_status,ensure_ascii=False)
print(1)
array = bc.encode(['First do it', 'then do it right', 'then do it better'])
print(3)
print(array)
print(4)
print(array.shape)
print("End")



