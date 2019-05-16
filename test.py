import os
import json

# for r in range(1,7):
#     with open(str(r)+'.json', 'r') as a:
#         aa = json.loads(a.read())
#         print(aa[0])
#         print(aa[1])


type_path = os.path.join(os.getcwd(),'MOST_original/') + str(1)
for each_file in os.listdir(type_path):
    aa = int(each_file[:len(each_file)-4])
    print(type(aa))
