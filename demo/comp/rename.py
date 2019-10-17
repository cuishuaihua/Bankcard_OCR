import os
path = os.getcwd()[:-5]+'/test_images/'

list = os.listdir(path)
n = 0
for i in list:
    oldname = path + list[n]
    newname = path + 'card_'+str(n+1)+'.jpg'
    os.rename(oldname,newname)
    n+=1

