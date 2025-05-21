import numpy as np
arr=np.random.randint(-5,10,size=(20,5))
#print (arr)
#print ()
new=[]
arr_mean=np.mean(arr,axis=0)
#添加一行数据:
for i in range(20):
    new.append(arr_mean)
arr_mean = np.array(new)
#print (arr_mean)
arr_std=np.std(arr,axis=0)
#print (arr_std)
standardized_arr=(arr-arr_mean)/arr_std
#print (standardized_arr)
#print(np.mean(standardized_arr,axis=0))
#print (np.std(standardized_arr,axis=0))
def replace_outliers(x,threshold=3):
    diff=np.abs(x)
    for i in range(20):
        if diff[i]>threshold*np.std(x):
            x[i]=np.mean(x)
    return x
for i in range(5):
    standardized_arr[:,i]=replace_outliers(standardized_arr[:,i])
print (standardized_arr[:5])