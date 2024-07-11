from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

df0 = pd.read_csv("T0_S_Cloudy Noon_2024-05-31_11-23-25.csv") 
X=df0.to_numpy() 
print(X)
print(X.shape)
sum=0
for i in range(45):
    sum=sum+X[i][7]
ave=10-(sum/45)*0.01
new_data = "T0_S_Cloudy Noon_2024-05-31_11-23-25\n"
with open("reward.txt", 'a', encoding='utf-8') as file:  
    file.write(new_data)

columns_to_drop = [0,3, 4,5,6,7]  # 列的索引  
df1 = df0.drop(df0.columns[columns_to_drop], axis=1) 
print(df1.head())  

A= df1.to_numpy() 
print(A)
print(A.shape)


df = pd.read_csv("trajectory3.csv") 
columns_to_drop = [0,3]  # 列的索引  
df = df.drop(df.columns[columns_to_drop], axis=1) 
print(df.head())  

B= df.to_numpy() 
print(B)
print(B.shape)

cosine = cosine_similarity(A,B)    # 第一行代表的是A中的第一行和B中的每一行的余弦相似度;第二行代表的是A中的第二行和B中每一行的余弦相似度
print("余弦相似度:\n", cosine)
max_col_indices = np.argmax(cosine, axis=1)
print(max_col_indices)
print(len(max_col_indices))

rows_to_concatenate = [0,1,2,3,4,5]
# 使用列表推导式从 A 中选择这些行，并使用 np.concatenate 沿着列方向拼接它们  
# 这里需要确保所有的行都有相同的列数  
concatenated_row = np.concatenate(B[rows_to_concatenate], axis=0).ravel()  # ravel() 将二维数组展平为一维 
print(concatenated_row)
C = concatenated_row[np.newaxis, :] 
print(C)
print(C.shape)


D=np.zeros((45, 12))

for i in range(45):
    if max_col_indices[i]<=94:
        j=max_col_indices[i]
        rows_to_concatenate = [j,j+1,j+2,j+3,j+4,j+5]
        concatenated_row = np.concatenate([B[row, :] for row in rows_to_concatenate])
        D[i,:]=concatenated_row
    else:
        k=max_col_indices[i]
        rows_to_concatenate = [j,j-1,j-2,j-3,j-4,j-5]
        concatenated_row = np.concatenate([B[row, :] for row in rows_to_concatenate])
        D[i,:]=concatenated_row

#print(D)
#print(D.shape)

E = np.hstack((X, D))
df3 = pd.read_csv("1.csv") 
F=df3.to_numpy() 
print(F)
print(F.shape)
G = np.hstack((E, F))

#print(E)
#print(E.shape)

# 将NumPy矩阵转换为pandas DataFrame  
df = pd.DataFrame(G)  
  
# 如果需要，可以给DataFrame的列添加名称  
df.columns = ['time', 'x', 'y','v km/h','isTOR','TOR_cnt','a km/h^2','j km/h^3','1.x','1.y','2.x','2.y','3.x','3.y','4.x','4.y','5.x','5.y','6.x','6.y','obs.x','obs.y']  
  
# 将DataFrame保存为CSV文件  
df.to_csv("2.csv", index=False)  # index=False表示不保存行索引

print(ave)