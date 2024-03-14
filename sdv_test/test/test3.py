import pandas as pd

# 假设我们有一个名为data的DataFrame
df = pd.read_csv('UID_ISO_FIPS_LookUp_Table.csv')

# 创建一个HMA1模型实例（也可以选择其他适合您数据的模型，如CTGAN、Copulas等）
model = HMA1()

# 指定表名（对于单个表的情况，可以任意命名）
table_name = 'test'

# 训练模型
model.fit(df, table_name=table_name)

# 使用模型生成新的合成数据
synthetic_data = model.sample(num_rows=1000, table_name=table_name)

# 查看生成的合成数据
print(synthetic_data.head())

# 保存生成的合成数据到CSV文件
synthetic_data.to_csv('synthetic_data.csv', index=False)



table_evaluator = TableEvaluator(df, synthetic_data)
try:
    table_evaluator.visual_evaluation()
except:
    print()