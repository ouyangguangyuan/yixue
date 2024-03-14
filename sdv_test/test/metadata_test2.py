import pandas as pd
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()

# 假设已有表名'table_name'和列名'AAA'
# numerical、datetime、categorical、boolean、id、unknown
metadata.add_column('AAA', sdtype='id')
metadata.set_primary_key('AAA')
metadata.set_sequence_key('AAA')
metadata.add_column('BBB', sdtype='numerical')
metadata.set_sequence_index("BBB")
#修改列
metadata.update_column('BBB', sdtype='numerical')
print(metadata.columns)
#检查元数据状态
python_dict = metadata.to_dict()
print(python_dict)
#生成可视化图片
metadata.visualize(
    show_table_details='summarized',
    output_filepath='my_metadata.png'
)
#验证元数据
metadata.validate()
my_dataset =pd.read_csv('../model/test.csv')
#验证数据
metadata.validate_data(data=my_dataset)
#metadata.remove_primary_key()
#保存元数据
metadata.save_to_json(filepath='../model/my_metadata_v1.json')#此处有个ubg
#加载元数据
metadata = SingleTableMetadata.load_from_json(filepath='../model/my_metadata_v1.json')

print(metadata.to_dict())
