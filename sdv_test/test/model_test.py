"""
接下来，我们可以创建一个SDV合成器，一个可以用来创建合成数据的对象。它从真实数据中学习模式，并将其复制以生成合成数据。让我们使用FAST_ML预设合成器，它针对性能进行了优化。
"""
from sdv.lite import SingleTablePreset
from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality,get_column_plot

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests')
#配置数据的元数据（配置合成器）
synthesizer = SingleTablePreset(metadata, name='FAST_ML')
#填入真实数据
synthesizer.fit(data=real_data)
#生成合成数据
synthetic_data = synthesizer.sample(num_rows=100)
#评估合成数据质量
quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata)
#表表展示
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='amenities_fee',
    metadata=metadata
)
print(fig)
fig.show()