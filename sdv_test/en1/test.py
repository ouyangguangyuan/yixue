from sdv.datasets.demo import download_demo
from sdv.lite import SingleTablePreset
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_pair_plot
from sdv.evaluation.single_table import get_column_plot

#1、加载元数据、真实数据
real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
#real_data.to_csv('fake_hotel_guests.csv', index=False)
#2、根据真实数据生成仿真数据
synthesizer = SingleTablePreset(metadata, name='FAST_ML')
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv('fake_hotel_guests_new.csv', index=False)
#3、数据诊断
diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

#4、数据质量
quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)

print('-----',quality_report.get_details('Column Shapes'))
sensitive_column_names = ['guest_email', 'billing_address', 'credit_card_number']
print('--real_data--',real_data[sensitive_column_names].head(3))
print('--synthetic_data--',synthetic_data[sensitive_column_names].head(3))
synthesizer.save('my_synthesizer.pkl')
synthesizer = SingleTablePreset.load('my_synthesizer.pkl')


#获取列图
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='amenities_fee',
    metadata=metadata
)
fig.show()
#获取列对图
fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['checkin_date', 'checkout_date'],
    metadata=metadata
)
fig.show()