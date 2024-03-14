from sdv.datasets.demo import download_demo

real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
print(metadata)
print(real_data.head())
metadata.visualize()
from sdv.lite import SingleTablePreset
synthesizer = SingleTablePreset(metadata, name='FAST_ML')
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=500)
print(synthetic_data.head())

from sdv.evaluation.single_table import run_diagnostic
diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

#数据质量
from sdv.evaluation.single_table import evaluate_quality
quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)

quality_report.get_details('Column Shapes')

sensitive_column_names = ['guest_email', 'billing_address', 'credit_card_number']

real_data[sensitive_column_names].head(3)

synthetic_data[sensitive_column_names].head(3)
synthesizer.save('my_folder/my_synthesizer.pkl')
#synthesizer = SingleTablePreset.load('my_folder/my_synthesizer.pkl')

from sdv.evaluation.single_table import get_column_plot

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='amenities_fee',
    metadata=metadata
)

fig.show()

from sdv.evaluation.single_table import get_column_pair_plot

fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['checkin_date', 'checkout_date'],
    metadata=metadata
)

fig.show()

