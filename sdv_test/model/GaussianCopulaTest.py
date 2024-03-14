import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# Step 1: Create the synthesizer
#synthesizer = GaussianCopulaSynthesizer(metadata)
metadata = SingleTableMetadata.load_from_json(filepath='my_metadata_v1.json')
"""
*`norm``：使用范数分布。
*``beta ``：使用beta发行版。
*`truncnorm``：使用truncnorm分布。
*“uniform”：使用均匀分布。
*`gamma``：使用gamma分布。
*`gaussian_kde``：使用GaussianKDE分发。该模型是非参数的，
因此使用它将使“get_parameters”不可用。
"""
synthesizer = GaussianCopulaSynthesizer(
    metadata, # required
    enforce_min_max_values=True,
    enforce_rounding=False,
    numerical_distributions={
        'CCC': 'uniform'
    },
    default_distribution='norm'
)
parameters = synthesizer.get_parameters()
metadata = synthesizer.get_metadata()
metadata.validate()
print(metadata)
real_data =pd.read_csv('test.csv')
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=100)
#synthesizer.reset_sampling()
#ynthetic_data1 = synthesizer.sample(scale=1.5)
synthetic_data.to_csv('synthetic_data.csv', index=False)
distributions = synthesizer.get_learned_distributions()
print(distributions)
synthesizer.save(filepath='my_synthesizer.pkl')

#synthesizer = GaussianCopulaSynthesizer.load(filepath='my_synthesizer.pkl')
#运行诊断程序
from sdv.evaluation.single_table import run_diagnostic
diagnostic_report = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata)
print(diagnostic_report)