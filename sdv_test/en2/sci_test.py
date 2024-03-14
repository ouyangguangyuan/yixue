import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# Step 1: Create the synthesizer
metadata = SingleTableMetadata.load_from_json(filepath='sci.json')
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
        'AU': 'uniform',
        'TI': 'beta',
        'LA': 'uniform',
        'DT': 'uniform',
        'AB': 'beta',
        'PY': 'uniform',
        'VL': 'uniform',
        'PG': 'norm',
        'GA': 'norm',
        'DA': 'norm',
    },
    default_distribution='norm'
)

parameters = synthesizer.get_parameters()
print('parameters',parameters)
metadata = synthesizer.get_metadata()
metadata.validate()
print('metadata',metadata)
real_data =pd.read_csv('sci.csv')
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=1000)
synthetic_data.to_csv('sci_new.csv', index=False)
distributions = synthesizer.get_learned_distributions()
print('distributions',distributions)
#运行诊断程序
from sdv.evaluation.single_table import run_diagnostic
diagnostic_report = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata)
print('diagnostic_report',diagnostic_report)