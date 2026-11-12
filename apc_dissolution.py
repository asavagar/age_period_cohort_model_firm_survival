"""Annual APC dissolution model. Run: python apc_dissolution.py
Install: pip install numpy pandas scipy patsy statsmodels matplotlib
Input: firm_id, birth_date, dissolution_date, industry, region.
Dates: YYYY-MM-DD. Blank dissolution means alive at the common cutoff.
"""
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from patsy import dmatrix
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Edit these settings. END must have complete dissolution coverage.
FILE = 'example_firms.csv'
START, END = 2001, 2024

# Read one row per firm. Prefer industry and region measured at birth.
d = pd.read_csv(FILE, dtype={'firm_id': str, 'industry': str, 'region': str})
for col in ['birth_date', 'dissolution_date']:
    d[col] = pd.to_datetime(d[col], format='%Y-%m-%d').dt.normalize()
assert d.firm_id.is_unique and d.firm_id.notna().all(), 'Check firm identifiers'
assert d[['birth_date', 'industry', 'region']].notna().all().all(), 'Missing data'
assert not (d.dissolution_date < d.birth_date).any(), 'Dissolution before birth'
assert START <= END, 'Check study years'
d['cohort'] = d.birth_date.dt.year
d = d[d.cohort.between(START, END)].copy()
stop = (d.dissolution_date + pd.Timedelta(days=1)).fillna(pd.Timestamp(END+1, 1, 1))

# Annual counts and exact days at risk. Birth starts its day; dissolution ends it.
keys = ['age', 'period', 'cohort', 'industry', 'region']
cells = []
for year in range(START, END+1):
    left, right = pd.Timestamp(year, 1, 1), pd.Timestamp(year+1, 1, 1)
    days = (stop.clip(upper=right) - d.birth_date.clip(lower=left)).dt.days
    z = d.loc[days > 0, ['cohort', 'industry', 'region']].copy()
    z['period'], z['age'] = year, year - z.cohort
    z['exposure'] = days[days > 0] / 365.25
    z['events'] = ((d.dissolution_date >= left) & (d.dissolution_date < right)).astype(int)
    cells.append(z.groupby(keys, as_index=False)[['events', 'exposure']].sum())
cells = pd.concat(cells, ignore_index=True)
assert cells.events.sum() > 0, 'No dissolutions in the sample'

# Cohort effects have zero mean and zero linear trend: gamma = B @ theta.
# Annual age = period - cohort. Individual linear APC trends are not identified.
cohorts = np.sort(cells.cohort.unique())
assert len(cohorts) >= 3, 'At least three birth cohorts are needed'
B = null_space(np.vstack([np.ones(len(cohorts)), cohorts - cohorts.mean()]))
base = dmatrix('1 + C(age) + C(period) + C(industry) + C(region)', cells)
index = pd.Index(cohorts).get_indexer(cells.cohort)
X = np.column_stack([base, B[index]])
assert np.linalg.matrix_rank(X) == X.shape[1], 'Redundant controls or insufficient support'

# Piecewise-exponential survival model fitted using a Poisson exposure offset.
fit = sm.GLM(cells.events.to_numpy(), X, family=sm.families.Poisson(),
             offset=np.log(cells.exposure.to_numpy())).fit(maxiter=200, tol=1e-9)
assert fit.converged, 'Model did not converge'
n = B.shape[1]
gamma = B @ fit.params[-n:]
V = B @ np.asarray(fit.cov_params())[-n:, -n:] @ B.T
se = np.sqrt(np.maximum(np.diag(V), 0))
results = pd.DataFrame({'cohort': cohorts, 'hazard_ratio': np.exp(gamma),
                        'ci_lower': np.exp(gamma - 1.96*se),
                        'ci_upper': np.exp(gamma + 1.96*se)})
results.to_csv('cohort_effects.csv', index=False)
print(results.to_string(index=False))

# Pointwise, model-based 95% intervals. A value of 1 means no cohort deviation.
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(cohorts, results.hazard_ratio,
            yerr=[results.hazard_ratio-results.ci_lower,
                  results.ci_upper-results.hazard_ratio], fmt='o', capsize=3)
ax.axhline(1, color='grey', linestyle='--')
ax.set(xlabel='Firm birth year', ylabel='Dissolution hazard relative to cohort trend',
       title='Adjusted for annual age, year, industry and region')
fig.tight_layout()
fig.savefig('cohort_effects.png', dpi=200)
plt.close(fig)
