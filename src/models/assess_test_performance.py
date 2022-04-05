def get_masks(y):
    return y.isna()

y_mask = get_masks(y_test)


# %%
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test.fillna(0).values,y_hat)
rmse_masked = mean_squared_error(y_test.fillna(0).values,y_hat_avg_naive ,sample_weight=(~y_mask).astype(int),squared=False,multioutput='raw_values')
# %%
