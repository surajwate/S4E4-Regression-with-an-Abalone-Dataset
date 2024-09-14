from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_jobs=-1),
    "xgboost": xgb.XGBRegressor(n_jobs=-1),
    "lightgbm": lgb.LGBMRegressor(n_jobs=-1),
    "catboost": CatBoostRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
    "svr": SVR(),
    "kneighbors": KNeighborsRegressor(n_jobs=-1),
    "ridge": Ridge(),
    "lasso": Lasso()
}