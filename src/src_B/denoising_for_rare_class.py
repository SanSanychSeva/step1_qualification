
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer


FEATURES = [f"f_{i:02d}" for i in range(24)]


class BaseFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features=FEATURES,
        keep_original=True,
        add_categorical=True,
        max_features=None,
    ):
        self.features = features
        self.keep_original = keep_original
        self.add_categorical = add_categorical
        self.max_features = max_features

    def _prepare(self, X):
        X = X.loc[:, self.features].copy()
        return X.replace([np.inf, -np.inf], np.nan)

    def _limit(self, X):
        if self.max_features is not None:
            return X.iloc[:, :self.max_features]
        return X

    def _original(self, X):
        if self.keep_original:
            return X.copy()
        return pd.DataFrame(index=X.index)

    def _fit_stats(self, X):
        self.stats_ = {}

        for col in self.features:
            x = X[col].dropna().to_numpy(dtype=float)

            if len(x) == 0:
                self.stats_[col] = {
                    "median": 0.,
                    "mad": 1.,
                    "iqr": 1.,
                    "q": np.array([0., 0., 0., 0., 0.]),
                }
                continue

            med = np.median(x)
            mad = np.median(np.abs(x - med))

            q = np.quantile(
                x, [0.01, 0.05, 0.95, 0.99, 0.995]
            )

            iqr = np.quantile(x, .75) - np.quantile(x, .25)

            self.stats_[col] = {
                "median": med,
                "mad": max(1.4826 * mad, 1e-9),
                "iqr": max(iqr, 1e-9),
                "q": q,
            }


class RobustTailFeatures(BaseFeatureTransformer):
    """
    Исходные признаки + робастные трансформации + индикаторы хвостов.
    """

    def fit(self, X, y=None):
        X = self._prepare(X)
        self._fit_stats(X)
        return self

    def transform(self, X):
        X = self._prepare(X)
        out = self._original(X)

        for col in self.features:
            s = X[col].astype(float)
            st = self.stats_[col]

            med = st["median"]
            mad = st["mad"]
            iqr = st["iqr"]

            z = (s - med) / mad

            # Сжатие хвостов с сохранением знака
            out[f"{col}__slog"] = np.sign(s) * np.log1p(np.abs(s))

            # Robust z-score
            out[f"{col}__rz"] = z

            # Сжатая версия robust z-score
            out[f"{col}__tanh"] = np.tanh(z / 3.)

            # Квантильные индикаторы
            q01, q05, q95, q99, q995 = st["q"]

            if self.add_categorical:
                out[f"{col}__tail_low"] = (s < q01).astype(float)
                out[f"{col}__tail_high"] = (s > q99).astype(float)
                out[f"{col}__extreme_low"] = (s < q05).astype(float)
                out[f"{col}__extreme_high"] = (s > q995).astype(float)

                # Категориальная дискретизация
                out[f"{col}__bin"] = np.digitize(
                    s.fillna(med),
                    bins=[q01, q05, med, q95, q99]
                ).astype(float)

            # Масштабирование через IQR
            out[f"{col}__iqr"] = (s - med) / iqr

        return self._limit(out)


class DistributionFeatures(BaseFeatureTransformer):
    """
    Исходные признаки + percentile rank + quantile bins.
    """

    def __init__(
        self,
        features=FEATURES,
        keep_original=True,
        add_categorical=True,
        max_features=None,
        n_quantiles=100,
    ):
        super().__init__(
            features=features,
            keep_original=keep_original,
            add_categorical=add_categorical,
            max_features=max_features,
        )
        self.n_quantiles = n_quantiles

    def fit(self, X, y=None):
        X = self._prepare(X)

        self.quantiles_ = {}

        for col in self.features:
            x = X[col].dropna().to_numpy(dtype=float)

            if len(x) == 0:
                self.quantiles_[col] = np.array([0.])
            else:
                self.quantiles_[col] = np.unique(
                    np.quantile(
                        x,
                        np.linspace(
                            0, 1,
                            min(self.n_quantiles, len(x))
                        )
                    )
                )

        return self

    def transform(self, X):
        X = self._prepare(X)
        out = self._original(X)

        for col in self.features:
            s = X[col].astype(float)
            bins = self.quantiles_[col]

            # Эмпирическая позиция в распределении
            out[f"{col}__rank"] = s.rank(pct=True)

            # Нормализованная квантильная позиция
            if len(bins) > 1:
                out[f"{col}__qbin"] = np.digitize(
                    s.fillna(bins[0]),
                    bins=bins[1:-1]
                ).astype(float) / len(bins)
            else:
                out[f"{col}__qbin"] = 0.

            # Дистанция от медианы в квантильных единицах
            med = np.nanmedian(s)
            out[f"{col}__distance"] = (
                s - med
            ).abs()

            if self.add_categorical:
                # Категории хвостов
                out[f"{col}__tail_cat"] = np.select(
                    [
                        s < bins[int(len(bins) * .01)],
                        s < bins[int(len(bins) * .05)],
                        s > bins[int(len(bins) * .99)],
                        s > bins[int(len(bins) * .95)],
                    ],
                    [0, 1, 4, 3],
                    default=2
                ).astype(float)

        return self._limit(out)
