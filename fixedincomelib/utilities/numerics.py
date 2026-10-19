import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def _bucket_index(self, x: float) -> int:
        # I find the first knot with axis1[i] >= x, so x in (x_{i-1}, x_i] maps to i.
        # A point sitting exactly on a knot takes that knot's value: left continuity.
        idx = int(np.searchsorted(self.axis1_, x, side='left'))
        # Past the last knot I hold the last value (flat right wing).
        # The flat left wing already falls out since searchsorted returns 0 there.
        return min(idx, self.length_ - 1)

    def _bucket_weights(self, start_x: float, end_x: float) -> np.ndarray:
        # I treat bucket i as the interval (e_i, e_{i+1}] with edges
        # [-inf, x_0, x_1, ..., x_{N-2}, +inf], so both flat wings are just
        # the first and last buckets extended to infinity.
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_[:-1], dtype=float), [np.inf]))
        lo, hi = min(start_x, end_x), max(start_x, end_x)
        # I take the overlap length of [lo, hi] with each bucket, floored at zero.
        w = np.clip(np.minimum(hi, edges[1:]) - np.maximum(lo, edges[:-1]), 0.0, None)
        # I flip the sign if the limits come in reversed, matching the usual integral convention.
        return w if end_x >= start_x else -w

    def interpolate(self, x: float) -> float:
        # I cast to float because the factory keeps integer inputs as an int array.
        return float(self.values_[self._bucket_index(x)])

    def integrate(self, start_x: float, end_x: float) -> float:
        # The integral is each ordinate times the length of its bucket inside [start_x, end_x].
        return float(np.dot(self._bucket_weights(start_x, end_x), self.values_))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        # f(x) picks out exactly one ordinate, so its gradient is one-hot.
        grad = np.zeros(self.length_)
        grad[self._bucket_index(x)] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        # The integral is linear in the ordinates, so its gradient is exactly the bucket weights.
        return self._bucket_weights(start_x, end_x)


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')