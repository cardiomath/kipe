"""Fit sampled 3D coordinates to ``z = a*x + b*y + c`` for the ``plane`` measurement model.

:func:`fit_from_coordinates` performs an orthogonal least-squares fit using SVD;
:func:`orient_normal` selects the sign of its unit normal.
These utilities use NumPy arrays and do not require a finite element backend.

:class:`PlaneModel` fits complete coordinate arrays and compares coefficients by subtraction.
Orientation tracking belongs to the plane model and is independent of coefficient values.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from kipe._types import NDArray_f64, NDArray_i64
from kipe.measurements.models import MeasurementModel
from kipe.sampling import SpatialSampler


@dataclass(frozen=True)
class Plane:
    """Geometry of an orthogonal least-squares plane fit.

    All arrays have shape ``(3,)`` and use the input coordinate system. Changing the normal's
    sign does not change the coefficients or the geometric plane.
    """

    coefficients: NDArray_f64
    """``[a, b, c]`` for ``z = a*x + b*y + c``; slopes are dimensionless and c is a length."""

    normal: NDArray_f64
    """Unit normal perpendicular to the fitted plane, with the selected orientation."""

    centroid: NDArray_f64
    """Mean of the fitted vertices; the fitted plane passes through this point."""


def orient_normal(
    normal: NDArray_f64,
    centroid: NDArray_f64,
    *,
    normal_toward: tuple[float, float, float] | None = None,
    reference_normal: NDArray_f64 | None = None,
) -> NDArray_f64:
    """Select a normal's sign without changing the plane or the input arrays.

    With ``reference_normal``, choose the sign giving a positive dot product with that
    reference. Otherwise, if ``normal_toward`` is supplied, choose the side containing that
    point. With neither option, keep the fitted sign. No orientation history is stored.

    Args:
        normal: fitted unit normal, shape ``(3,)``
        centroid: fitted plane's centroid, shape ``(3,)``
        normal_toward: optional point in the same coordinates and units as the centroid
        reference_normal: finite nonzero normal, shape ``(3,)``; takes precedence over
            the point, which is then ignored

    Returns:
        a copy of ``normal``, multiplied by -1 if required; its magnitude is unchanged

    Raises:
        ValueError: if the reference normal or point has invalid dimensions or nonfinite
            entries, the reference normal is zero or perpendicular to the fitted normal,
            or the reference point lies on the plane, up to numerical tolerance
    """
    oriented = np.asarray(normal, dtype=np.float64).copy()
    if reference_normal is not None:
        reference = np.asarray(reference_normal, dtype=np.float64)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError("reference_normal must contain three finite coordinates")
        length = np.linalg.norm(reference)
        if length == 0:
            raise ValueError("reference_normal must be nonzero")
        # Every subsequent fit uses the first normal, never the preceding one.
        alignment = float(np.dot(oriented, reference / length))
        if abs(alignment) <= 32 * np.finfo(float).eps:
            raise ValueError("reference_normal is perpendicular; orientation is undefined")
    elif normal_toward is not None:
        target = np.asarray(normal_toward, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("normal_toward must contain three finite coordinates")
        # Only the first fit uses the point to choose a side of its plane.
        direction = target - centroid
        alignment = float(np.dot(oriented, direction))
        tolerance = (
            32
            * np.finfo(float).eps
            * max(np.linalg.norm(target), np.linalg.norm(centroid), np.linalg.norm(direction))
        )
        if abs(alignment) <= tolerance:
            raise ValueError("normal_toward lies on the fitted plane; orientation is undefined")
    else:
        return oriented
    if alignment < 0:
        oriented = -oriented
    return oriented


def fit_from_coordinates(
    vertices: NDArray_f64,
    *,
    normal_toward: tuple[float, float, float] | None = None,
    reference_normal: NDArray_f64 | None = None,
) -> Plane:
    """Fit ``z = a*x + b*y + c`` by minimizing squared perpendicular distances.

    Center the vertices at their centroid and use the least varying SVD direction as the
    plane normal. The fit treats all three coordinate directions equally; it does not
    minimize only vertical z residuals. Convert the normal and centroid to ``[a, b, c]``
    after selecting the normal's sign with :func:`orient_normal`.

    Args:
        vertices: finite 3D coordinates, shape ``(n_vertices, 3)``; at least three
            noncollinear vertices
        normal_toward: optional point on the desired side of the fitted plane, in the same
            coordinates and units as the vertices; must not lie on the fitted plane
        reference_normal: normal with which to align the fitted normal; overrides the point

    Returns:
        coefficients ``[a, b, c]``, centroid and unit normal. Without a reference or point,
        the normal retains the SVD sign.

    Raises:
        ValueError: if the coordinates have the wrong shape, are nonfinite or collinear,
            do not determine a unique best-fit plane, or SVD fails. Also raised for vertical
            or nearly vertical planes that cannot be represented reliably by this equation,
            and for invalid or ambiguous orientation references (see :func:`orient_normal`).
    """
    # Convert the input to floating-point coordinates and require at least three 3D vertices.
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("plane fit needs at least three 3D vertices, shape (n_vertices, 3)")
    # Reject NaN and infinity before computing the centroid or the SVD.
    if not np.all(np.isfinite(vertices)):
        raise ValueError("plane fit needs finite vertex coordinates")

    # Center the vertices and compute their principal directions. The least varying
    # direction is normal to the plane that minimizes squared perpendicular distances.
    centroid = vertices.mean(axis=0)
    try:
        _, singular_values, directions = np.linalg.svd(vertices - centroid, full_matrices=False)
    except np.linalg.LinAlgError as err:
        raise ValueError("plane fit did not converge") from err
    # Require variation in at least two independent directions: a line or a single location
    # cannot define a plane. Scale the tolerance to the vertex spread and floating precision.
    tolerance = np.finfo(np.float64).eps * max(vertices.shape) * singular_values[0]
    if singular_values[1] <= tolerance:
        raise ValueError("plane fit needs noncollinear vertices")
    # Equal smallest singular values leave multiple equally good plane normals.
    if singular_values[1] - singular_values[2] <= tolerance:
        raise ValueError("vertices do not determine a unique best-fit plane")

    # Extract the normal and check its z component before dividing by it. A vertical plane
    # cannot be expressed as z = a*x + b*y + c; near-vertical fits amplify numerical errors.
    normal = directions[-1]
    if abs(normal[2]) <= np.sqrt(np.finfo(np.float64).eps):
        raise ValueError(
            "vertical or nearly vertical plane cannot be written as z = a*x + b*y + c"
        )
    normal = orient_normal(
        normal, centroid, normal_toward=normal_toward, reference_normal=reference_normal
    )
    # Convert n . (X - centroid) = 0 to [a, b, c]. These ratios are unchanged if SVD flips
    # the normal, so selecting its orientation does not change the coefficients.
    coefficients = np.array([-normal[0], -normal[1], np.dot(normal, centroid)]) / normal[2]
    return Plane(coefficients, normal, centroid)


@dataclass
class PlaneOrientation:
    """Orientation history for one plane model, independent of other model instances.

    Args:
        toward: optional reference point used only to orient the first fitted normal
    """

    toward: tuple[float, float, float] | None = None
    reference: NDArray_f64 | None = field(default=None, init=False)
    latest: NDArray_f64 | None = field(default=None, init=False)

    def record(self, plane: Plane) -> None:
        """Store the oriented normal and retain the first normal as the reference.

        Use the optional point only when establishing the reference. Later normals align
        with that reference. The supplied plane is unchanged.

        Args:
            plane: fitted geometry, before applying this history's orientation reference

        Raises:
            ValueError: if the reference point or first normal cannot determine an orientation
        """
        self.latest = orient_normal(
            plane.normal,
            plane.centroid,
            normal_toward=self.toward,
            reference_normal=self.reference,
        )
        if self.reference is None:
            self.reference = self.latest.copy()


class PlaneModel(MeasurementModel):
    """Predict three plane coefficients from a complete array of sampled coordinates.

    The first successful fit establishes an orientation reference, optionally directed
    toward a point. Subsequent normals align with that first normal. Orientation does not
    change the coefficients used for prediction or innovation.
    """

    gather_input: ClassVar[bool] = True
    orientation: PlaneOrientation | None = None
    """Optional reference and latest oriented normal."""

    @property
    def normal(self) -> NDArray_f64 | None:
        """Copy of the latest oriented normal, or None before prediction or after failure.

        None is also returned when orientation tracking is not configured. The first
        reference is not modified through this copy.
        """
        normal = None if self.orientation is None else self.orientation.latest
        return None if normal is None else normal.copy()

    def clear_prediction(self) -> None:
        """Clear the latest normal without discarding the first orientation reference."""
        if self.orientation is not None:
            self.orientation.latest = None

    def layout(self, sampler: SpatialSampler) -> tuple[int, NDArray_i64]:
        """Return size 3 and indices ``[0, 1, 2]``, independently of the sampler.

        Args:
            sampler: accepted for the model interface; the output always has three entries

        Returns:
            size 3 and indices ``[0, 1, 2]``
        """
        return 3, np.arange(3, dtype=np.int64)

    def predict(self, y: NDArray_f64) -> NDArray_f64:
        """Fit the complete coordinate array and return ``[a, b, c]``.

        Clear the latest normal before fitting and record the fitted normal when
        orientation tracking is configured.

        Args:
            y: flattened coordinates in vertex order, ``x0, y0, z0, x1, y1, z1, ...``

        Returns:
            three plane coefficients, independently of vertex count

        Raises:
            ValueError: if the input cannot be reshaped to 3D points, cannot define a
                supported plane, or the orientation reference is ambiguous
        """
        self.clear_prediction()
        plane = fit_from_coordinates(y.reshape(-1, 3))
        if self.orientation is not None:
            self.orientation.record(plane)
        return plane.coefficients

    def innovation(self, z_hat: NDArray_f64, z: NDArray_f64) -> NDArray_f64:
        """Return measured minus predicted coefficients, before noise weighting.

        Args:
            z_hat: predicted coefficients in the order ``[a, b, c]``
            z: measured coefficients in the same order

        Returns:
            ``z - z_hat``
        """
        return z - z_hat
