"""Fit sampled 3D coordinates to ``z = a*x + b*y + c`` for the ``plane`` measurement model.

:func:`fit_plane_geometry` performs an orthogonal least-squares fit using SVD;
:func:`orient_normal` selects the sign of its unit normal. :func:`fit_plane` returns only
the coefficients and can also prepare measurement data from externally measured points.
These utilities use NumPy arrays and do not require a finite element backend.

:class:`PlaneModel` gathers distributed sampled coordinates, fits the complete point set
on MPI rank 0, and compares predicted and measured coefficients by subtraction.
"""

from dataclasses import dataclass

from mpi4py import MPI

import numpy as np

from kipe._types import NDArray_f64, NDArray_i64
from kipe.measurements.models import MeasurementModel
from kipe.options import StudyFileError
from kipe.sampling import SpatialSampler


@dataclass(frozen=True)
class PlaneFit:
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
    point. With neither option, keep the fitted sign. This function stores no history:
    callers retain the first oriented normal and pass it as the reference for later fits.

    Args:
        normal: fitted unit normal, shape ``(3,)``
        centroid: fitted plane's centroid, shape ``(3,)``
        normal_toward: optional point in the same coordinates and units as the centroid
        reference_normal: finite nonzero normal, shape ``(3,)``; takes precedence over
            the point, which is then ignored. Use the first fit's normal for a fixed reference.

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


def fit_plane(vertices: NDArray_f64) -> NDArray_f64:
    """Fit a plane to 3D vertices and return only its measurement coefficients.

    Args:
        vertices: finite coordinates, shape ``(n_vertices, 3)``; at least three
            noncollinear vertices

    Returns:
        ``[a, b, c]`` for ``z = a*x + b*y + c``, shape ``(3,)``. The slopes are
        dimensionless; c uses the coordinates' length unit.

    Raises:
        ValueError: if the vertices do not define a unique supported plane; see
            :func:`fit_plane_geometry` for the geometric restrictions
    """
    return fit_plane_geometry(vertices).coefficients


def fit_plane_geometry(
    vertices: NDArray_f64,
    *,
    normal_toward: tuple[float, float, float] | None = None,
    reference_normal: NDArray_f64 | None = None,
) -> PlaneFit:
    """Fit ``z = a*x + b*y + c`` by minimizing squared perpendicular distances.

    Center the vertices at their centroid and use the least varying SVD direction as the
    plane normal. The fit treats all three coordinate directions equally; it does not
    minimize only vertical z residuals. Convert the normal and centroid to ``[a, b, c]``
    after selecting the normal's sign with :func:`orient_normal`.

    Args:
        vertices: finite 3D coordinates, shape ``(n_vertices, 3)``; at least three
            noncollinear vertices
        normal_toward: optional point on the desired side of the first plane, in the same
            coordinates and units as the vertices; must not lie on the fitted plane
        reference_normal: first oriented normal; aligns later fits and overrides the point.
            The caller must retain it: this fitting utility has no internal history.

    Returns:
        coefficients ``[a, b, c]``, centroid and unit normal. Without a reference or point,
        the normal retains the SVD sign. Pass the first fit's normal to align later fits.

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
    return PlaneFit(coefficients, normal, centroid)


class PlaneModel(MeasurementModel):
    """Fit sampled 3D vertices and compare the measured and predicted plane coefficients.

    Vertex coordinates are gathered in the sampler's global order for every prediction. Rank
    0 returns ``[a, b, c]``; other ranks return an empty array. All ranks must participate in
    construction and prediction, including those without sampled entries.

    The first successful prediction establishes this instance's normal reference, optionally
    oriented toward ``normal_toward``. Later predictions align with that first normal,
    including sigma-point predictions and later outer iterations using the same instance.
    A failed fit does not replace the reference. A new instance, including one created on
    resume, establishes a new reference; orientation never changes the measured coefficients.

    Args:
        sampler: sampler of flattened coordinates ``x0, y0, z0, x1, y1, z1, ...``
        comm: communicator shared with the sampler; root is communicator rank 0
        normal_toward: optional point toward which the first fitted unit normal points;
            later predictions align with that first normal

    Raises:
        StudyFileError: if the sampled size is not three coordinates per vertex with at least
            three vertices, or the global indices do not cover every entry exactly once
    """

    def __init__(
        self,
        sampler: SpatialSampler,
        comm: MPI.Comm = MPI.COMM_WORLD,
        *,
        normal_toward: tuple[float, float, float] | None = None,
    ) -> None:
        # Each vertex contributes x, y and z; fitting needs at least nine sampled entries.
        if sampler.size < 9 or sampler.size % 3:
            raise StudyFileError("plane model needs at least three 3D vertices (3N coordinates)")
        # Keep the sampler's global input size and the communicator used for all collectives.
        self._comm = comm
        self._size = sampler.size
        self._normal_toward = normal_toward
        self._normal: NDArray_f64 | None = None
        self._reference_normal: NDArray_f64 | None = None

        # Cache the global positions, which the sampler fixes at construction. This also
        # supports a partition that splits the coordinate components of a vertex.
        self._indices = comm.gather(sampler.global_indices.copy(), root=0)
        # Verify on rank 0 that every coordinate is present exactly once, with no missing
        # entries or duplicate owners, before using these positions for later predictions.
        error = None
        if comm.rank == 0:
            assert self._indices is not None
            indices = np.concatenate(self._indices)
            if not np.array_equal(np.sort(indices), np.arange(self._size)):
                error = (
                    "plane model needs each sampled coordinate to be owned by exactly one rank"
                )
        # All ranks must either finish construction or raise the same validation error.
        error = comm.bcast(error, root=0)
        if error is not None:
            raise StudyFileError(error)

    @property
    def normal(self) -> NDArray_f64 | None:
        """Copy of the latest predicted unit normal on rank 0, or None when unavailable.

        This describes the latest prediction, which may be a sigma point rather than the
        final estimated state. None is returned before a successful fit, on non-root ranks,
        and after a failed prediction. Failure clears this value but retains the first
        normal used as the orientation reference. Modifying the returned copy has no effect
        on subsequent predictions.
        """
        return None if self._normal is None else self._normal.copy()

    def layout(self, sampler: SpatialSampler | None = None) -> tuple[int, NDArray_i64]:
        """Return three coefficients, owned by rank 0 independently of the vertex count.

        Args:
            sampler: accepted for the measurement-model interface; the plane output layout
                is fixed and does not depend on this argument

        Returns:
            global size 3 and indices ``[0, 1, 2]`` on rank 0, or empty indices on other ranks
        """
        # The model returns three coefficients rather than 3N coordinates. Assign them only
        # to rank 0 so synthesis, estimation and MPI reductions count each coefficient once.
        return 3, np.arange(3 if self._comm.rank == 0 else 0, dtype=np.int64)

    def predict(self, y: NDArray_f64) -> NDArray_f64:
        """Gather the current vertices and return this rank's fitted coefficients.

        Reassemble flattened coordinates using the sampler's global indices, then fit the
        complete ``(n_vertices, 3)`` array on rank 0. This supports coordinate components
        split between ranks and ranks with no entries. Broadcast fit errors before raising
        so all ranks leave the collective prediction together.

        Args:
            y: one-dimensional array of this rank's coordinate components, in exactly the
                local order of the sampler's ``global_indices``

        Returns:
            ``[a, b, c]`` on rank 0, an empty array on other ranks

        Raises:
            ValueError: on every rank if sampled entries do not match the cached layout,
                the vertices cannot be fitted by :func:`fit_plane_geometry`, or orientation
                relative to the first normal is ambiguous
        """
        # Gather the current coordinate components, including empty arrays from ranks that
        # own no sampled entries. The stored global indices restore the vertex order.
        self._normal = None
        parts = self._comm.gather(y, root=0)
        # Other ranks keep an empty prediction because rank 0 owns all three coefficients.
        coefficients = np.empty(0, dtype=np.float64)
        error = None
        if self._comm.rank == 0:
            assert parts is not None and self._indices is not None
            # Reconstruct the global coordinate vector and check each rank's local size.
            vertices_coordinates = np.empty(self._size, dtype=np.float64)
            try:
                for indices, values in zip(self._indices, parts, strict=True):
                    if values.ndim != 1 or len(values) != len(indices):
                        raise ValueError(
                            "sampled coordinate entries do not match the sampler layout"
                        )
                    vertices_coordinates[indices] = values
                # Group each vertex's x, y and z, then fit one plane to all vertices.
                fit = fit_plane_geometry(
                    vertices_coordinates.reshape(-1, 3),
                    normal_toward=self._normal_toward,
                    reference_normal=self._reference_normal,
                )
                coefficients = fit.coefficients
                self._normal = fit.normal
                if self._reference_normal is None:
                    self._reference_normal = fit.normal.copy()
            except ValueError as err:
                # Delay raising until the failure has been communicated to every rank.
                error = str(err)

        # Share failures before returning so other ranks cannot continue into a collective
        # operation while rank 0 has already stopped with an invalid fit.
        error = self._comm.bcast(error, root=0)
        if error is not None:
            raise ValueError(error)
        return coefficients

    def innovation(self, z_hat: NDArray_f64, z: NDArray_f64) -> NDArray_f64:
        """Return measured minus predicted coefficients, before noise weighting.

        Both arrays follow the output ownership of :meth:`layout`: three coefficients on
        rank 0 and no entries on peers. Estimation applies the measurement noise weights.

        Args:
            z_hat: this rank's predicted coefficients
            z: this rank's measured coefficients, in the same order

        Returns:
            ``z - z_hat``
        """
        # Compare [a, b, c] component by component; estimation applies noise weighting later.
        return z - z_hat
