r"""Measurements, and how predicted data are compared with them.

The observation operator compares a model state with a measurement in two steps: a spatial
sampler (:mod:`kipe.sampling`) evaluates the observed fields at the measurement locations,
giving :math:`y`; a measurement model predicts the data from it, :math:`\hat{z} = M(y)`, and
compares them with the measured data :math:`z`, giving the innovation
:math:`\Gamma(z, \hat{z})`. The observation operator :math:`\mathcal{H} = M \circ S` combines
the two steps: it takes a model state to predicted data.
"""
