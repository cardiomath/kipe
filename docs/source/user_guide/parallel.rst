Parallel and cluster runs
=========================

kipe uses MPI to run a forward solver across multiple processes (*ranks*). This helps with large
models when the solver supports MPI. See the `mpi4py tutorial
<https://mpi4py.readthedocs.io/en/stable/tutorial.html>`_ for an introduction to MPI.

For example, run an estimation using four ranks:

.. code-block:: bash

    mpirun -n 4 kipe estimation study.yaml

Each rank works on its part of the solver's fields. All ranks cooperate on each forward-solver
call; the sigma points are propagated one after another, not split across MPI ranks. The study
file needs no special parallel settings. Without ``mpirun``, kipe uses one rank.

Measurement data do not depend on the number of ranks, so synthesis and estimation can use
different rank counts. Checkpoints written by the forward solver may depend on the rank count;
use the same count when reading those checkpoints.

For multiple noise realizations, ``kipe estimation study.yaml`` estimates each realization in
turn. To run realizations as separate cluster tasks, select each one with
:option:`--realization <kipe estimation --realization>`:

.. code-block:: bash

    #SBATCH --array=0-29
    srun kipe estimation study.yaml --realization "$SLURM_ARRAY_TASK_ID"

Only rank 0 reports progress on the console; other ranks report warnings and errors.
