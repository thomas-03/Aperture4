=======================
 Setting Up a Problem
=======================

Here's a minimal example of setting up a PIC simulation in *Aperture*. This example shows the essential components needed for a basic 2D simulation:
This guide provides a step-by-step walkthrough for creating a new simulation problem in Aperture. We will use the existing ``reconnection`` problem as a reference.

A new problem is essentially a new executable that links against the core ``Aperture`` library. It defines the specific systems, parameters, and initial conditions for a simulation run.

Step 1: Create the Directory Structure
---------------------------------------

    using namespace Aperture;

    int main(int argc, char* argv[]) {
        // Define a 2D simulation configuration
        typedef Config<2> Conf;
        
        // Initialize the simulation environment
        auto &env = sim_environment::instance(&argc, &argv);
        
        // Set up the execution policy (automatically handles GPU/CPU)
        using exec_policy = exec_policy_dynamic<Conf>;

        // Create the domain communicator for parallel processing
        domain_comm<Conf, exec_policy_dynamic> comm;
        
        // Initialize the simulation grid
        grid_t<Conf> grid(comm);

        // Register the core simulation systems
        // 1. Particle pusher - handles particle motion
        auto pusher = env.register_system<
            ptc_updater<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid, &comm);
        
        // 2. Field solver - updates electromagnetic fields
        auto solver = env.register_system<
            field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid, &comm);
        
        // 3. Data exporter - handles output of simulation data
        auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);

        // Initialize all registered systems
        env.init();

        // Set up initial conditions here
        // ... (see below for examples)

        // Start the simulation
        env.run();
    }

The above code sets up a basic 2D PIC simulation with three essential components:

1. **Particle Pusher**: Handles the motion of particles in the simulation
2. **Field Solver**: Updates the electromagnetic fields based on particle positions
3. **Data Exporter**: Manages the output of simulation data

The ``Config`` class is a template class that defines compile-time configurations for your simulation. The template parameter (2 in this example) specifies the dimensionality of the simulation. Other configurations include:
- Data type for floating-point numbers
- Particle pusher type
- Indexing scheme
- Other compile-time parameters

The simulation environment (``sim_environment``) manages the lifecycle of all systems. When you register a system using ``register_system<T>``, it:
1. Creates an instance of the system
2. Adds it to the system registry
3. Returns a pointer to the system for further configuration

Systems are executed in the order they are registered. In this example, each timestep will:
1. Update particle positions (pusher)
2. Update electromagnetic fields (solver)
3. Export data if needed (exporter)

There are two main ways to customize your simulation:

1. Through the configuration file (``config.toml``)
2. Programmatically in the source code

Config File
-----------

Every ``system`` has a number of parameters that can be customized through run
time parameters. All parameters are read from a configuration file in the
`toml <https://github.com/toml-lang/toml>`_ format. By default, the code will
look for a file named `config.toml` in the same directory as the executable. A
different config file can also be specified through a launch parameter:
First, create a new directory for your problem inside the ``problems/`` directory. For this example, we'll call it ``my_new_problem``.

.. code-block:: console

   $ cd /path/to/Aperture4
   $ mkdir problems/my_new_problem
   $ mkdir problems/my_new_problem/src

To see all available parameters and their default values without running the simulation,
you can use the dry-run option:

.. code-block:: console

   $ ./aperture --dry-run

This will print out all parameters that can be configured, along with their current
values, making it easier to understand what can be customized in your simulation.

Parameters are stored in an instance of :ref:`params_store` in the
:ref:`sim_environment` class. One can also define all the required parameters
programmatically:
All source code for your problem will reside in the ``problems/my_new_problem/src/`` directory.

Step 2: Create the CMakeLists.txt File
---------------------------------------

Each problem needs its own ``CMakeLists.txt`` file to tell the build system how to compile it. Create the file ``problems/my_new_problem/CMakeLists.txt`` with the following content:

.. code-block:: cmake

   add_aperture_executable(my_new_problem src/main.cpp)

This command, provided by the Aperture build system, creates a new executable target named ``my_new_problem`` from the source file ``src/main.cpp``.

Step 3: Write the main.cpp Entry Point
---------------------------------------

The ``main.cpp`` file is the heart of your problem. It's where you assemble the simulation components. Here is a breakdown of its structure, based on the ``reconnection`` problem.

1. **Includes and Namespace:**
   Start by including the necessary headers for the framework and the systems you intend to use.

   .. code-block:: cpp

      #include "framework/config.h"
      #include "framework/environment.h"
      #include "framework/system.h"
      #include "systems/data_exporter.h"
      #include "systems/field_solver_default.h"
      #include "systems/grid.h"
      #include "systems/ptc_updater.h"
      #include "systems/domain_comm.h"

      #include <iostream>

      using namespace std;
      using namespace Aperture;


2. **Main Function and Environment Setup:**
   The ``main`` function initializes the ``sim_environment``, which manages the entire simulation. You also define the configuration (e.g., ``Config<2>`` for 2D).

   .. code-block:: cpp

      int main(int argc, char *argv[]) {
        typedef Config<2> Conf;
        sim_environment env(&argc, &argv);
        // ...
      }

3. **Register Systems:**
   Next, you register all the systems (modules) required for the simulation. The order of registration can be important, as some systems depend on others.

   .. code-block:: cpp

      auto comm = env.register_system<domain_comm<Conf>>(env);
      auto grid = env.register_system<grid_t<Conf>>(env, *comm);
      auto solver = env.register_system<field_solver_default<Conf>>(env, *grid, comm);
      auto pusher = env.register_system<ptc_updater<Conf>>(env, *grid, &comm);
      auto exporter = env.register_system<data_exporter<Conf>>(env, *grid, comm);

4. **Set Initial Conditions:**
   This is a critical step where you define the problem-specific initial state. As discussed in the architecture overview, this is distinct from system configuration. The typical pattern is:

   a. Get pointers to the data components managed by the systems (e.g., particles and fields).
   b. Call a dedicated function to populate these components with initial values.

   Following the ``reconnection`` example:

   .. code-block:: cpp

      // Get pointers to data components
      auto *ptcs = pusher->get_ptcs();
      auto *ems = solver->get_ems();

      // Define and call the initial condition function
      auto IC = [&](const typename Conf::coord_t &x) {
        // ... logic for setting initial particle and field values ...
        // Example: set density, temperature, magnetic field, etc.
        double B0 = env.params().get_or("B0", 1.0);
        ems->b(x, 0) = B0 * tanh(x[1] / 0.5);
        // ... more initial condition logic ...
      };

      grid->init_data(ptcs, IC);
      grid->init_data(ems, IC);


5. **Run the Simulation:**
   Finally, initialize the environment and start the simulation loop.

   .. code-block:: cpp

      env.init();
      env.run();
      return 0;

Step 4: Add to the Main Build
-----------------------------

The last step is to tell the main Aperture build system about your new problem. Open the file ``problems/CMakeLists.txt`` and add your problem's directory to the list:

.. code-block:: cmake

   # ... existing problems ...
   add_subdirectory(reconnection)
   add_subdirectory(my_new_problem) # Add this line

After completing these steps, you can re-run CMake and build your new problem executable.