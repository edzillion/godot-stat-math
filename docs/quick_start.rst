Quick Start Guide
==================

Get up and running with Godot Stat Math in minutes.

Basic Usage
-----------

StatMath organizes all functionality into modules accessible through the global singleton:

.. code-block:: gdscript

   # Access through StatMath.ModuleName.function_name()
   var result = StatMath.BasicStats.mean([1.0, 2.0, 3.0, 4.0, 5.0])

Core Modules Overview
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Purpose
   * - ``BasicStats``
     - Descriptive statistics (mean, variance, percentiles)
   * - ``Distributions``
     - Random number generation from probability distributions
   * - ``CdfFunctions``
     - Cumulative distribution functions
   * - ``PpfFunctions``
     - Inverse CDF functions (quantiles)
   * - ``PmfPdfFunctions``
     - Probability mass/density functions
   * - ``ErrorFunctions``
     - Error functions and related special functions
   * - ``HelperFunctions``
     - Core mathematical utilities
   * - ``SamplingGen``
     - Advanced sampling methods

Common Use Cases
----------------

Analyzing Player Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: gdscript

   # Player scores from last 10 matches
   var player_scores = [85.2, 92.1, 78.5, 88.9, 91.3, 82.7, 95.1, 89.4, 87.6, 90.8]
   
   # Calculate key statistics
   var avg_score = StatMath.BasicStats.mean(player_scores)
   var consistency = StatMath.BasicStats.standard_deviation(player_scores)
   var best_percentile = StatMath.BasicStats.percentile(player_scores, 90.0)
   
   print("Average Score: ", avg_score)
   print("Consistency (lower = more consistent): ", consistency)
   print("90th Percentile: ", best_percentile)

Procedural Content Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: gdscript

   # Generate enemy spawn rates using Poisson distribution
   var avg_enemies_per_wave = 5.0
   var enemies_this_wave = StatMath.Distributions.randi_poisson(avg_enemies_per_wave)
   
   # Generate damage values with normal distribution
   var base_damage = 100.0
   var damage_variance = 15.0
   var actual_damage = StatMath.Distributions.randf_normal(base_damage, damage_variance)
   
   # Generate rare item drops with exponential distribution
   var rarity_factor = 2.0
   var drop_rarity = StatMath.Distributions.randf_exponential(rarity_factor)

Random Events with Probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: gdscript

   # Critical hit system
   var crit_chance = 0.15  # 15% chance
   var is_critical = StatMath.Distributions.randi_bernoulli(crit_chance)
   
   # Loot box contents with different rarities
   var loot_probabilities = [0.5, 0.3, 0.15, 0.05]  # Common, Rare, Epic, Legendary
   var loot_tier = StatMath.Distributions.randi_categorical(loot_probabilities)

Advanced Sampling for Level Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: gdscript

   # Generate evenly distributed points for resource placement
   var resource_positions = StatMath.SamplingGen.generate_samples(
       50,  # number of resources
       2,   # 2D positions
       StatMath.SamplingGen.SamplingMethod.SOBOL  # quasi-random for even distribution
   )
   
   # Convert to Vector2 array for use in Godot
   var spawn_points: Array[Vector2] = []
   for pos in resource_positions:
       # Scale from [0,1] to your map size
       spawn_points.append(Vector2(pos.x * map_width, pos.y * map_height))

Working with Distributions
---------------------------

Understanding the Flow
~~~~~~~~~~~~~~~~~~~~~~

Most statistical workflows follow this pattern:

1. **Generate** random values (``Distributions`` module)
2. **Analyze** probability of events (``CdfFunctions``, ``PmfPdfFunctions``)
3. **Find** critical values (``PpfFunctions`` for quantiles)
4. **Summarize** collected data (``BasicStats``)

Example: Balancing a Random Event
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: gdscript

   # Design a boss encounter with dynamic difficulty
   func generate_boss_stats(player_level: int):
       # Base stats scale with player level
       var base_hp = player_level * 50.0
       var hp_variance = base_hp * 0.2
       
       # Generate HP with some randomness
       var boss_hp = StatMath.Distributions.randf_normal(base_hp, hp_variance)
       
       # Ensure boss isn't too weak (use 10th percentile as minimum)
       var min_hp = StatMath.PpfFunctions.normal_ppf(0.10, base_hp, hp_variance)
       boss_hp = max(boss_hp, min_hp)
       
       # Calculate probability this boss is "challenging" (top 25%)
       var challenge_threshold = StatMath.PpfFunctions.normal_ppf(0.75, base_hp, hp_variance)
       var is_challenging = boss_hp >= challenge_threshold
       
       return {
           "hp": boss_hp,
           "is_challenging": is_challenging,
           "difficulty_score": StatMath.CdfFunctions.normal_cdf(boss_hp, base_hp, hp_variance)
       }

Data Cleaning and Preprocessing
-------------------------------

Real game data often contains invalid values:

.. code-block:: gdscript

   # Raw player data from analytics
   var raw_session_times = [45.2, "invalid", 67.8, null, 23.1, -5.0, 89.3]
   
   # Clean the data automatically
   var clean_times = StatMath.HelperFunctions.sanitize_numeric_array(raw_session_times)
   # Result: [23.1, 45.2, 67.8, 89.3] (sorted, negatives and invalid values removed)
   
   # Now analyze clean data
   var avg_session = StatMath.BasicStats.mean(clean_times)
   var session_variability = StatMath.BasicStats.median_absolute_deviation(clean_times)

Reproducible Results
--------------------

For testing and debugging, use seeding:

.. code-block:: gdscript

   # Set global seed for all StatMath operations
   StatMath.set_global_seed(12345)
   
   # Or use per-function seeding for sampling
   var samples = StatMath.SamplingGen.generate_samples(
       100, 2, 
       StatMath.SamplingGen.SamplingMethod.SOBOL,
       0,      # starting_index
       54321   # specific seed for this operation
   )

Next Steps
----------

* Explore the :doc:`modules/basic_stats` for data analysis
* Learn about :doc:`modules/distributions` for random generation
* Check out :doc:`examples/game_analytics` for real-world examples
* Review :doc:`seeding_rng` for reproducible results 