Installation
============

Installing Godot Stat Math addon in your Godot 4 project.

Prerequisites
-------------

* Godot 4.0 or higher
* Basic understanding of GDScript

Installation Methods
--------------------

Asset Library (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open Godot Editor
2. Go to **Project** → **Project Settings** → **Plugins**
3. Search for "Godot Stat Math" in the Asset Library
4. Click **Download** and **Install**
5. Enable the plugin in the Plugins tab

Manual Installation
~~~~~~~~~~~~~~~~~~~

1. Download the latest release from the `GitHub repository <https://github.com/edzillion/godot-stat-math>`_
2. Extract the contents to your project's ``addons/`` folder
3. The structure should look like: ``addons/godot-stat-math/``
4. In Godot Editor, go to **Project** → **Project Settings** → **Plugins**
5. Find "Godot Stat Math" and enable it

Git Submodule (For Developers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git submodule add https://github.com/edzillion/godot-stat-math.git addons/godot-stat-math

Verification
------------

After installation, verify the addon is working:

.. code-block:: gdscript

   func _ready():
       # Test that StatMath is available
       var random_normal = StatMath.Distributions.randf_normal(0.0, 1.0)
       print("Random normal value: ", random_normal)
       
       # Test basic statistics
       var data = [1.0, 2.0, 3.0, 4.0, 5.0]
       var mean_val = StatMath.BasicStats.mean(data)
       print("Mean of [1,2,3,4,5]: ", mean_val)  # Should print 3.0

Configuration
-------------

Optional: Set Global Seed
~~~~~~~~~~~~~~~~~~~~~~~~~

For reproducible results across your entire project, add this to your ``project.godot`` file:

.. code-block:: ini

   [application]
   config/name="Your Game"
   godot_stat_math_seed=12345

This ensures all StatMath functions use the same seed for consistent results.

Troubleshooting
---------------

Plugin Not Appearing
~~~~~~~~~~~~~~~~~~~~~

* Ensure the folder structure is correct: ``addons/godot-stat-math/plugin.cfg`` should exist
* Refresh the project (Project → Reload Current Project)
* Check that you're using Godot 4.0 or higher

StatMath Not Available
~~~~~~~~~~~~~~~~~~~~~~

* Verify the plugin is enabled in Project Settings → Plugins
* Make sure you're not trying to use StatMath in ``_init()`` - use ``_ready()`` instead
* Check the console for any error messages

Getting Help
~~~~~~~~~~~~

* Check the `GitHub Issues <https://github.com/edzillion/godot-stat-math/issues>`_
* Review the examples in the ``addons/godot-stat-math/examples/`` folder 