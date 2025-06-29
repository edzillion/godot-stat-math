StatMath.AcklamNormalPpfCoefficients
====================================

Coefficients for Peter John Acklam's approximation algorithm for the inverse
cumulative standard normal distribution function (normal PPF/quantile function).

This algorithm provides a minimax approximation by rational functions with
relative error whose absolute value is less than 1.15e-9.
Source: Peter John Acklam's algorithm
Reference: https://gist.github.com/roguetrainer/8188638
Original Author: Peter John Acklam (pjacklam@online.no)
Time-stamp: 2000-07-19 18:26:14
WWW URL: http://home.online.no/~pjacklam

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.AcklamNormalPpfCoefficients.function_name(parameters)

Constants
---------

.. data:: VALUES

   Value: ``Dictionary = {``

