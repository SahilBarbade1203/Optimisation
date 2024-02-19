# Optimisation
Assignment - 2

Gradient and Hessian Computation Assignment

Introduction

The goal of this assignment is to understand and implement gradient and hessian computation in Python. A scalar function f is defined as f : R^n → R and the gradient of the function g and the hessian h are calculated using the central difference scheme.

Requirements

This assignment requires only the NumPy library.

Functions

The following functions are necessary to achieve the aim of this assignment:

func: This function contains the input function of two variables.
compute_hessian: This function evaluates the approximate hessian of the input function at a point x* using the central difference method.
compute_gradient: This function evaluates the approximate derivative of the input function at a point x* using the central difference method.
Usage

Using the above functions, the gradient and hessian matrix can be obtained as follows:

delF = compute_gradient(func, x_input)
del2F = compute_hessian(func, x_input)


Skills Acquired

Understanding of gradient and hessian computation
Implementation of gradient and hessian computation using central difference scheme in Python
Usage of NumPy library in mathematical calculations
Conclusion

This assignment provided a good hands-on experience in implementing gradient and hessian computation using the central difference scheme in Python. It helped in developing the skills of using NumPy library for mathematical calculations and understanding the concepts of gradient and hessian computation.


