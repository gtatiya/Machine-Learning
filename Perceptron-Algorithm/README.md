# Perceptron Algorithm
- Primal Perceptron with Margin
- Kernel/Dual Perceptron with Margin

In this project, I have implemented the primal and dual/kernel versions of the perceptron algorithm and evaluate its performance on six datasets with different d and s. d is the parameter of polynomial kernel and s is the parameter of Radial Basis Function (RBF) kernel.
<br>
Accuracy for datasets:
Parameter | A | B | C | back | sonar | breast
--- | --- | --- | --- | --- | --- | --- 
Primal | 75.86% | 77.24% | 74.0% | 67.74% | 85.0 % | 95.17%
d = 1 | 75.86% | 77.24% | 74.0% | 67.74% | 85.0 % | 95.17%
d = 2 | **94.10%** | **84.90%** | **84.0%** | 80.64% | 85.0 % | 93.85%
d = 3 | 93.65% | 83.01% | 82.0% | **83.87%** | **100.0 %** | **96.05%**
d = 4 | 92.65% | 81.70% | 74.0% | **83.87%** | 85.0 % | 94.73%
d = 5 | 91.76% | 79.41% | 70.0% | 74.19% | 85.0 % | 94.73%
s = 0.1 | 89.37% | 76.82% | 68.0% | **77.41%** | 20.0 % | 94.73%
s = 0.5 | 91.21% | 78.84% | 74.0% | **77.41%** | 85.0 % | 95.61%
s = 1 | **93.93%** | 82.58% | 74.0% | **77.41%** | 85.0 % | **96.05%**
s = 2 | 90.32% | **86.02%** | **82.0%** | 58.06% | **95.0 %** | **96.05%**
s = 5 | 80.70% | 77.55% | 76.0% | 48.38% | 90.0 % | 95.17%
s = 10 | 81.47% | 74.90% | 78.0% | 48.38% | 90.0 % | 95.17%
<br>
- The primal and dual version of algorithms with linear kernel are identical. You can see in the table that for all datasets the primal and dual version with linear kernel have the same accuracy.
- For polynomial kernel, the accuracy is highest when d = 2 for A, B, C datasets and when d = 3 for back, sonar, breast datasets. And, as d increases, the accuracy decreases. For RBF kernel, the accuracy is generally highest when s = 1 or 2, but as s increases, sometimes accuracy increases and sometimes it decreases.