# Perceptron Algorithm
- Primal Perceptron with Margin
- Kernel/Dual Perceptron with Margin

In this project, I have implemented the primal and dual/kernel versions of the perceptron algorithm and evaluate its performance on six datasets with different d and s. d is the parameter of polynomial kernel and s is the parameter of Radial Basis Function (RBF) kernel.
<br><br>
Accuracy for datasets:
<table class="tableizer-table">
<thead><tr class="tableizer-firstrow"><th></th><th>A</th><th>B</th><th>C</th><th>back</th><th>sonar</th><th>breast</th></tr></thead><tbody>
 <tr><td>Primal</td><td>75.86%</td><td>77.24%</td><td>74.0%</td><td>67.74%</td><td>85.0 %</td><td>95.17%</td></tr>
 <tr><td>d = 1</td><td>75.86%</td><td>77.24%</td><td>74.0%</td><td>67.74%</td><td>85.0 %</td><td>95.17%</td></tr>
 <tr><td>d = 2</td><td>94.10%</td><td>84.90%</td><td>84.0%</td><td>80.64%</td><td>85.0 %</td><td>93.85%</td></tr>
 <tr><td>d = 3</td><td>93.65%</td><td>83.01%</td><td>82.0%</td><td>83.87%</td><td>100.0 %</td><td>96.05%</td></tr>
 <tr><td>d = 4</td><td>92.65%</td><td>81.70%</td><td>74.0%</td><td>83.87%</td><td>85.0 %</td><td>94.73%</td></tr>
 <tr><td>d = 5</td><td>91.76%</td><td>79.41%</td><td>70.0%</td><td>74.19%</td><td>85.0 %</td><td>94.73%</td></tr>
 <tr><td>s = 0.1</td><td>89.37%</td><td>76.82%</td><td>68.0%</td><td>77.41%</td><td>20.0 %</td><td>94.73%</td></tr>
 <tr><td>s = 0.5</td><td>91.21%</td><td>78.84%</td><td>74.0%</td><td>77.41%</td><td>85.0 %</td><td>95.61%</td></tr>
 <tr><td>s = 1</td><td>93.93%</td><td>82.58%</td><td>74.0%</td><td>77.41%</td><td>85.0 %</td><td>96.05%</td></tr>
 <tr><td>s = 2</td><td>90.32%</td><td>86.02%</td><td>82.0%</td><td>58.06%</td><td>95.0 %</td><td>96.05%</td></tr>
 <tr><td>s = 5</td><td>80.70%</td><td>77.55%</td><td>76.0%</td><td>48.38%</td><td>90.0 %</td><td>95.17%</td></tr>
 <tr><td>s = 10</td><td>81.47%</td><td>74.90%</td><td>78.0%</td><td>48.38%</td><td>90.0 %</td><td>95.17%</td></tr>
</tbody></table>

- The primal and dual version of algorithms with linear kernel are identical. You can see in the table that for all datasets the primal and dual version with linear kernel have the same accuracy.
- For polynomial kernel, the accuracy is highest when d = 2 for A, B, C datasets and when d = 3 for back, sonar, breast datasets. And, as d increases, the accuracy decreases. For RBF kernel, the accuracy is generally highest when s = 1 or 2, but as s increases, sometimes accuracy increases and sometimes it decreases.