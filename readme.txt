(Written by Ruijie, Modified by Chatgpt)

Code Design

The implementation is entirely based on NumPy, which allows matrix operations to be used for all core computations. While this design provides transparency and full control over the underlying math, it also makes the execution relatively slow compared to optimized libraries.

Model Structure:
A base model class is defined, requiring all subclasses to implement specific methods and variables. Model parameters are stored in a dictionary, which makes it easier to flatten and reshape them during updates. This design is particularly helpful when dealing with multi-layer neural networks, since weights and biases can be easily converted between flattened vectors and structured matrices.

Gradient Computation:
Gradients are computed using a finite difference method, which approximates derivatives by perturbing parameters step by step. Although conceptually straightforward, this method is computationally expensive and scales poorly with the number of parameters.

Gradient Descent:
The optimization logic follows these steps:

Flatten the current model parameters.

Apply perturbations to compute approximate gradients.

Update the flattened parameters using the gradient descent rule.

Reshape the updated parameters back into their matrix forms so the forward() function can be used in the next iteration.



Code Functionality

Data Processing (data.py):

Handles loading of various formats (MNIST, TXT, CSV).

Provides preprocessing (normalization, reshaping) and splitting into training, validation, and test sets.

Models (models.py):

Each model subclass implements forward(), parameter initialization, flattening, and update methods.

The design supports both simple models (linear/logistic regression) and more complex architectures (multi-layer feedforward networks).

Optimization (optimizers.py):

Implements loss functions: Mean Squared Error (MSE), Binary Cross-Entropy (BCE), and Categorical Cross-Entropy (CCE).

Uses finite-difference approximations to compute gradients.

Supports regularization (L1, L2) and early stopping during training.

Reporting (reporter.py):

Provides visualization tools such as loss curves, parity plots, and confusion matrices.

Prints accuracy, precision, recall, and F1 scores for classification tasks.




Code Demonstration

Three types of datasets are used to test the models:

Synthetic Dataset:

A human-made dataset is tested with the LinearRegression model.

Current results are unsatisfactory and need further debugging and inspection.

Text Dataset:

A tabular/text dataset is tested with the LogisticRegression model.

The data loading, preprocessing, and splitting functions work correctly.

However, due to the large number of parameters and the slow finite-difference gradient computation, each training step is time-consuming, making the overall usability of the model hard to assess.

MNIST Dataset:

The dataset is tested with the DenseFeedForwardNetwork model.

The first few gradient steps run without errors, confirming correctness of preprocessing and forward propagation.

However, training is extremely slow due to the high dimensionality of MNIST and the inefficiency of the finite-difference method, so the model’s full performance has not yet been demonstrated.