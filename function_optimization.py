import torch
from torch import optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator



def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def ackley(x, y):
    return -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) - torch.exp(0.5 * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))) + torch.exp(torch.tensor([1.0])) + 20


###############################################################################################
# Plot the objective function

# You will need to use Matplotlib's 3D plotting capabilities to plot the objective functions.
# Alternate plotting libraries are acceptable.
###############################################################################################

'''
YOUR CODE HERE
'''

# Create a grid of points in the input space
x = torch.linspace(-30, 30, 100)
y = torch.linspace(-30, 30, 100)
X, Y = torch.meshgrid(x, y)


#ROSENBROCK FUNCTION PLOT

# Computes the function values
Z_rosenbrock = rosenbrock(X, Y)

# Creates the 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plots the surface
surface = ax.plot_surface(X, Y, Z_rosenbrock, cmap=cm.RdYlBu, linewidth=0, antialiased=False)

# Adds a color bar for values
fig.colorbar(surface, shrink=0.5, aspect=5)

plt.title('Rosenbrock Function')
plt.xlabel('Number of Iterations')
plt.ylabel('Rosenbrock Function Values')
# plt.show()

#BEALE FUNCTION PLOT

# Computes the function values
Z_beale = beale(X, Y)

# Creates the 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plots the surface
surface = ax.plot_surface(X, Y, Z_beale, cmap=cm.RdYlBu, linewidth=0, antialiased=False)

# Adds a color bar for values
fig.colorbar(surface, shrink=0.5, aspect=5)

plt.title('Beale Function')
plt.xlabel('Number of Iterations')
plt.ylabel('Beale Function Values')
# plt.show()


#ACKLEY FUNCTION PLOT

# Computes the function values
Z_ackley = ackley(X, Y)

# Creates the 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plots the surface
surface = ax.plot_surface(X, Y, Z_ackley, cmap=cm.RdYlBu, linewidth=0, antialiased=False)

# Adds a color bar for values
fig.colorbar(surface, shrink=0.5, aspect=5)

plt.title('Ackley Function')
plt.xlabel('Number of Iterations')
plt.ylabel('Ackley Function Values')
# plt.show()


#RESTRICTED ACKLEY FUNCTION PLOT

# Creates a grid of points for the restricted range of [-3, 3]
x_range = torch.linspace(-3, 3, 100)
y_range = torch.linspace(-3, 3, 100)
X_ackley_restricted, Y_ackley_restricted = torch.meshgrid(x_range, y_range)

# Computes the function values for the Ackley restricted grid
Z_ackley_restricted = ackley(X_ackley_restricted, Y_ackley_restricted)

# Create a 3D plot for the restricted range Ackley function
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface
surf = ax.plot_surface(X_ackley_restricted , Y_ackley_restricted, Z_ackley_restricted, cmap=cm.RdYlBu, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.title('Ackley Function with Restricted Range -3 to 3')
plt.xlabel('Number of Iterations')
plt.ylabel('Ackley Function with Restricted Range Values')
# plt.show()




###############################################################################################
# STOCHASTIC GRADIENT DESCENT

# Initialize x and y to 10.0 (ensure you set requires_grad=True when converting to tensor)

# Use Stochastic Gradient Descent in Pytorch to optimize the objective function.

# Saev the values of the objective function over 5000 iterations in a list.

# Print the values of x, y, and the objective function after optimization.
###############################################################################################

'''
YOUR CODE HERE
'''

# Learning rate for SGD
sgd_learning_rate= 0.0000001

#ROSENBROCK FUNCTION!

# Initializes x and y to 10.0 and sets requires_grad=True when converting to tensor
r_x = torch.tensor([10.0], requires_grad=True)
r_y = torch.tensor([10.0], requires_grad=True)

# Stochastic Gradient Descent optimizer and assigns the learning rate
rosenbrock_optimizer_sgd = optim.SGD([r_x, r_y], lr=sgd_learning_rate)

# Lists to store the rosenbrock values for plotting
sgd_rosenbrock_iterations = []
sgd_rosenbrock_objective_values = []

#Optimization for Rosenbrock
for iteration in range(10000):
    # Computes the objective function value for rosenbrock
    objective = rosenbrock(r_x, r_y)

    # optimization??
    rosenbrock_optimizer_sgd.zero_grad()
    objective.backward()
    rosenbrock_optimizer_sgd.step()

    # saves the values of the objective function over 5000 iterations in a list
    sgd_rosenbrock_iterations.append(iteration)
    sgd_rosenbrock_objective_values.append(objective.item())

# Print final values
print(f"Rosenbrock values after SGD optimization: x: {r_x.item()}, y: {r_y.item()}, Objective: {objective.item()}")


# #BEALE FUNCTION!

# Initializes x and y to 10.0 and sets requires_grad=True when converting to tensor
b_x = torch.tensor([10.0], requires_grad=True)
b_y = torch.tensor([10.0], requires_grad=True)

# Stochastic Gradient Descent optimizer and assigns the learning rate
beale_optimizer_sgd = optim.SGD([b_x, b_y], lr=sgd_learning_rate)

# Lists to store the beale values for plotting
sgd_beale_iterations = []
sgd_beale_objective_values = []

#Optimization for Beale
for iteration in range(10000):
    # Computes the objective function value for beale
    objective = beale(b_x, b_y)

    # optimization??
    beale_optimizer_sgd.zero_grad()
    objective.backward()
    beale_optimizer_sgd.step()

    # saves the values of the objective function over 5000 iterations in a list
    sgd_beale_iterations.append(iteration)
    sgd_beale_objective_values.append(objective.item())

# Print final values
print(f"Beale values after SGD optimization: {b_x.item()}, y: {b_y.item()}, Objective: {objective.item()}")




# #ACKLEY FUNCTION!

# Initializes x and y to 10.0 and sets requires_grad=True when converting to tensor
a_x = torch.tensor([10.0], requires_grad=True)
a_y = torch.tensor([10.0], requires_grad=True)

# Stochastic Gradient Descent optimizer and assigns the learning rate
ackley_optimizer_sgd = optim.SGD([a_x, a_y], lr=sgd_learning_rate)

# Lists to store the ackley values for plotting
sgd_ackley_iterations = []
sgd_ackley_objective_values = []

#Optimization for Ackley
for iteration in range(10000):
    # Computes the objective function value for ackley
    objective = ackley(a_x, a_y)

    # optimization??
    ackley_optimizer_sgd.zero_grad()
    objective.backward()
    ackley_optimizer_sgd.step()

    # saves the values of the objective function over 5000 iterations in a list
    sgd_ackley_iterations.append(iteration)
    sgd_ackley_objective_values.append(objective.item())

# Print final values
print(f"Ackley values after SGD optimization: {a_x.item()}, y: {a_y.item()}, Objective: {objective.item()}")





# ###############################################################################################
# # Adam Optimizer

# # Re-initialize x and y to 10.0 (ensure you set requires_grad=True when converting to tensor)

# # Use the Adam optimizer in Pytorch to optimize the objective function.

# # Saev the values of the objective function over 5000 iterations in a list.

# # Print the values of x, y, and the objective function after optimization.
# ###############################################################################################

# '''
# YOUR CODE HERE
# '''
# #the setup should be similar to the previous question

# Learning rate for adam
adam_learning_rate = 0.001

#ROSENBROCK FUNCTION!

# Initializes x and y to 10.0 and sets requires_grad=True when converting to tensor
ro_x = torch.tensor([10.0], requires_grad=True)
ro_y = torch.tensor([10.0], requires_grad=True)

# Stochastic Gradient Descent optimizer and assigns the learning rate
rosenbrock_optimizer_adam = optim.Adam([ro_x, ro_y], lr=adam_learning_rate)

# Lists to store the rosenbrock values for plotting
adam_rosenbrock_iterations = []
adam_rosenbrock_objective_values = []

#Optimization for Rosenbrock
for iteration in range(10000):

    # optimization??
    rosenbrock_optimizer_adam.zero_grad()
    # Computes the objective function value for rosenbrock
    objective = rosenbrock(ro_x, ro_y)
    objective.backward()
    rosenbrock_optimizer_adam.step()

    # saves the values of the objective function over 5000 iterations in a list
    adam_rosenbrock_iterations.append(iteration)
    adam_rosenbrock_objective_values.append(objective.item())

# Print final values
print(f"Rosenbrock values after Adam optimization: x: {ro_x.item()}, y: {ro_y.item()}, Objective: {objective.item()}")



# #BEALE FUNCTION!

# Initializes x and y to 10.0 and sets requires_grad=True when converting to tensor
be_x = torch.tensor([10.0], requires_grad=True)
be_y = torch.tensor([10.0], requires_grad=True)

# Stochastic Gradient Descent optimizer and assigns the learning rate
beale_optimizer_adam = optim.Adam([be_x, be_y], lr=adam_learning_rate)

# Lists to store the beale values for plotting
adam_beale_iterations = []
adam_beale_objective_values = []

#Optimization for Beale
for iteration in range(10000):
    # Computes the objective function value for beale
    objective = beale(be_x, be_y)

    # optimization??
    beale_optimizer_adam.zero_grad()
    objective.backward()
    beale_optimizer_adam.step()

    # saves the values of the objective function over 5000 iterations in a list
    adam_beale_iterations.append(iteration)
    adam_beale_objective_values.append(objective.item())

# Print final values
print(f"Beale values after Adam optimization: {be_x.item()}, y: {be_y.item()}, Objective: {objective.item()}")




# #ACKLEY FUNCTION!

# Initializes x and y to 10.0 and sets requires_grad=True when converting to tensor
ac_x = torch.tensor([10.0], requires_grad=True)
ac_y = torch.tensor([10.0], requires_grad=True)

# Stochastic Gradient Descent optimizer and assigns the learning rate
ackley_optimizer_adam = optim.Adam([ac_x, ac_y], lr=adam_learning_rate)

# Lists to store the ackley values for plotting
adam_ackley_iterations = []
adam_ackley_objective_values = []

#Optimization for Ackley
for iteration in range(10000):
    # Computes the objective function value for ackley
    objective = ackley(ac_x, ac_y)

    # optimization??
    ackley_optimizer_adam.zero_grad()
    objective.backward()
    ackley_optimizer_adam.step()

    # saves the values of the objective function over 5000 iterations in a list
    adam_ackley_iterations.append(iteration)
    adam_ackley_objective_values.append(objective.item())

# Print final values
print(f"Ackley values after Adam optimization: {ac_x.item()}, y: {ac_y.item()}, Objective: {objective.item()}")



###############################################################################################
# Comparing convergence rates

# Plot the previously stored values of the objective function over 5000 iterations for both SGD and Adam in a single plot.
###############################################################################################

'''
YOUR CODE HERE
'''

# Plots for Rosenbrock function
plt.figure(figsize=(10, 10))
plt.plot(sgd_rosenbrock_iterations, sgd_rosenbrock_objective_values, label='SGD')
plt.plot(adam_rosenbrock_iterations, adam_rosenbrock_objective_values, label='Adam')
plt.title('Rosenbrock Function Convergence Rates')
plt.xlabel('Number of Iterations')
plt.ylabel('Values of the objective function')
plt.legend()
plt.show()

# Plots for Beale function
plt.figure(figsize=(10, 10))
plt.plot(sgd_beale_iterations, sgd_beale_objective_values, label='SGD')
plt.plot(adam_beale_iterations, adam_beale_objective_values, label='Adam')
plt.title('Beale Function Convergence Rates')
plt.xlabel('Number of Iterations')
plt.ylabel('Values of the objective function')
plt.legend()
plt.show()

# Plots for Ackley function
plt.figure(figsize=(10, 10))
plt.plot(sgd_ackley_iterations, sgd_ackley_objective_values, label='SGD')
plt.plot(adam_ackley_iterations, adam_ackley_objective_values, label='Adam')
plt.title('Ackley Function Convergence Rates')
plt.xlabel('Number of Iterations')
plt.ylabel('Values of the objective function')
plt.legend()
plt.show()

