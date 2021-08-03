import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from gp_oed_surrogate.gp.gp_create import create_gp, save_gp, load_gp

def kernel(x_1, x_2, params):
    lengths = jnp.array([params[f"length_{i}"] for i in range(2)])
    inv_lengths = jnp.diag(lengths**(-1))
    ln_k_d = -0.5*(x_1 - x_2).T @ inv_lengths @ (x_1 - x_2) 
    return params["const"]*jnp.exp(ln_k_d)

def load_data(data_dir):
    # Import data from text file:
    training_data = np.loadtxt(data_dir)
    # Note that columns of beam data arranged as: [angle, c10, disp]
    # Rearrange to form: [c10, angle, x, y, z, disp]:
    training_data.T[[0,1]] = training_data.T[[1,0]]
    # Convert to Jax.numpy array:
    x_train = jnp.array(training_data[:,0:-1])
    y_train = jnp.array(training_data[:,-1])
    return (x_train, y_train)

def plot_gp_surface(d_pts, theta_pts, y_pts, save_name="plot.png"):
    # Check if supplied save name has .png extension:
    if save_name[-4:] != ".png":
        save_name += ".png"
    # Create surface plot:
    fig, ax = plt.subplots()
    contour_fig = ax.contourf(theta_pts, d_pts, y_pts, cmap='viridis')
    cbar = fig.colorbar(contour_fig)
    ax.set_xlabel('Beam stiffness c10')
    ax.set_ylabel('Angle of Beam in Degrees')
    cbar.set_label('Displacement', rotation=270, labelpad=15)
    # Save image of plot:
    fig.savefig(save_name, dpi=300)

if __name__ == "__main__":
    # Load training data:
    data_dir = "beam_data_one_pt.txt"
    x_train, y_train =  load_data(data_dir)
    constraints = {"length_0": {">": 10**-1, "<": 10**3}, 
                   "length_1": {">": 10**-1, "<": 10**3}, 
                   "const": {">": 10**-1, "<": 10**4}}
    surrogate = create_gp(kernel, x_train, y_train, constraints)

    # Plot surrogate model surface:
    d_pts, theta_pts = jnp.linspace(0, 5, 1000), jnp.linspace(0, 180, 1000)
    d_grid, theta_grid = jnp.meshgrid(theta_pts, d_pts)
    x = jnp.vstack((theta_grid.flatten(), d_grid.flatten())).T
    y_pts = surrogate.predict_mean(x)
    y_grid = y_pts.reshape(d_pts.size,theta_pts.size)
    plot_gp_surface(d_grid, theta_grid, y_grid, save_name="nonlinear_kernel")

    # Plot surface of training data:
    d_pts, theta_pts, y_pts = x_train[:,1], x_train[:,0], y_train
    theta_grid, d_grid, y_grid = theta_pts.reshape(10,10), d_pts.reshape(10,10), y_pts.reshape(10,10)      
    plot_gp_surface(d_grid, theta_grid, y_grid, save_name="training_data")

    # Save Gaussian process model:
    save_gp(surrogate, "nonlinear_kernel_gp")

    # Attempt to reload GP model:
    loaded_gp = load_gp("nonlinear_kernel_gp.json")
    # Check to make sure predictions are the same as before:
    loaded_y_pt = loaded_gp.predict_mean(x)
    y_pts = surrogate.predict_mean(x)
    print(f"Does loaded_y_pt == y_pt? {jnp.all(loaded_y_pt == y_pts)}")