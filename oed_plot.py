import numpy as np
import matplotlib.pyplot as plt

def plot_loss(plot_dict, x_label, y_label, save_name):
    for key, val in plot_dict.items():
        x, y = val[0,:], val[1,:]
        plt.plot(x, y, label=key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_name[-4:] != ".png":
        save_name += ".png"
    plt.legend()
    plt.savefig(save_name, dpi=300)
    plt.close()

if __name__ == "__main__":
    to_load = [f"opt_{chr(i)}.txt" for i in range(ord('A'),ord('I'))]
    d_dict, ape_dict = {}, {}
    for file in to_load:
        d_load, ape_load = np.loadtxt(file, delimiter=", ", usecols=(0)), np.loadtxt(file, delimiter=", ", usecols=(1))
        d_dict[file] = np.vstack((np.array(range(0, len(d_load))), d_load))
        ape_dict[file] = np.vstack((np.array(range(0, len(ape_load))), ape_load))
    x_label = "Iteration Number"
    save_name = "d.png"
    y_label = "Design d"
    plot_loss(d_dict, x_label, y_label, save_name)
    save_name = "loss.png"
    y_label = "Loss"
    plot_loss(ape_dict, x_label, y_label, save_name)


