import numpy as np
import matplotlib.pyplot as plt
import os


def design_and_save_curve(theta_points, phi_points, filename="sphere_plot"):
    # 1. Ensure the directory exists
    save_path = "riemannian_score_sde/visualisations/plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 2. Map to Extrinsic (3D) Coordinates
    # x = sin(th)cos(ph), y = sin(th)sin(ph), z = cos(th)
    x = np.sin(theta_points) * np.cos(phi_points)
    y = np.sin(theta_points) * np.sin(phi_points)
    z = np.cos(theta_points)

    # 3. Plotting logic
    fig = plt.figure(figsize=(12, 5))

    # --- Intrinsic Space ---
    ax1 = fig.add_subplot(121)
    ax1.plot(phi_points, theta_points, 'b-', linewidth=2)
    ax1.set_title(f'Intrinsic: {filename}')
    ax1.set_xlabel('$\phi$')
    ax1.set_ylabel('$\\theta$')
    ax1.grid(True, linestyle='--')
    ax1.invert_yaxis()

    # --- Manifold ---
    ax2 = fig.add_subplot(122, projection='3d')

    # Contextual Wireframe
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    ax2.plot_wireframe(np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v),
                       color="lightgray", alpha=0.3)

    ax2.plot(x, y, z, 'r-', linewidth=3)
    ax2.set_title('3D Manifold Curve')

    # 4. Save to disk
    full_path = os.path.join(save_path, f"{filename}.png")
    plt.savefig(full_path, dpi=150)
    plt.close(fig)  # Close to save memory during batch processing
    print(f"Saved: {full_path}")


# --- Design Session ---

# # Design 1: The "Loxodrome" (Spiral)
# t = np.linspace(0, 1, 500)
# theta_spiral = np.linspace(0.1, np.pi - 0.1, 500)
# phi_spiral = np.linspace(0, 8 * np.pi, 500)
# design_and_save_curve(theta_spiral, phi_spiral, "loxodrome_spiral")

# Design 2: A small circle around the North Pole
theta_pole = np.full(100, np.pi/2)
phi_pole = np.linspace(0, np.pi, 100)
design_and_save_curve(theta_pole, phi_pole, "horizontal_circle plot")

# Design 3. Define a simple, short range for theta and phi
n_points = 50
theta_small = np.linspace(0.5, 1.0, n_points)
phi_small = np.linspace(0.5, 1.5, n_points)
design_and_save_curve(theta_small, phi_small, "small_intrinsic_curve")

# --- Design Session: Simple Quadratic Arc ---
phi_arc = np.linspace(1.0, 3.0, 100)
t_relative = np.linspace(-1, 1, 100)
theta_arc = 1.0 + 0.5 * (t_relative**2)
design_and_save_curve(theta_arc, phi_arc, "small_quadratic_curve")