import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Read the data from the Excel file
df = pd.read_excel('file_omega_Re.xlsx')
omega = df['omega']

# Define fixed values for parameters
nu = 1e-06
rho = 1000
r_o = 35 * 1e-03
u_Rm = 30
u_o_star = 10
k = 0.44
R = 70 * 1e-03
u = 1
L = 140 * 1e-03
delta_o = 11.6

# Radius ratio eta:
eta = r_o / R
H = (R - r_o)

# Calculate Re_c using the provided formula
df['Re_c'] = df['omega'] * r_o * (R - r_o) / nu

alpha_star_values = np.random.uniform(1e-5, 1e-3, size=10)
alpha_star_values_with_zero = np.concatenate(([0], alpha_star_values))

# Create a single figure with multiple subplots
fig, ax = plt.subplots(figsize=(12, 9))

# Initialize the 'G' column in the DataFrame
df['G'] = np.nan


# Calculate the Euclidean distance between the reference plot and the current plot
def calculate_distance(reference_plot, current_plot):
    distance = np.sqrt(np.sum((current_plot['G'] - reference_plot['G']) ** 2))
    return distance


# Plot the data for each alpha_star value
closest_distance = float('inf')
closest_alpha_star = None

# Plot the data for each alpha_star value
distances = {}
reference_alpha_star = 1e-4
reference_plot = df.copy()

# Plot the data for each alpha_star value
distances = {}
reference_alpha_star = 1e-4
reference_plot = df.copy()


# Define a function to calculate alpha_star
def calculate_alpha_star():
    return reference_alpha_star


# Define a function to calculate Rplus
def calculate_Rplus(G):
    Rplus = np.sqrt(G / (2 * np.pi))
    return Rplus


# Define a function to calculate D_i
def calculate_D_i(G, eta):
    alpha_star = calculate_alpha_star()
    D_i = 1 + (alpha_star / 2) * ((1 - eta) / eta) * np.sqrt(G / (2 * np.pi))
    return D_i


# Define a function to calculate lambda_val
def calculate_lambda_val(G, eta):
    D_i = calculate_D_i(G, eta)
    lambda_val = 11.6 * D_i ** 2
    return lambda_val


def calculate_delta_i(G, eta):
    D_i = calculate_D_i(G, eta)
    delta_i = 11.6 * D_i ** 3
    return delta_i


# Define a function to calculate 'a'
def calculate_a(G):
    delta_i = calculate_delta_i(G, eta)
    R_plus = calculate_Rplus(G)
    a_bounds = (0, 0.99)
    a = delta_i / R_plus
    return np.clip(a, *a_bounds)


# Define a function to calculate gamma
def calculate_gamma(G, k):
    lambda_val = calculate_lambda_val(G, eta)
    a = calculate_a(G)
    if (1 - a) <= 0 or a <= 0:
        return np.nan
    else:
        return (lambda_val - (1 / k) * (1 + (1 - a) * np.log(a / (1 - a)))) / (1 - a)


# Define a function to calculate norm_vel_centerline
def calculate_norm_vel_centerline(G, eta, k):
    gamma = calculate_gamma(G, k)
    lambda_val = calculate_lambda_val(G, eta)
    return 1 / k * (1 + (1 + eta) / 2 * math.log((1 - eta) / (1 + eta))) + gamma * (1 + eta) / 2


# Define a function to calculate norm_vel_innerbound
def calculate_norm_vel_innerbound(G, k, eta):
    a = calculate_a(G)
    norm_vel_centerline = calculate_norm_vel_centerline(G, eta, k)
    return norm_vel_centerline * (2 * eta / (1 + eta)) * (1 + a) + (1 / k) * (
                (-1 / eta) + (2 * (1 + a)) / (1 + eta) + (1 + a) / eta * np.log(((1 - eta) / (1 + eta)) * (1 + a / a)))


# Define the main equation
def main_equation(G, Re_c, eta):
    lambda_val = calculate_lambda_val(G, eta)
    norm_vel_innerbound = calculate_norm_vel_innerbound(G, k, eta)
    if G <= 0:
        return np.inf
    elif np.isnan(norm_vel_innerbound):
        return np.inf
    else:
        return (np.sqrt(2 * math.pi) / (1 - eta)) * (Re_c / np.sqrt(G)) - norm_vel_innerbound - (lambda_val / eta)


# Define bounds for G
G_bounds = (1, 1e+20)

G_list_reference = []
for Re_c in df['Re_c']:
    res = minimize_scalar(lambda G: abs(main_equation(G, Re_c, eta)), bounds=G_bounds, method='bounded')
    G_list_reference.append(res.x)

# Store the G values in the reference plot DataFrame
reference_plot = df.copy()
reference_plot['G'] = G_list_reference

# Plot the data
ax.scatter(df['Re_c'], df['G'], label=f'Reference $\\alpha_*$: {reference_alpha_star:.3e}')
ax.plot(df['Re_c'], reference_plot['G'])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Re_c')
ax.set_ylabel('G')

for alpha_star in alpha_star_values_with_zero:
    def calculate_alpha_star():
        return alpha_star


    def calculate_Rplus(G):
        Rplus = np.sqrt(G / (2 * np.pi))
        return Rplus


    def calculate_D_i(G, eta):
        alpha_star = calculate_alpha_star()
        D_i = 1 + (alpha_star / 2) * ((1 - eta) / eta) * np.sqrt(G / (2 * np.pi))
        return D_i


    def calculate_lambda_val(G, eta):
        D_i = calculate_D_i(G, eta)
        lambda_val = 11.6 * D_i ** 2
        return lambda_val


    def calculate_delta_i(G, eta):
        D_i = calculate_D_i(G, eta)
        delta_i = 11.6 * D_i ** 3
        return delta_i


    def calculate_a(G):
        delta_i = calculate_delta_i(G, eta)
        R_plus = calculate_Rplus(G)
        a_bounds = (0, 0.99)
        a = delta_i / R_plus
        return np.clip(a, *a_bounds)


    def calculate_gamma(G, k):
        lambda_val = calculate_lambda_val(G, eta)
        a = calculate_a(G)
        if (1 - a) <= 0 or a <= 0:
            return np.nan
        else:
            return (lambda_val - (1 / k) * (1 + (1 - a) * np.log(a / (1 - a)))) / (1 - a)


    def calculate_norm_vel_centerline(G, eta, k):
        gamma = calculate_gamma(G, k)
        lambda_val = calculate_lambda_val(G, eta)
        return 1 / k * (1 + (1 + eta) / 2 * math.log((1 - eta) / (1 + eta))) + gamma * (1 + eta) / 2


    def calculate_norm_vel_innerbound(G, k, eta):
        a = calculate_a(G)
        norm_vel_centerline = calculate_norm_vel_centerline(G, eta, k)
        return norm_vel_centerline * (2 * eta / (1 + eta)) * (1 + a) + (1 / k) \
            * ((-1 / eta) + (2 * (1 + a)) / (1 + eta) + (1 + a) / eta * np.log(((1 - eta) / (1 + eta)) * (1 + a / a)))


    def main_equation(G, Re_c, eta):
        lambda_val = calculate_lambda_val(G, eta)
        norm_vel_innerbound = calculate_norm_vel_innerbound(G, k, eta)
        if G <= 0:
            return np.inf
        elif np.isnan(norm_vel_innerbound):
            return np.inf
        else:
            return (np.sqrt(2 * math.pi) / (1 - eta)) * (Re_c / np.sqrt(G)) - norm_vel_innerbound - (lambda_val / eta)


    G_bounds = (1, 1e+20)

    G_list = []
    for Re_c in df['Re_c']:
        res = minimize_scalar(lambda G: abs(main_equation(G, Re_c, eta)), bounds=G_bounds, method='bounded')
        G_list.append(res.x)

    df['G'] = G_list

    # Calculate the Euclidean distance between the reference plot and the current plot
    distance = calculate_distance(reference_plot, df)
    distances[alpha_star] = distance

    ax.scatter(df['Re_c'], df['G'], label=f'$\\alpha_* $: {alpha_star:.3e}')
    ax.plot(df['Re_c'], df['G'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$Re_c$')
    ax.set_ylabel('$G$')

    distance = calculate_distance(reference_plot, df)
    distances[alpha_star] = distance

    if distance < closest_distance:
        closest_distance = distance
        closest_alpha_star = alpha_star
        df_closest_alpha_star = df.copy()

# Add a legend with the parameter table
legend = ax.legend(loc='upper left')
legend.get_frame().set_alpha(0.5)

# Create a table for parameters in the lower right corner
parameter_table = ax.table(cellText=[
    ['Parameter', 'Value'],
    ['$\\nu$', str(nu)],
    ['$\\rho$', str(rho)],
    ['$r_o$', str(r_o)],
    ['$u_Rm$', str(u_Rm)],
    ['R', str(R)],
    ['$\\eta$', str(eta)],
    ['H', str(H)],
],
    cellLoc='center',
    colLabels=None,
    cellColours=[['#f3f3f3', '#f3f3f3']] + [['white', 'white']] * 7,  # Color the header row differently
    loc='lower right'  # Moved to the lower right corner
)

# Style the parameter table
parameter_table.auto_set_font_size(False)
parameter_table.set_fontsize(10)
parameter_table.scale(1.2, 1.2)
parameter_table.auto_set_column_width([0, 1])

# Add a title and grid lines
ax.set_title('Optimal $\\alpha_{*}$ Selection')
ax.grid(True, linestyle='--', alpha=0.5)

print('The data frame which we deserve \n', df_closest_alpha_star)

closest_alpha_star = min(distances, key=distances.get)
closest_distance = distances[closest_alpha_star]

plt.tight_layout()
plt.show()
print(f"Closest alpha star: {closest_alpha_star:.3e}")


print ('closest_alpha_star', closest_alpha_star)
print( 'The data frame which we deserve \n', df_closest_alpha_star)


# Constant
const = closest_alpha_star * H / 2
print ('The alpha_star * H / 2 is' , const )

#START OF FUNCTIONS

# Function to calculate T
def calculate_T(G, rho, nu, L):
    T = G * rho * nu ** 2 * L
    return T
    print(T)

# Function to calculate tau_w
def calculate_tau_w(T, R, L):
    tau_w = T / (2 * np.pi * R ** 2 * L)
    return tau_w
    print(tau_w)

# Function to calculate u_star
def calculate_u_star(tau_w, rho):
    u_star = np.sqrt(tau_w / rho)
    return u_star
    print(u_star)


# Function to calculate D_0_star
def calculate_D_0_star(u_star, nu):
    D_0_star = 1 + const * u_star / nu
    return D_0_star

def calculate_f(f_value, D_0_star):
    # Define the objective function
    def objective(f_value):
        f_sqrt = np.sqrt(f_value)
        return abs(4 * np.log10(Re_c * f_sqrt) + 8.2 * D_0_star ** 2 - 8.6 - 12.2 * np.log10(D_0_star) - 1 / f_sqrt)

    # Minimize the objective function
    result = minimize_scalar(objective, method='bounded', bounds=(0, 10))  # Adjust the bounds if needed

    if result.success:
        f_value = result.x
    else:
        # Handle the case when the optimization fails
        f_value = None
        print("Optimization failed:", result.message)

    return f_value

def calculate_graph(Re_c, f_value):
    y_drag = 1 / np.sqrt(f_value)
    x_drag = Re_c * np.sqrt(f_value)
    return x_drag, y_drag

#END OF FUNCTIONS


# Iterate over the DataFrame rows and calculate f_value for each row
for index, row in df_closest_alpha_star.iterrows():
    Re_c = row['Re_c']
    G = row['G']
    # Calculate T, tau_w, and u_star
    T = calculate_T(G, rho, nu, L)
    tau_w = calculate_tau_w(T, R, L)
    u_star = calculate_u_star(tau_w, rho)

    # Calculate D_0_star
    D_0_star = calculate_D_0_star(u_star, nu)

    # Calculate f_value
    f_value = calculate_f(Re_c, D_0_star)
    #Calculate graph
    x_drag, y_drag = calculate_graph(Re_c, f_value)

  # Update the DataFrame with f_value
    df_closest_alpha_star.at[index, 'f_value'] = f_value
    df_closest_alpha_star.at[index, 'T'] = T
    df_closest_alpha_star.at[index, 'tau_w'] = tau_w
    df_closest_alpha_star.at[index, 'u_star'] = u_star
    df_closest_alpha_star.at[index, 'x_drag'] = x_drag
    df_closest_alpha_star.at[index, 'y_drag'] = y_drag

# Print the updated DataFrame
print('The updated DataFrame:')
print(df_closest_alpha_star)

#STYLE of plot


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your data)
x = df_closest_alpha_star['x_drag']
y = df_closest_alpha_star['y_drag']

# Create a figure with a custom background color
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#f2f2f2')  # Light gray background

# Customize the appearance of the plot
ax.set_xlabel('$\sqrt{Re_c} \cdot \sqrt{f}$', fontsize=14)  # Label with square root signs
ax.set_ylabel('$1/\sqrt{f}$', fontsize=14)  # Label with square root signs
ax.set_title(' Characteristic line of the drag reduction effect, $\sqrt{Re_c} \cdot \sqrt{f}$ vs $1/\sqrt{f}$ (Logarithmic Scale)', fontsize=16)  # Indicate logarithmic scale
ax.set_xscale('log')
ax.grid(True, linestyle='--', alpha=0.7)

# Add connecting lines between data points
for i in range(1, len(x)):
    ax.plot([x[i - 1], x[i]], [y[i - 1], y[i]], color='gray', linestyle='-', linewidth=1, zorder=1)

# Define a colormap for the scatter points
cmap = plt.get_cmap('viridis')  # You can choose any colormap you like
colors = np.linspace(0, 1, len(x))  # Generate colors based on data

# Create the scatter plot with colored markers (above the lines)
scatter = ax.scatter(x, y, c=colors, cmap=cmap, s=80, edgecolors='k', linewidths=0.5, zorder=2)


# Customize tick label fonts and sizes
ax.tick_params(axis='both', which='both', labelsize=12)

# Remove spines (optional)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a table with parameters and their values in the upper left corner
table_data = [('$\\alpha_*$', f'{closest_alpha_star:.3e}'),
              ['$\\alpha_*$ * R', f'{const:.3e}']]

table = ax.table(cellText=table_data, loc='upper left', cellLoc='center', colWidths=[0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# Show the plot
plt.tight_layout()
plt.show()

# Save DataFrame to Excel
output_filename = 'updated_data_frame.xlsx'
df_closest_alpha_star.to_excel(output_filename, index=False)



