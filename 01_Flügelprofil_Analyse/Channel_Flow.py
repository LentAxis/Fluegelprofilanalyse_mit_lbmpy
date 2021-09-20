import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from lbmpy.session import *
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity
from lbmpy.parameterization import Scaling
from lbmpy.boundaries import NoSlip

import numpy as np
import os
import subprocess
import pandas as pd
from IPython.display import display
from IPython.display import display_html
from airfoils import Airfoil
from itertools import chain, cycle

class Obstacle_Airfoil:
    # Handles generation and transformation of the NACA airfoil.

    def __init__(self, NACA4_Number='0010', length=30, angle=0, domain=(100, 50), mid=(20, 25)):

        # Generate the airfoil via the NACA number and call the init_Airfoil() methode
        self.foil = Airfoil.NACA4(NACA4_Number)
        self.foil_upper = None
        self.foil_lower = None
        self.NACA4_Number = NACA4_Number
        self.length = length
        self.angle = angle
        self.mid = mid
        self.domain = domain

        self.init_Airfoil(length, mid, angle, (1 / (2 * length)))

    def init_Airfoil(self, length, mid, angle, resolution):

        # Convert angle in rad
        rotation = angle * np.pi / 180

        # Set X Array for the airfoil
        airfoil_x = np.arange(0, 1 + resolution, resolution, dtype=np.float64)

        # Get Y Values from the X Array
        airfoil_y_lower = np.array(self.foil.y_lower(airfoil_x), dtype=np.float64)
        airfoil_y_upper = np.array(self.foil.y_upper(airfoil_x), dtype=np.float64)

        # Combine X with Y coordinates for the upper and lower profile
        airfoil_upper = np.stack((airfoil_x - 0.5, airfoil_y_upper))
        airfoil_lower = np.stack((airfoil_x - 0.5, airfoil_y_lower))

        # Rotate and transform upper profile
        airfoil_upper_rotated = airfoil_upper.copy()
        airfoil_upper_rotated[0, :] = (airfoil_upper[0, :] * np.cos(rotation)
                                       + airfoil_upper[1, :] * np.sin(rotation))

        airfoil_upper_rotated[1, :] = (-airfoil_upper[0, :] * np.sin(rotation)
                                       + airfoil_upper[1, :] * np.cos(rotation))

        airfoil_upper_rotated[0, :] = airfoil_upper_rotated[0, :] * length + mid[0]
        airfoil_upper_rotated[1, :] = airfoil_upper_rotated[1, :] * length + mid[1]

        # Rotate and transform lower profile
        airfoil_lower_rotated = airfoil_lower.copy()
        airfoil_lower_rotated[0, :] = (airfoil_lower[0, :] * np.cos(rotation)
                                       + airfoil_lower[1, :] * np.sin(rotation))

        airfoil_lower_rotated[1, :] = (-airfoil_lower[0, :] * np.sin(rotation)
                                       + airfoil_lower[1, :] * np.cos(rotation))

        airfoil_lower_rotated[0, :] = airfoil_lower_rotated[0, :] * length + mid[0]
        airfoil_lower_rotated[1, :] = airfoil_lower_rotated[1, :] * length + mid[1]

        # Set final coordinates as global variable for upper and lower profile
        self.foil_upper = airfoil_upper_rotated
        self.foil_lower = airfoil_lower_rotated

    def get_parameters(self):
        return self.NACA4_Number, self.length, self.angle, self.mid

    def get_upper(self):
        return self.foil_upper

    def get_lower(self):
        return self.foil_lower

    def plot_domain(self):

        # Get coordinates from upper profile and plot them
        x1 = self.foil_upper[0]
        y1 = self.foil_upper[1]
        plt.plot(x1, y1)

        # Get coordinates from lower profile and plot them
        x2 = self.foil_lower[0]
        y2 = self.foil_lower[1]
        plt.plot(x2, y2)

        # Settings for the plot
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('Airfoil Profile')
        plt.xlim(0, self.domain[0])
        plt.ylim(0, self.domain[1])
        plt.gca().set_aspect('equal')
        plt.show()

class Channel_Flow:
    # Takes physical values and converts them into the lattice-room. The class handles the creation of the lbmpy
    # simulation channel and also has some methods for analysing the airfoil.

    def __init__(self, target="cpu", method="cumulant", length=0.5, height=0.1, length_per_cell=0.005,
                 length_airfoil=0.05, velocity=0.1, density=1.204, viscosity=1.516e-5):

        self.airfoil = None
        self.mid = None

        # Conversion with a fixed lattice velocity
        self.lattice_velocity = 0.1
        self.velocity_physic = velocity
        
        # Calculate characteristic length corresponding to airfoil length
        # (important for calculating the right reynolds number)
        self.cells_per_length = round(length_airfoil / length_per_cell)
        self.length_airfoil = length_airfoil

        # Convert the physical values in the lattice-Room
        self.scaling_physic = Scaling(physical_length=length_per_cell*self.cells_per_length,
                                      physical_velocity=velocity,
                                      kinematic_viscosity=viscosity,
                                      cells_per_length=self.cells_per_length)

        # Get lattice values with fixed velocity and set conversion factors (Krüger S.274)
        self.scaling_lattice = self.scaling_physic.fixed_lattice_velocity_scaling(self.lattice_velocity)
        self.Cu = velocity / self.lattice_velocity
        self.Cp = density
        self.Cl = self.scaling_physic.dx
        self.Ct = self.Cl / self.Cu
        self.CF = self.Cp * self.Cl / (self.Ct**2)

        # Set domain size
        self.domain_size = (round(length / self.scaling_physic.dx), round(height / self.scaling_physic.dx))
        
        # Prepare pysical parameter display
        bezeichnung = ['Länge Kanal:', 'Höhe Kanal:', 'Distanz pro Zelle:', 'Geschwindigkeit:',
                       'Viskosität:', 'Reynolds Zahl:']
        einheit = ['m', 'm', 'm', 'm/s', 'm^2/s', '']
        wert = [length, height, self.scaling_physic.dx, "%.3f" % velocity, viscosity,
                "%.3f" % self.scaling_physic.reynolds_number]
        d = {'Bezeichnung': bezeichnung, 'Wert': wert, 'Einheit': einheit}
        df_physic = pd.DataFrame(data=d)

        # Prepare lattice parameter display
        bezeichnung = ['Länge Kanal:', 'Höhe Kanal:', 'Geschwindigkeit:', 'Relaxationszeit:']
        wert = [self.domain_size[0], self.domain_size[1], "%.3f" % self.lattice_velocity,
                self.scaling_lattice.relaxation_rate]
        d = {'Bezeichnung': bezeichnung, 'Wert': wert}
        df_lattice = pd.DataFrame(data=d)

        # Display prepared dataframes
        self.display_side_by_side(df_physic, df_lattice, titles=['Pysikalische Grössen:', 'Lattice Grössen:'])

        # Parameter Check. Wait for user input
        check = input("Sind die Parameter zufriedenstellend? [ [y] / n ]")
        if check == "n":
            print("Szenario wurde nicht erstellt.")
            return
        print("Erstelle Szenario...")

        # Create variables for masks and helpful addons
        self.mask = None
        self.density_mask_upper = None
        self.density_mask_lower = None
        self.airfoil_area = None

        # Create channel scenario with user input
        optimization = {'target': target}
        self.scenario = create_channel(domain_size=self.domain_size,
                                       u_max=self.lattice_velocity,
                                       compressible=True,
                                       wall_boundary=NoSlip(),
                                       method=method,
                                       relaxation_rate=self.scaling_lattice.relaxation_rate,
                                       optimization=optimization)

        print("Szenario wurde erfolgreich erstellt.")

    def set_obstacle(self, NACA4_Number='0010', angle=0, loc_horizontal=3):

        # Calculate position of obstacle
        self.mid = (self.domain_size[0] // loc_horizontal, self.domain_size[1] // 2)

        # Create a new airfoil and plot the domain
        self.airfoil = Obstacle_Airfoil(NACA4_Number=NACA4_Number, length=self.cells_per_length, angle=angle,
                                        domain=self.domain_size, mid=self.mid)
        self.airfoil.plot_domain()

        # Parameter Check. Wait for user input
        check = input("Soll dieses Profil in das Scenario eingebaut werden? [ [y] / n ]")
        if check == "n":
            print("Profil wurde verworfen.")
            return
        print("Profil wird als Hindernis eingebaut...")

        # Update the obstacle mask with the airfoil we just added. The mask can now be used for the boundary_handling
        self.update_mask()
        self.scenario.boundary_handling.set_boundary(NoSlip("obstacle"), mask_callback=self.get_airfoil_mask)

        print("Profil wurde erfolgreich eingebaut.")

    def get_airfoil_mask(self, x, y, *_):
        return self.mask

    def update_obstacle_angle(self, angle_inc=0.0):

        # remove current obstacle and get the parameters
        self.scenario.boundary_handling.set_boundary("domain", mask_callback=self.get_airfoil_mask)
        NACA4_Number, length, angle_current, mid = self.airfoil.get_parameters()

        # Add the input value to the current angle
        angle_next = angle_current + angle_inc

        # Create new airfoil with the updated angle, update the obstacle mask and call the boundary_handling
        self.airfoil = Obstacle_Airfoil(NACA4_Number=NACA4_Number, length=length, angle=angle_next, mid=mid)
        self.update_mask()
        self.scenario.boundary_handling.set_boundary(NoSlip("obstacle"), mask_callback=self.get_airfoil_mask)

    def update_mask(self):

        # Reset masks (the +2 comes from the way lbmpy builds their domains)
        self.mask = np.zeros((self.domain_size[0] + 2, self.domain_size[1] + 2), dtype=bool)
        self.density_mask_upper = self.mask.copy()
        self.density_mask_lower = self.mask.copy()

        # Get the coordinates from the airfoil profile
        airfoil_upper = self.airfoil.get_upper()
        airfoil_lower = self.airfoil.get_lower()

        # Create x and y matrix, which is used by lbmpy for boundary handling
        x = np.ones((self.domain_size[0] + 2, self.domain_size[1] + 2))
        y = np.ones((self.domain_size[0] + 2, self.domain_size[1] + 2))

        # We recreate the return values for the boundary_handling mask_callback function
        inc = -0.5
        for i in range(self.domain_size[0] + 2):
            x[i, :] *= inc
            inc += 1

        inc = -0.5
        for i in range(self.domain_size[1] + 2):
            y[:, i] *= inc
            inc += 1

        # Used for upper and lower hull. This will be used to get the density next to the obstacle
        min_airfoil_x = min(airfoil_upper[0, :])
        max_airfoil_x = max(airfoil_upper[0, :])
        max_airfoil_y = 0
        min_airfoil_y = self.domain_size[1]

        airfoil_x_length = max_airfoil_x - min_airfoil_x
        airfoil_x_mid = (airfoil_x_length / 2) + min_airfoil_x

        # Overlap the airfoil with the x and y matrix and search for the nearest values.
        # Those values will be set as true in our mask
        for j, value_x in zip(range(len(airfoil_upper[0])), airfoil_upper[0, :]):

            # Search for the index in x, which is closest to the current x-value in our profile
            idx_x = np.abs((x[:, 0] - value_x)).argmin()

            # Get the corresponding y-values for our current x-value from the profile
            value_y_upper = airfoil_upper[1, j]
            value_y_lower = airfoil_lower[1, j]

            # Search for the indices in y, which are closest to the current y-values in our profile
            idx_y_upper = np.abs((y[0, :] - value_y_upper)).argmin()
            idx_y_lower = np.abs((y[0, :] - value_y_lower)).argmin()

            # update the mask
            self.mask[idx_x, idx_y_lower:idx_y_upper] = True

            # Set the cells next to the profile as true in the density mask
            if (x[idx_x, 0] - airfoil_x_mid) <= -(airfoil_x_length / 6):
                self.density_mask_upper[idx_x - 1, idx_y_upper + 2] = True
                self.density_mask_lower[idx_x - 1, idx_y_lower - 2] = True
            elif (x[idx_x, 0] - airfoil_x_mid) <= -(airfoil_x_length / 6):
                self.density_mask_upper[idx_x + 1, idx_y_upper + 2] = True
                self.density_mask_lower[idx_x + 1, idx_y_lower - 2] = True
            else:
                self.density_mask_upper[idx_x, idx_y_upper + 2] = True
                self.density_mask_lower[idx_x, idx_y_lower - 2] = True

            # Check for highest and lowest point for the airfoil Area
            max_airfoil_y = max(max_airfoil_y, idx_y_upper)
            min_airfoil_y = min(min_airfoil_y, idx_y_lower)

        # Saves coordinates for a rectangle, which contains the airfoil (used for the density plot)
        self.airfoil_area = [min_airfoil_x, max_airfoil_x, min_airfoil_y, max_airfoil_y]

    def update_scenario(self, scenario):
        self.scenario = scenario

    def plot_velocity(self):
        
        # With Airfoil
        if self.airfoil:
            NACA4_Number, length, angle_current, mid = self.airfoil.get_parameters()
            plt.figure(figsize=(11, self.domain_size[1]/(self.domain_size[0]/10)), dpi=100)
            plt.vector_field_magnitude(self.scenario.velocity[:, :] * self.Cu)
            plt.title("NACA {0} | Profillänge: {1}m | Angriffswinkel: {2}°".format(NACA4_Number,
                      (length * self.scaling_physic.dx), angle_current))
            plt.colorbar().set_label("Geschwindigkeit [m/s]")

        # Without Airfoil
        else:
            plt.figure(figsize=(11, self.domain_size[1]/(self.domain_size[0]/10)), dpi=100)
            plt.vector_field_magnitude(self.scenario.velocity[:, :] * self.Cu)
            plt.title("Kanalfluss | Reynoldszahl: {0: 0.2f}".format(self.scaling_physic.reynolds_number))
            plt.colorbar().set_label("Geschwindigkeit [m/s]")

    def plot_velocity_with_force(self):

        # Add an Arrow to represent the direction of the lift and drag forces combined
        NACA4_Number, length, angle_current, mid = self.airfoil.get_parameters()
        force = self.scenario.boundary_handling.force_on_boundary(NoSlip("obstacle"))
        force_hat = force / np.linalg.norm(force)

        plt.figure(figsize=(11, self.domain_size[1]/(self.domain_size[0]/10)), dpi=100)
        plt.vector_field_magnitude(self.scenario.velocity[:, :] * self.Cu)
        ax1 = plt.axes()
        ax1.arrow(self.mid[0], self.mid[1],
                 force_hat[0] * (self.domain_size[1] / 3),
                 force_hat[1] * (self.domain_size[1] / 3),
                 head_width=(self.domain_size[1] / 8), head_length=(self.domain_size[1] / 6),
                 fc='black', ec='red')
        plt.title("NACA {0} | Profillänge: {1}m | Angriffswinkel: {2}° | Drag: {3: 0.3f}N | Lift: {4: 0.3f}N"
                  .format(NACA4_Number, (length * self.scaling_physic.dx), angle_current, force[0], force[1]))
        plt.colorbar().set_label("Geschwindigkeit [m/s]")

    def plot_forces_against_eachother(self, max_angle=20, steps=100, iter=300):

        # Create array for drag and lift
        drag = []
        lift = []

        # Calculate the angle step size with the number of iterations
        angle_inc = max_angle / iter

        # Get airfoil parameters for modifying
        NACA4_Number, length, angle_current, mid = self.airfoil.get_parameters()

        # Mainloop changes the rotation of the airfoil, runs the scenario until stable and saves the forces
        for i in range(iter):
            self.update_obstacle_angle(angle_inc=angle_inc)
            self.scenario.run(steps)
            force = self.scenario.boundary_handling.force_on_boundary(NoSlip("obstacle"))
            drag.append(force[0])
            lift.append(force[1])

        # Get parameter for the plot title
        NACA4_Number, length, angle_after, mid = self.airfoil.get_parameters()

        # Plot settings
        plt.plot(drag, lift, marker='D')
        plt.grid()
        plt.title("Drag vs. Lift Kraft von {0: 0.2f}° bis {1: 0.2f}°".format(angle_current, angle_after))
        plt.xlabel("Drag Kraft [N]")
        plt.ylabel("Lift Kraft [N]")
        
    def plot_forces_with_angle_change(self, max_angle=20, steps=100, iter=300):

        # Create array for drag, lift and angles
        drag = []
        lift = []
        angle = []

        # Calculate the angle step size with the number of iterations
        angle_inc = max_angle / iter

        # Mainloop changes the rotation of the airfoil, runs the scenario until stable and saves the forces and angles
        for i in range(iter):
            self.update_obstacle_angle(angle_inc=angle_inc)
            NACA4_Number, length, angle_current, mid = self.airfoil.get_parameters()
            self.scenario.run(steps)
            force = self.scenario.boundary_handling.force_on_boundary(NoSlip("obstacle"))
            drag.append(force[0])
            lift.append(force[1])
            angle.append(angle_current)

        # Plot settings
        plt.plot(angle, drag, label="Drag")
        plt.plot(angle, lift, label="Lift")
        plt.grid()
        plt.title("Drag und Lift Kräfte")
        plt.xlabel("Angriffswinkel [°]")
        plt.ylabel("Kraft [N]")
        plt.legend(loc="upper left")

    def plot_density(self, figsize=(10, 20), dpi=100):

        # Get all indices with the value true
        density_idx_upper = np.transpose(self.density_mask_upper.nonzero())
        density_idx_lower = np.transpose(self.density_mask_lower.nonzero())

        density_upper = []
        density_lower = []

        # Get the density date from the simulation and convert it to physical values
        d = self.scenario.density[:, :].data * self.Cp

        # Create an array with the initial density for a comparison
        d_norm = np.ones(len(density_idx_upper)) * self.Cp

        # Extract the densities at the defined indices for the upper and lower hull
        for i in range(len(density_idx_upper)):
            density_upper.append(d[density_idx_upper[i][0]][density_idx_upper[i][1]])

        for i in range(len(density_idx_lower)):
            density_lower.append(d[density_idx_lower[i][0]][density_idx_lower[i][1]])

        # Create a rectangle around the airfoil for the plot
        x_min = int(self.airfoil_area[0] - 50)
        x_max = int(self.airfoil_area[1] + 50)
        y_min = int(self.airfoil_area[2] - 50)
        y_max = int(self.airfoil_area[3] + 50)

        # Set the density values of the airfoil to nan. This gives a better contrast for the plot
        density = np.where(self.mask[1:self.domain_size[0] + 1, 1:self.domain_size[1] + 1] == False, d, np.nan)
        
        plt.figure(figsize=figsize, dpi=dpi)

        # Heat plot of the area around the airfoil
        plt.subplot(2, 1, 1)
        plt.title("Dichteverteilung um Flügelprofil")
        plt.scalar_field(density[x_min:x_max, y_min:y_max])
        plt.colorbar()

        # Upper Profile Plot
        plt.subplot(2, 2, 3)
        plt.title("Oberes Profil")
        plt.ylabel("Dichte [kg/m^3]")
        plt.plot(density_upper)
        plt.plot(d_norm, color='green', linestyle='dashed')

        # Lower Profile Plot
        plt.subplot(2, 2, 4)
        plt.title("Unteres Profil")
        plt.ylabel("Dichte [kg/m^3]")
        plt.plot(density_lower)
        plt.plot(d_norm, color='green', linestyle='dashed')

        # Format settings
        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=1, 
                            top=2, 
                            wspace=0.5, 
                            hspace=0.5)    

    def display_side_by_side(self, *args, titles=cycle([''])):

        # Used by pandas dataframe to display tables next to each other
        html_str = ''
        for df, title in zip(args, chain(titles, cycle(['</br>']))):
            html_str += '<th style="text-align:center"><td style="vertical-align:top">'
            html_str += f'<h2>{title}</h2>'
            html_str += df.to_html().replace('table', 'table style="display:inline"')
            html_str += '</td></th>'
        display_html(html_str, raw=True)

class xfoil:
    # This class acts as an interface for the xFoil application (https://github.com/JARC99/Youtube)

    def __init__(self, airfoil="NACA0012", alpha_i=0, alpha_f=10, alpha_step=0.5, Re=100000, n_iter=100):

        # User Input
        self.airfoil = airfoil
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.alpha_step = alpha_step
        self.Re = Re
        self.n_iter = n_iter

    def get_polar_data(self):

        # Define the paths for executable, input and output files
        script_dir = os.path.dirname(__file__)
        polar_path = os.path.join(script_dir, "{0}_polar_file.txt".format(self.airfoil))
        coord_path = os.path.join(script_dir, "{0}_coordinates.txt".format(self.airfoil))
        input_path = os.path.join(script_dir, "foildata\\input_file.in")
        exe_path = os.path.join(script_dir, "foildata\\xfoil.exe")

        if os.path.exists(polar_path):
            os.remove(polar_path)

        # Create a text file with the commands for xFoil
        input_file = open(input_path, 'w')
        input_file.write("LOAD {0}.dat\n".format(self.airfoil))
        input_file.write(self.airfoil + '\n')
        input_file.write("PSAVE {0}_coordinates.txt\n".format(self.airfoil))
        input_file.write("PPAR\n")
        input_file.write("N 200\n")
        input_file.write("T 1\n\n\n")
        input_file.write("OPER\n")
        input_file.write("Visc {0}\n".format(self.Re))
        input_file.write("PACC\n")
        input_file.write("{0}_polar_file.txt\n\n".format(self.airfoil))
        input_file.write("ITER {0}\n".format(self.n_iter))
        input_file.write("ASeq {0} {1} {2}\n".format(self.alpha_i, self.alpha_f, self.alpha_step))
        input_file.write("\n\n")
        input_file.write("quit\n")
        input_file.close()

        # Run the commands on xFoil
        subprocess.call(exe_path + "<" + input_path, shell=True)

        # Load the created output files in variables
        polar = np.loadtxt(polar_path, skiprows=12)
        coord = np.loadtxt(coord_path)
        return polar, coord
        
    def plot_polar_data(self):
        polar, coord = self.get_polar_data()
        
        fig = plt.figure()

        # Plot airfoil profile
        ax = fig.add_subplot(311)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid()
        ax.plot(coord[:, 0], coord[:, 1])
        ax.set_title("{0} von {1}° bis {2}° ".format(self.airfoil, self.alpha_i, self.alpha_f))
        ax.set_aspect('equal')

        # Plot lift coefficient
        ax = fig.add_subplot(334)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title("CL")
        ax.set_xlabel("Angriffswinkel")
        ax.grid()
        ax.plot(polar[:, 0], polar[:, 1])

        # Plot drag coefficient
        ax = fig.add_subplot(335)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title("CD")
        ax.set_xlabel("Angriffswinkel")
        ax.grid()
        ax.plot(polar[:, 0], polar[:, 2])

        # Plot drag-pressure coefficient
        ax = fig.add_subplot(336)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title("CDp")
        ax.set_xlabel("Angriffswinkel")
        ax.grid()
        ax.plot(polar[:, 0], polar[:, 3])

        # Plot moment coefficient
        ax = fig.add_subplot(337)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title("CM")
        ax.set_xlabel("Angriffswinkel")
        ax.grid()
        ax.plot(polar[:, 0], polar[:, 4])

        # Plot top contour flow
        ax = fig.add_subplot(338)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_title("Top_Xtr")
        ax.set_xlabel("Angriffswinkel")
        ax.set_ylabel("Laminarität über Profillinie [%]")
        ax.grid()
        ax.plot(polar[:, 0], polar[:, 5] * 100)

        # Plot bottom contour flow
        ax = fig.add_subplot(339)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_title("Bot_Xtr")
        ax.set_xlabel("Angriffswinkel")
        ax.set_ylabel("Laminarität über Profillinie [%]")
        ax.grid()
        ax.plot(polar[:, 0], polar[:, 6] * 100)

        # Plot settings
        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=1, 
                            top=2, 
                            wspace=0.2, 
                            hspace=0.5)    
                            
    def plot_CL_to_CD(self):

        # Plot the lift and drag coefficient against each other
        polar, coord = self.get_polar_data()

        plt.title("CL und CD")
        plt.xlabel("CD")
        plt.ylabel("CL")
        plt.grid()
        plt.plot(polar[:, 2], polar[:, 1])