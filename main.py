from src.pde_systems.navier_strokes_2d import  simulate_save_ns
from src.utils.save_data import h5_create_file , h5_load_file,append_experiment
from src.plotting.plotter import animate_speed

DATASTORE_FOLDER_NAME = "dataset"
# NAVIER_STROKES_2D_H5_FILENAME = "navier_strokes_2d_cylinder_wake.h5"

if __name__ == '__main__':

    STATE = "sim"
    NAVIER_STROKES_2D_H5_FILENAME = "navier_strokes_2d_cylinder_wake.h5"

    re=300

    re_list=[100,200,300,400,500,600,700,800,900,1000]
    for each_re in re_list:
        visc=1.0/each_re
        sim_data,sim_meta_data=simulate_save_ns(
            initial_placeholder_velocity=(1, 0.0, 0.0),
            inflow_initial_velocity_x=1.0,
            inflow_initial_velocity_y=0.0,
            inflow_initial_velocity_z=0.0,
            viscosity=visc,
            rel_tol=1e-4,
            max_iterations_integrator=10000,
            time_steps=2000,
            dt_step=0.2,
            domain_size_x=(0, 20),
            domain_size_y=(0, 10),
            domain_size_z=(0, 5),
            grid_x=128,
            grid_y=128,
            grid_z=8,
            cylinder_radius=0.5,
            cylinder_y=5,
            cylinder_x=3,
            plot_folder="plots",
            animate=True
        )
        ret=append_experiment(folder_path=DATASTORE_FOLDER_NAME, filename=NAVIER_STROKES_2D_H5_FILENAME,
                          data_dict=sim_data,group_name=f"NS_{int(re)}", extra_attrs=sim_meta_data)

    # elif STATE == "animate":
    #     loaded_data = h5_load_file(folder_path=DATASTORE_FOLDER_NAME, filename=NAVIER_STROKES_2D_H5_FILENAME)
    #     print(loaded_data.keys())
    #     data_dict=loaded_data[f"NS_{int(re)}"]
    #     print(data_dict.keys())
    #     vx=data_dict["velocity_x"][700:900,50:125,:]
    #     vy=data_dict["velocity_y"][700:900,50:125,:]
    #     animate_speed(vx, vy, interval=50, save_path=f"plots/navier_strokes_2d_cylinder_wake_re{int(re)}_small.gif")
    # # Confirm
    #
