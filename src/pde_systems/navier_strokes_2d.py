import pickle
from phi.jax.flow import *
from tqdm import trange
from pathlib import Path

from src.plotting.plotter import animate_quiver,animate_speed,custom_plot,sanity_check_vx,plot_curl
from src.utils.save_data import append_experiment


def get_numpy_vx_vy_p(v_trj_2d, p_trj_2d):
    vx = v_trj_2d.vector['x'].values.numpy('time,x,y')
    vy = v_trj_2d.vector['y'].values.numpy('time,x,y')
    p = p_trj_2d.values.numpy('time,x,y')
    return vx, vy, p

def slices_v(v, z_index=4):
    return v[{'z': z_index, 'vector': 'x,y'}]

def simulate_save_ns(
        initial_placeholder_velocity = (1, 0.0, 0.0),
        inflow_initial_velocity_x = 1.0,
        inflow_initial_velocity_y = 0.0,
        inflow_initial_velocity_z = 0.0,
        viscosity = 0.001,
        rel_tol = 1e-4,
        max_iterations_integrator = 10000,

        time_steps = 500,
        dt_step=1.0,

        domain_size_x = (0, 80),
        domain_size_y = (0, 20),
        domain_size_z = (0,5),
        grid_x = 256,
        grid_y = 64,
        grid_z = 8,
        cylinder_radius = 0.5,
        cylinder_y = 10,
        cylinder_x = 15,

        plot_folder = "plots",
        animate =False,

        chosen_z=4):
    save_data={}
    meta_data={}
    meta_data["inflow_velocity"] =inflow_initial_velocity_x
    meta_data["viscosity"] = viscosity
    meta_data["domain_size_x"] = domain_size_x
    meta_data["domain_size_y"] = domain_size_y
    meta_data["domain_size_z"] = domain_size_z
    meta_data["grid_x"] = grid_x
    meta_data["grid_y"] = grid_y
    meta_data["grid_z"] = grid_z
    meta_data["cylinder_radius"] = cylinder_radius
    meta_data["cylinder_y"] = cylinder_y
    meta_data["cylinder_x"] = cylinder_x
    meta_data["time_steps"] = time_steps
    meta_data["dt_step"] = dt_step

    inflow_initial_velocity = vec(x=inflow_initial_velocity_x,
                                  y=inflow_initial_velocity_y,
                                  z=inflow_initial_velocity_z)

    box_defined = Box(x=domain_size_x, y=domain_size_y, z=domain_size_z)
    cylinder = geom.infinite_cylinder(x=cylinder_x, y=cylinder_y, radius=cylinder_radius, inf_dim='z')

    boundary = {'x-': inflow_initial_velocity, 'x+': ZERO_GRADIENT, 'y': PERIODIC, 'z': PERIODIC}
    v_init = StaggeredGrid(initial_placeholder_velocity, boundary, x=grid_x, y=grid_y, z=grid_z, bounds=box_defined)
    v0, p0 = fluid.make_incompressible(v_init, cylinder, Solve(rel_tol=rel_tol,
                                                               max_iterations=max_iterations_integrator))
    reynolds_number = (inflow_initial_velocity_x*(cylinder_radius*2)) / viscosity

    reynolds_number_title = f'Reynolds number: {reynolds_number}: V={inflow_initial_velocity_x}, D={cylinder_radius*2}, nu={viscosity}'
    meta_data["reynolds_number"] = reynolds_number

    @jit_compile(forget_traces=True)
    def step(v, p, dt=1.0, viscosity=0.01):
        v = diffuse.explicit(v, viscosity, dt)
        v = advect.semi_lagrangian(v, v, dt)
        v, p = fluid.make_incompressible(v, cylinder, Solve(x0=p))
        return v, p

    v_trj, p_trj = iterate(step, batch(time=time_steps), v0, p0, dt=dt_step,viscosity=viscosity, range=trange)

    v_trj_2d = slices_v(v_trj)
    p_trj_2d = slices_v(p_trj)

    vx_np, vy_np, p_np = get_numpy_vx_vy_p(v_trj_2d, p_trj_2d)

    plot_folder_path = Path(plot_folder)
    simfolder_path = plot_folder_path / f'reynolds={int(reynolds_number)}_time={time_steps}_dt={dt_step}_grid={grid_x},{grid_y}'
    simfolder_path.mkdir(parents=True, exist_ok=True)
    plot_path = str(simfolder_path / f'Re_{int(reynolds_number)}_ns_field.jpg')
    animation_path = str(simfolder_path / f'Re_{int(reynolds_number)}_ns_ani.mp4')
    sim_data_path= str(simfolder_path / f'Re_{int(reynolds_number)}_ns_data.pkl')

    save_data["velocity_x"] = vx_np
    save_data["velocity_y"] = vy_np
    save_data["pressure"] = p_np
    save_data["vorticity"] = v_trj_2d.curl().values.numpy('time,x,y')

    # with open(sim_data_path, "wb") as f:
    #     pickle.dump(save_data, f)

    custom_plot(v_trj_2d, plot_path,title=reynolds_number_title)
    if animate:
        animate_speed(vx_np, vy_np, interval=40, save_path=animation_path,animation_title=reynolds_number_title)
    return save_data, meta_data



def run_navier_strokes_2d(reynolds_number,DATASTORE_FOLDER_NAME="dataset",NAVIER_STROKES_2D_H5_FILENAME="navier_strokes_2d_cylinder_wake.h5"):

    visc=1.0/reynolds_number
    sim_data,sim_meta_data=simulate_save_ns(
        initial_placeholder_velocity=(1, 0.0, 0.0),
        inflow_initial_velocity_x=1.0,
        inflow_initial_velocity_y=0.0,
        inflow_initial_velocity_z=0.0,
        viscosity=visc,
        rel_tol=1e-4,
        max_iterations_integrator=10000,
        time_steps=2000,
        dt_step=0.1,
        domain_size_x=(0, 20),
        domain_size_y=(0, 10),
        domain_size_z=(0, 5),
        grid_x=256,
        grid_y=128,
        grid_z=8,
        cylinder_radius=0.5,
        cylinder_y=5,
        cylinder_x=3,
        plot_folder="plots",
        animate=True
    )
    ret=append_experiment(folder_path=DATASTORE_FOLDER_NAME, filename=NAVIER_STROKES_2D_H5_FILENAME,
                      data_dict=sim_data,group_name=f"NS_{int(reynolds_number)}", extra_attrs=sim_meta_data)
    return sime_data