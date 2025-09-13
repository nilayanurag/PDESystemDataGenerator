from src.pde_systems.navier_strokes_2d import simulate_save_ns, run_navier_strokes_2d
from src.utils.save_data import h5_create_file , h5_load_file,append_experiment
from src.plotting.plotter import animate_speed
import typer

DATASTORE_FOLDER_NAME = "dataset"
NAVIER_STROKES_2D_H5_FILENAME = "navier_strokes_2d_cylinder_wake.h5"

def generate_pde_system(
        system_name: str = typer.Option("navier_strokes_2d", help="Name of the PDE system to generate"),
        system_variant: str = typer.Option("cylinder_wake", help="Variant of the PDE system"),
        reynolds_number: int = typer.Option(100, help="Reynolds number for the simulation"),
        seed: int = typer.Option(42, help="Random seed for reproducibility"),
        log_dir: str = typer.Option(DATASTORE_FOLDER_NAME, help="Directory to save logs"),
):
    if system_name == "navier_strokes_2d" and system_variant == "cylinder_wake":
        sim_data = run_navier_strokes_2d(reynolds_number, DATASTORE_FOLDER_NAME, NAVIER_STROKES_2D_H5_FILENAME)
        typer.echo(f"Simulation data for Reynolds number {reynolds_number} saved successfully.")

if __name__ == '__main__':
    typer.run(generate_pde_system)