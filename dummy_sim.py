import subprocess
import shutil
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def run_gprmax_simulation_tool(input_file: str) -> str:
    """
    Run gprMax simulation with the given input file.
    
    First checks if gprMax is installed. If not, clones the repository,
    installs it using conda, and then runs the simulation.
    
    Args:
        input_file: Path to the gprMax input file (.in file)
    
    Returns:
        str: Success message or error message
    """
    logger.info(f"[TOOL CALL] run_gprmax_simulation_tool - Running simulation with input file: {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        error_msg = f"Input file not found: {input_file}"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg
    
    # Check if gprMax is installed by trying to run it as a module
    gprmax_installed = False
    try:
        # Try to execute the module (it may show usage/help or error, but module exists)
        result = subprocess.run(
            ["python", "-m", "gprMax"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Check stderr for "No module named" error
        if result.stderr and ("No module named" in result.stderr or "ModuleNotFoundError" in result.stderr):
            logger.debug("gprMax module not found")
            gprmax_installed = False
        else:
            # Module exists (may have other errors, but module is installed)
            gprmax_installed = True
            logger.debug("gprMax module found")
    except subprocess.TimeoutExpired:
        logger.debug("gprMax check timed out")
        gprmax_installed = False
    except FileNotFoundError:
        logger.debug("python command not found")
        gprmax_installed = False
    except Exception as e:
        logger.debug(f"gprMax check error: {e}")
        gprmax_installed = False
    
    if not gprmax_installed:
        logger.info("gprMax not found. Installing gprMax...")
        
        # Check if conda is available
        conda_available = shutil.which("conda") is not None
        if not conda_available:
            error_msg = "conda is not available. Please install conda to use gprMax."
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            return error_msg
        
        # Clone repository if it doesn't exist
        gprmax_dir = Path("gprMax")
        if not gprmax_dir.exists():
            logger.info("Cloning gprMax repository...")
            try:
                clone_result = subprocess.run(
                    ["git", "clone", "https://github.com/gprMax/gprMax.git"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout for clone
                )
                if clone_result.returncode != 0:
                    error_msg = f"Failed to clone gprMax repository: {clone_result.stderr}"
                    logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                    return error_msg
                logger.info("Successfully cloned gprMax repository")
            except subprocess.TimeoutExpired:
                error_msg = "Timeout while cloning gprMax repository"
                logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                return error_msg
            except Exception as e:
                error_msg = f"Error cloning gprMax repository: {str(e)}"
                logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                return error_msg
        
        # Change to gprMax directory
        original_dir = os.getcwd()
        try:
            os.chdir(gprmax_dir)
            
            # Check if conda environment already exists
            env_exists_result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True
            )
            env_exists = "gprMax" in env_exists_result.stdout
            
            if not env_exists:
                logger.info("Creating conda environment from conda_env.yml...")
                # Create conda environment
                env_result = subprocess.run(
                    ["conda", "env", "create", "-f", "conda_env.yml"],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes timeout for environment creation
                )
                if env_result.returncode != 0:
                    error_msg = f"Failed to create conda environment: {env_result.stderr}"
                    logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                    os.chdir(original_dir)
                    return error_msg
                logger.info("Successfully created conda environment")
            
            # Build and install gprMax
            logger.info("Building and installing gprMax...")
            
            # Use conda run to execute commands in the gprMax environment
            build_result = subprocess.run(
                ["conda", "run", "-n", "gprMax", "python", "setup.py", "build"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for build
            )
            if build_result.returncode != 0:
                error_msg = f"Failed to build gprMax: {build_result.stderr}"
                logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                os.chdir(original_dir)
                return error_msg
            
            install_result = subprocess.run(
                ["conda", "run", "-n", "gprMax", "python", "setup.py", "install"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for install
            )
            if install_result.returncode != 0:
                error_msg = f"Failed to install gprMax: {install_result.stderr}"
                logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                os.chdir(original_dir)
                return error_msg
            
            logger.info("Successfully installed gprMax")
            
            # Return to original directory
            os.chdir(original_dir)
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Timeout during gprMax installation: {str(e)}"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            os.chdir(original_dir)
            return error_msg
        except Exception as e:
            error_msg = f"Error during gprMax installation: {str(e)}"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            os.chdir(original_dir)
            return error_msg
    
    # Run the simulation
    logger.info(f"Running gprMax simulation with input file: {input_file}")
    try:
        # Use absolute path for input file
        abs_input_file = os.path.abspath(input_file)
        
        # Check if we need to use conda run (if we just installed it)
        if not gprmax_installed:
            # Run using conda environment
            sim_result = subprocess.run(
                ["conda", "run", "-n", "gprMax", "python", "-m", "gprMax", abs_input_file],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for simulation
            )
        else:
            # Run directly
            sim_result = subprocess.run(
                ["python", "-m", "gprMax", abs_input_file],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for simulation
            )
        
        if sim_result.returncode != 0:
            error_msg = f"gprMax simulation failed:\nSTDOUT: {sim_result.stdout}\nSTDERR: {sim_result.stderr}"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            return error_msg
        
        success_msg = f"Successfully ran gprMax simulation with input file: {input_file}\nSTDOUT: {sim_result.stdout}"
        logger.info(f"[TOOL RESULT] run_gprmax_simulation_tool - {success_msg}")
        return success_msg
        
    except subprocess.TimeoutExpired:
        error_msg = "Simulation timed out after 1 hour"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error running gprMax simulation: {str(e)}"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg


if __name__ == "__main__":
    input_file = "generated.in"
    result = run_gprmax_simulation_tool(input_file)
    print(result)