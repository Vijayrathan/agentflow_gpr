import logging
import dotenv
import os
import openai
from typing import List, Optional
import json
from init import logger, openai_client, openai_model
from sim_setup_agent import get_workspace_directory
import subprocess
import shutil
import sys
import time
import secrets
from pathlib import Path
from langchain.tools import tool

@tool
def run_gprmax_simulation_tool(input_file_content: str) -> str:
    """
    Run gprMax simulation with the given input file content.
    
    Creates a temporary input file in gpr_workspace/generated_files/ and runs the simulation.
    First checks if gprMax is installed. If not, clones the repository,
    installs it using conda, and then runs the simulation.
    
    Args:
        input_file_content: Content of the gprMax input file (.in file content)
    
    Returns:
        str: Success message or error message
    """
    logger.info(f"[TOOL CALL] run_gprmax_simulation_tool - Creating input file and running simulation")
    
    # Validate input content
    if not input_file_content or not input_file_content.strip():
        error_msg = "Input file content is empty"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg
    
    # Find project root (where gpr_workspace directory is located)
    # Start from current file location and go up to find project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up from backend/ to project root
    
    # Create gpr_workspace/generated_files directory if it doesn't exist
    generated_files_dir = project_root / "gpr_workspace" / "generated_files"
    generated_files_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename with timestamp and random suffix
    timestamp = int(time.time() * 1000)
    random_suffix = secrets.token_hex(8)
    input_filename = f"generated_session_{timestamp}_{random_suffix}.in"
    input_file_path = generated_files_dir / input_filename
    
    # Write input file content to the file
    try:
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(input_file_content)
        logger.info(f"[TOOL CALL] run_gprmax_simulation_tool - Created input file: {input_file_path}")
    except Exception as e:
        error_msg = f"Failed to write input file: {str(e)}"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg
    
    # Use the created file path for the rest of the function
    input_file = str(input_file_path)
    
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
        
        # Get workspace directory (outside Flask working directory)
        workspace_dir = get_workspace_directory()
        
        # Clone repository to workspace directory if it doesn't exist
        gprmax_dir = workspace_dir / "gprMax"
        if not gprmax_dir.exists():
            logger.info(f"Cloning gprMax repository to {gprmax_dir}...")
            # Change to workspace directory for cloning
            original_dir = os.getcwd()
            try:
                os.chdir(workspace_dir)
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
            finally:
                # Always restore original directory
                os.chdir(original_dir)
        
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
    
    # Run the simulation with live streaming to terminal
    logger.info(f"Running gprMax simulation with input file: {input_file}")
    try:
        # Use absolute path for input file
        abs_input_file = os.path.abspath(input_file)
        
        # Build the command with unbuffered Python flag
        # Set environment variable for unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        
        if not gprmax_installed:
            # Get conda python executable path to avoid conda run buffering
            # Try to find the python in the conda environment
            conda_python = None
            try:
                # Get conda python path
                conda_info = subprocess.run(
                    ["conda", "info", "--base"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if conda_info.returncode == 0:
                    conda_base = conda_info.stdout.strip()
                    conda_python = os.path.join(conda_base, "envs", "gprMax", "bin", "python")
                    if not os.path.exists(conda_python):
                        # Try Windows path
                        conda_python = os.path.join(conda_base, "envs", "gprMax", "python.exe")
                    if not os.path.exists(conda_python):
                        conda_python = None
            except:
                pass
            
            if conda_python and os.path.exists(conda_python):
                # Use python directly from conda env (avoids conda run buffering)
                cmd = [conda_python, "-u", "-m", "gprMax", abs_input_file]
            else:
                # Fallback to conda run with -u flag
                cmd = ["conda", "run", "-n", "gprMax", "python", "-u", "-m", "gprMax", abs_input_file]
        else:
            cmd = ["python", "-u", "-m", "gprMax", abs_input_file]
        
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env=env
        )
        
        # Capture output and stream to terminal simultaneously
        combined_output = ""
        print("\n" + "="*80, flush=True)
        print("gprMax Simulation Output (Live):", flush=True)
        print("="*80 + "\n", flush=True)
        
        try:
            # Read line by line and print immediately
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    # Process has finished
                    break
                if output:
                    # Print to terminal immediately (no buffering)
                    print(output, end='', flush=True)
                    # Also capture for return value
                    combined_output += output
                    # Log important lines
                    line_lower = output.lower()
                    if any(keyword in line_lower for keyword in ['error', 'warning', 'completed', 'failed']):
                        logger.info(f"[GPRMAX] {output.strip()}")
            
            # Get return code
            return_code = process.poll()
            
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            error_msg = "Simulation timed out after 1 hour"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            print(f"\n{error_msg}\n")
            return error_msg
        
        print("\n" + "="*80)
        print("Simulation Complete")
        print("="*80 + "\n")
        sys.stdout.flush()
        
        if return_code != 0:
            error_msg = f"gprMax simulation failed (exit code: {process.returncode})\n\n{combined_output}"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            return error_msg
        
        # Return the full simulation logs
        if combined_output:
            return combined_output
        else:
            # Fallback if no output captured
            return "Simulation completed successfully, but no output was captured."
        
    except subprocess.TimeoutExpired:
        error_msg = "Simulation timed out after 1 hour"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error running gprMax simulation: {str(e)}"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg
