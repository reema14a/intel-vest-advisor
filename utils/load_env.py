from pathlib import Path
from dotenv import load_dotenv
import logging
import os

logger = logging.getLogger(__name__)

def load_agent_env(current_file: str, agent_folder_name: str = None, env: str = "dev"):
    """
    Load environment variables in layers:
    1. Root .env (base configuration)
    2. Environment-specific .env (.env.dev or .env.prd)
    3. Agent-specific .env (if agent_folder_name is provided)
    
    Args:
        current_file (str): The __file__ from the calling module.
        agent_folder_name (str, optional): The name of the agent folder (e.g., 'risk_assessment').
                                          If None, only loads root and env-specific .env files.
        env (str): The environment to load ('dev' or 'prd'). Defaults to 'dev'.
    """
    # Get the root directory
    if "pytest" in current_file:
        # If running in pytest, go up to project root from conftest.py
        root_dir = Path(current_file).resolve().parents[1]
    else:
        # If running normally, go up to project root from the calling file
        root_dir = Path(current_file).resolve().parents[2]
    
    # 1. Load root .env first (base configuration)
    root_env_path = root_dir / ".env"
    if root_env_path.exists():
        load_dotenv(dotenv_path=root_env_path, override=False, verbose=False)
        logger.debug(f"Loaded root environment from: {root_env_path}")
    else:
        logger.warning(f"No root .env file found at: {root_env_path}")
    
    # 2. Load environment-specific .env (overrides root .env)
    env_specific_path = root_dir / f".env.{env}"
    if env_specific_path.exists():
        load_dotenv(dotenv_path=env_specific_path, override=True, verbose=False)
        logger.debug(f"Loaded environment-specific configuration from: {env_specific_path}")
    else:
        logger.warning(f"No environment-specific .env file found at: {env_specific_path}")
    
    # 3. Load agent-specific .env if agent_folder_name is provided
    if agent_folder_name:
        agent_env_path = root_dir / "agents" / agent_folder_name / ".env"
        if agent_env_path.exists():
            load_dotenv(dotenv_path=agent_env_path, override=True, verbose=False)
            logger.debug(f"Loaded agent environment from: {agent_env_path}")
        else:
            logger.debug(f"No agent-specific .env file found at: {agent_env_path}")
    
    return env
