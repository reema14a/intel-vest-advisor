from pathlib import Path
from dotenv import load_dotenv
# import os


def load_agent_env(current_file: str, agent_folder_name: str):
    """
    Load `.env` file from the given agent's subdirectory.

    Args:
        current_file (str): The __file__ from the calling module.
        agent_folder_name (str): The name of the agent folder (e.g., 'model_management').
    """
    root_dir = Path(current_file).resolve().parents[2]
    env_path = root_dir / "agents" / agent_folder_name / ".env"

    # print(f"[DEBUG] current_file resolved to: {Path(current_file).resolve()}")
    # print(f"[DEBUG] Looking for .env at: {env_path}")
    # print(f"[DEBUG] Does .env file exist? {env_path.exists()}")

    load_dotenv(dotenv_path=env_path, override=True, verbose=True)

    # print(f"[DEBUG] PROJECT_ID loaded as: {os.getenv('PROJECT_ID')}")
