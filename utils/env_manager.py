import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EnvironmentManager:
    """Manages environment-specific configurations"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.environment = os.getenv("ENV", "dev").lower()
        self.loaded_files = []
        
    def load_environment(self, env_name: str = None):
        """Load environment-specific configuration"""
        env_name = env_name or self.environment
        
        # Load base environment file first
        base_env_file = self.project_root / ".env"
        if base_env_file.exists():
            load_dotenv(base_env_file, override=False)
            self.loaded_files.append(str(base_env_file))
            logger.info(f"Loaded base environment file: {base_env_file}")
        
        # Load environment-specific file
        env_file = self.project_root / f".env.{env_name}"
        if env_file.exists():
            load_dotenv(env_file, override=True)  # Override base settings
            self.loaded_files.append(str(env_file))
            logger.info(f"Loaded {env_name} environment file: {env_file}")
        else:
            logger.warning(f"Environment file not found: {env_file}")
        
        # Set final environment variable
        os.environ["ENVIRONMENT"] = env_name
        logger.info(f"Environment set to: {env_name}")
        
    def load_agent_environment(self, agent_path: str, agent_name: str):
        """Load agent-specific environment variables"""
        agent_dir = Path(agent_path).parent
        agent_env_file = agent_dir / ".env"
        
        if agent_env_file.exists():
            load_dotenv(agent_env_file, override=True)
            self.loaded_files.append(str(agent_env_file))
            logger.debug(f"Loaded agent environment for {agent_name}: {agent_env_file}")
        else:
            logger.debug(f"No agent-specific .env file found for {agent_name}")
    
    def get_loaded_files(self):
        """Return list of loaded environment files"""
        return self.loaded_files
    
    def validate_required_vars(self, required_vars: list):
        """Validate that required environment variables are set"""
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        logger.info("All required environment variables are set")
        return True

# Global instance
env_manager = EnvironmentManager()

# Compatibility function for existing load_agent_env calls
def load_agent_env(current_file: str, agent_folder_name: str):
    """
    Load `.env` file from the given agent's subdirectory.
    This function maintains compatibility with existing code.
    """
    env_manager.load_agent_environment(current_file, agent_folder_name) 