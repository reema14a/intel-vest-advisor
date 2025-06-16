import gradio as gr
import os
import logging
import argparse
from utils.logging_config import setup_logging, get_logger
from utils.env_manager import env_manager
from utils.session_manager import SessionManager
from ui.investment_advisor_ui import InvestmentAdvisorUI

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Intel Vest Advisor Application')
    parser.add_argument(
        '--env', 
        choices=['dev', 'prd'], 
        default='dev',
        help='Environment to run the application in'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=7860,
        help='Port to run the Gradio interface on'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse arguments
    args = parse_arguments()
    
    # Load environment first
    env_manager.load_environment(args.env)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level)
    logger.info(f"Starting Investment Advisor in {args.env} environment")
    
    # Validate environment
    required_vars = ['GCP_PROJECT_ID', 'MODEL_NAME']
    env_manager.validate_required_vars(required_vars)
    
    try:
        # Initialize session manager
        session_manager = SessionManager()
        
        # Create UI
        ui = InvestmentAdvisorUI(session_manager)
        demo = ui.create_ui()
        
        # Launch Gradio interface
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 