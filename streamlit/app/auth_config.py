"""
Authentication configuration for the Streamlit application.
"""

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from pathlib import Path

# Path to the configuration file
auth_file = Path(__file__).parent / 'auth_config.yaml'

# Default configuration
DEFAULT_CONFIG = {
    'credentials': {
        'usernames': {
            'admin': {
                'email': 'admin@example.com',
                'name': 'Administrateur',
                'password': '',  # Will be filled during initialization
                'role': 'admin'
            },
            'user': {
                'email': 'user@example.com',
                'name': 'Utilisateur Standard',
                'password': '',  # Will be filled during initialization
                'role': 'user'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'streamlit_auth_key',
        'name': 'streamlit_auth_cookie'
    },
    'preauthorized': {
        'emails': []
    }
}

def init_auth_config():
    """Initializes the authentication configuration file if it doesn't exist."""
    if not auth_file.exists():
        # Create a copy of the default configuration
        config = DEFAULT_CONFIG.copy()
        
        # Set plain text passwords
        config['credentials']['usernames']['admin']['password'] = 'admin123'
        config['credentials']['usernames']['user']['password'] = 'user123'
        
        # Hash passwords with the new method
        config['credentials'] = stauth.Hasher.hash_passwords(config['credentials'])
        
        # Create parent directory if needed
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the configuration
        with open(auth_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

def load_auth_config():
    """Loads the authentication configuration."""
    # Ensure the file exists
    if not auth_file.exists():
        init_auth_config()
    
    # Load the configuration
    with open(auth_file) as f:
        return yaml.load(f, Loader=SafeLoader)

def get_authenticator():
    """Returns a configured Streamlit authentication object."""
    config = load_auth_config()
    return stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

def get_user_role(username):
    """Retrieves a user's role."""
    config = load_auth_config()
    return config['credentials']['usernames'].get(username, {}).get('role', 'user')
