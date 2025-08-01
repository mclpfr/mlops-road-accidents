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
        
        # Set plain text passwords (will be hashed below)
        admin_pwd = 'admin123'
        user_pwd = 'user123'

        # Hash passwords with the correct Hasher API
        hashed_pwds = stauth.Hasher([admin_pwd, user_pwd]).generate()
        config['credentials']['usernames']['admin']['password'] = hashed_pwds[0]
        config['credentials']['usernames']['user']['password'] = hashed_pwds[1]
        
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
        config = yaml.load(f, Loader=SafeLoader)

    # Auto-migrate: ensure stored passwords look hashed (start with '$')
    migrated = False
    plain_pwds = []
    for user, data in config['credentials']['usernames'].items():
        pwd = data.get('password','')
        if not pwd.startswith('$2'):  # not bcrypt-like hash
            plain_pwds.append((user, pwd))
    if plain_pwds:
        # Re-hash and update
        hashes = stauth.Hasher([pwd for _, pwd in plain_pwds]).generate()
        for (user,_), hashed in zip(plain_pwds, hashes):
            config['credentials']['usernames'][user]['password'] = hashed
        migrated = True
    if migrated:
        # Persist migrated file
        with open(auth_file,'w') as f_out:
            yaml.dump(config, f_out, default_flow_style=False)
    return config

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
