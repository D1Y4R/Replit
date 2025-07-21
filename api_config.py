import os
import json
import logging

logger = logging.getLogger(__name__)

class APIConfig:
    def __init__(self):
        self.config_file = 'api_config.json'
        self.default_api_key = '39bc8dd65153ff5c7c0f37b4939009de04f4a70c593ee1aea0b8f70dd33268c0'
        self.current_api_key = None
        self.load_config()
    
    def load_config(self):
        """Load API configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.current_api_key = config.get('api_key', self.default_api_key)
                    logger.info("API configuration loaded from file")
            else:
                self.current_api_key = self.default_api_key
                logger.info("Using default API key")
        except Exception as e:
            logger.error(f"Error loading API config: {e}")
            self.current_api_key = self.default_api_key
    
    def save_config(self, api_key):
        """Save API configuration to file and update all files using the API key"""
        try:
            from datetime import datetime
            config = {
                'api_key': api_key,
                'updated_at': str(datetime.now())
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.current_api_key = api_key
            
            # Update all files that use the API key
            self._update_all_api_key_references(api_key)
            
            logger.info("API configuration saved successfully and all files updated")
            return True
        except Exception as e:
            logger.error(f"Error saving API config: {e}")
            return False
    
    def _update_all_api_key_references(self, new_api_key):
        """Update API key in all relevant files"""
        files_to_update = [
            ('api_routes.py', self._update_api_routes),
            ('match_prediction.py', self._update_match_prediction),
        ]
        
        for filename, update_func in files_to_update:
            try:
                if os.path.exists(filename):
                    update_func(filename, new_api_key)
                    logger.info(f"Updated API key in {filename}")
            except Exception as e:
                logger.error(f"Error updating {filename}: {e}")
        
        # Update the default API key for future instances
        self.default_api_key = new_api_key
    
    def _update_api_routes(self, filename, new_api_key):
        """Update API key references in api_routes.py"""
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace hardcoded API key occurrences
        old_key = self.default_api_key
        content = content.replace(f"'{old_key}'", f"'{new_api_key}'")
        content = content.replace(f'"{old_key}"', f'"{new_api_key}"')
        
        # Update environment variable fallbacks
        content = content.replace(
            f"os.environ.get('API_FOOTBALL_KEY', '{old_key}')",
            f"os.environ.get('API_FOOTBALL_KEY', '{new_api_key}')"
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _update_match_prediction(self, filename, new_api_key):
        """Update API key references in match_prediction.py"""
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace hardcoded API key in MatchPredictor class
        old_key = self.default_api_key
        content = content.replace(f"'{old_key}'", f"'{new_api_key}'")
        content = content.replace(f'"{old_key}"', f'"{new_api_key}"')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def get_api_key(self):
        """Get current API key"""
        return self.current_api_key or self.default_api_key
    
    def test_api_key(self, api_key):
        """Test API key validity"""
        import requests
        from datetime import datetime
        
        try:
            url = "https://apiv3.apifootball.com/"
            params = {
                'action': 'get_events',
                'APIkey': api_key,
                'from': datetime.now().strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'timezone': 'Europe/Istanbul'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if API returned error
                if isinstance(data, dict) and 'message' in data:
                    error_msg = data.get('message', '')
                    if 'invalid' in error_msg.lower() or 'unauthorized' in error_msg.lower():
                        return False, error_msg
                    elif 'no event found' in error_msg.lower():
                        # This is OK - just means no matches today, but API key is valid
                        return True, "API key is valid"
                
                # If we get a list (even empty), API key is valid
                if isinstance(data, list):
                    return True, "API key is valid"
                
                return True, "API key appears to be valid"
            else:
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error testing API key: {e}")
            return False, f"Connection error: {str(e)}"

# Global API config instance
api_config = APIConfig()