import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from model_manager import ModelManager

class TestModelManager(unittest.TestCase):
    def setUp(self):
        # Mock home directory to avoid messing with real files
        self.patcher = patch('pathlib.Path.home')
        self.mock_home = self.patcher.start()
        self.mock_home.return_value = Path('/tmp/mock_home')
        
    def tearDown(self):
        self.patcher.stop()

    def test_init_creates_dir(self):
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('json.load', return_value={'installed_models': {}}) as mock_json_load, \
             patch('builtins.open', mock_open(read_data='{}')):
            
            manager = ModelManager()
            # Check if directory creation was attempted
            self.assertTrue(mock_mkdir.called)

    def test_check_installation_whisper(self):
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data='{}')):
            
            # Mock exists sequence: 
            # 1. config_file.exists() -> False (to skip loading) OR True (if we mock open)
            # 2. models_dir.mkdir -> (mocked automatically if we rely on it, but here we just need init to pass)
            # 3. model existence check -> True
            
            # Let's say config exists is False so we start fresh
            # Then specific model checks return True
            
            # side_effect allows different returns for sequential calls
            # But the path info is lost if we just use a list. 
            # Better to use a function or verify call args.
            # Simple approach: Mock ModelManager._load_config to avoid file issues during init
            
            with patch.object(ModelManager, '_load_config', return_value={'installed_models': {}}):
                manager = ModelManager()
                # Now mock the specific check
                # The loop checks all models. Whisper uses _check_whisper_model which uses Path.exists
                # We need to ensure that when checking for whisper model it returns True
                
                # Mock _check_whisper_model directly to isolate logic?
                # Or just mock Path.exists for the cache path.
                
                mock_exists.return_value = True
                
                status = manager.check_installation()
                self.assertIn('whisper_medium', status)
                self.assertTrue(status['whisper_medium'])
            
    def test_get_recommended_model(self):
        manager = ModelManager()
        self.assertEqual(manager.get_recommended_model('zh'), 'funasr')
        self.assertEqual(manager.get_recommended_model('en'), 'whisper_medium')
        # Default fallback
        self.assertEqual(manager.get_recommended_model(None), 'whisper_medium')

if __name__ == '__main__':
    unittest.main()
