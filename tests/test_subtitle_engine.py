import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Mock whisper and funasr modules BEFORE importing subtitle_engine
mock_whisper = MagicMock()
sys.modules['whisper'] = mock_whisper

mock_funasr = MagicMock()
sys.modules['funasr'] = mock_funasr

mock_tqdm = MagicMock()
sys.modules['tqdm'] = mock_tqdm

from subtitle_engine import SubtitleEngine

class TestSubtitleEngine(unittest.TestCase):
    def setUp(self):
        self.patcher_home = patch('pathlib.Path.home')
        self.mock_home = self.patcher_home.start()
        self.mock_home.return_value = Path('/tmp/mock_home')
        
        # Mock ModelManager initialization to avoid disk checks
        self.patcher_mm = patch('subtitle_engine.ModelManager')
        self.mock_mm_class = self.patcher_mm.start()
        self.mock_mm = self.mock_mm_class.return_value
        self.mock_mm.check_installation.return_value = {} # Assume nothing installed or checks mocked
        self.mock_mm.MODELS = {
            'whisper_test': {'name': 'Test Whisper', 'provider': 'openai', 'model_id': 'base', 'required': False},
            'funasr_test': {'name': 'Test FunASR', 'provider': 'modelscope', 'model_id': 'paraformer', 'required': False}
        }
        
    def tearDown(self):
        self.patcher_home.stop()
        self.patcher_mm.stop()

    def test_srt_formatting(self):
        # We can test the helper method directly if we access it, or via save_srt
        # Accessing protected method for unit test
        engine = SubtitleEngine()
        timestamp = engine._format_timestamp(65.5)
        self.assertEqual(timestamp, "00:01:05,500")
        
        timestamp = engine._format_timestamp(0)
        self.assertEqual(timestamp, "00:00:00,000")

    def test_save_srt(self):
        engine = SubtitleEngine()
        result = {
            'segments': [
                {'start': 0, 'end': 2.5, 'text': 'Hello world'}
            ]
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            engine.save_srt(result, 'output.srt')
            mock_file.assert_called_with('output.srt', 'w', encoding='utf-8')
            handle = mock_file()
            # Check writes
            # 1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n
            handle.write.assert_any_call("1\n")
            handle.write.assert_any_call("00:00:00,000 --> 00:00:02,500\n")
            handle.write.assert_any_call("Hello world\n\n")

    def test_user_dict_application(self):
        engine = SubtitleEngine()
        engine.user_dict = {'foo': 'bar'}
        
        raw_result = {
            'segments': [{'text': 'This is foo text', 'start': 0, 'end': 1}]
        }
        
        processed = engine._apply_user_dict(raw_result)
        self.assertEqual(processed['segments'][0]['text'], 'This is bar text')

if __name__ == '__main__':
    unittest.main()
