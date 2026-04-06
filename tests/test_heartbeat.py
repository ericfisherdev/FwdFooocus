import time
import unittest


class TestHeartbeat(unittest.TestCase):
    def test_initial_state_is_connected(self):
        from modules.heartbeat import is_browser_connected
        self.assertTrue(is_browser_connected(timeout_seconds=15))

    def test_update_heartbeat_keeps_connected(self):
        from modules.heartbeat import is_browser_connected, update_heartbeat
        update_heartbeat()
        self.assertTrue(is_browser_connected(timeout_seconds=15))

    def test_stale_heartbeat_is_disconnected(self):
        from modules import heartbeat
        heartbeat._last_heartbeat_time = time.time() - 20.0
        self.assertFalse(heartbeat.is_browser_connected(timeout_seconds=15))

    def test_update_after_stale_reconnects(self):
        from modules import heartbeat
        heartbeat._last_heartbeat_time = time.time() - 20.0
        self.assertFalse(heartbeat.is_browser_connected(timeout_seconds=15))
        heartbeat.update_heartbeat()
        self.assertTrue(heartbeat.is_browser_connected(timeout_seconds=15))

    def test_custom_timeout(self):
        from modules import heartbeat
        heartbeat._last_heartbeat_time = time.time() - 5.0
        self.assertFalse(heartbeat.is_browser_connected(timeout_seconds=3))
        self.assertTrue(heartbeat.is_browser_connected(timeout_seconds=10))


if __name__ == '__main__':
    unittest.main()
