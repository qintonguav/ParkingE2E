import threading
import signal
import sys

class CommandThread(threading.Thread):
    def __init__(self, keyboard_signal):
        super(CommandThread, self).__init__()
        self.keyboard_signal = keyboard_signal
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)

    def run(self):
        while self.running:
            if sys.stdin.isatty():
                keyboard_info = sys.stdin.readline().strip()
                if keyboard_info:
                    self.keyboard_signal.append(keyboard_info)

                if len(self.keyboard_signal) > 1:
                    self.keyboard_signal.pop(0)

    def signal_handler(self, sig, frame):
        self.running = False
        print('Ctrl+C detected, exiting...')
        sys.exit(0)