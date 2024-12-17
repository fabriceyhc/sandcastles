from attack.oracles.base import Oracle
import socketio
import time

class HumanOracle():
    def __init__(self, server_url="http://131.179.88.53:5000"):
        self.sio = socketio.Client()
        self.response_data = None
        self.roomid = None
        self.task = None
        self.taskid = 0

        # Attach event handlers
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('set_room_id', self.on_set_room_id)
        self.sio.on('receive_answer', self.on_receive_answer)
        self.sio.on('new_user', self.on_new_user)

        # Connect to the server
        self.sio.connect(server_url)

    def on_connect(self):
        if self.roomid is None:
            self.sio.emit('create', {})
        else:
            self.sio.emit('create', {'roomid': self.roomid})
    
    def on_disconnect(self):
        print("Disconnected from the server.")

    def on_set_room_id(self, data):
        self.roomid = data
        print(f"Connected to room ID: {data}")
    
    def on_new_user(self, sid):
        self.sio.emit('resend_task', {'sid': sid, 'data': self.task})
    
    def on_receive_answer(self, data):
        if data['taskid'] == self.taskid:
            self.response_data = data

    def create_task(self, instruction, original_text, mutated_text):
        self.taskid += 1
        self.task = {
            'roomid': self.roomid,
            'taskid': self.taskid,
            'instruction': instruction,
            'original_text': original_text,
            'mutated_text': mutated_text
        }

        self.sio.emit('create_task', self.task)

    def is_quality_preserved(self, instruction, original_text, mutated_text):
        self.response_data = None
        self.create_task(instruction, original_text, mutated_text)
        print("Waiting for a response...")
        while self.response_data is None:
            time.sleep(0.1)
        original = self.response_data
        self.task = None
        original.update({'quality_preserved': original['choice'] in ['preserved']})
        return original