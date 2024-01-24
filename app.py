from flask import Flask, render_template, request
import random
import time
from threading import Thread
from models import GameOfLifeModel
import torch


app = Flask(__name__)

default_box_count = 32
game = GameOfLifeModel(default_box_count)
initial_state = [[0] * default_box_count for _ in range(default_box_count)]
current_state = initial_state
running = False


@app.route('/update', methods=['POST'])
def update():
    global current_state
    data = request.get_json()

    # Assuming the JSON contains a 'state' key
    flat_list = data.get('state', [])
    new_state = [flat_list[i:i+32] for i in range(0, len(flat_list), 32)]

    current_state = new_state

    return 'Update received successfully'

def generate_random_state():
    global current_state
    env = torch.tensor([[current_state]]).float()
    while running:
        frame = game(env)
        env = torch.where(
            ((env > 0) & torch.isin(frame, torch.tensor([2, 3]))) | ((env == 0) & (frame == 3)),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32)
        )
        current_state = env.squeeze().squeeze().int().tolist()
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html', state=current_state, box_count=default_box_count)

@app.route('/start')
def start():
    global running
    running = True
    Thread(target=generate_random_state).start()
    return 'Started'

@app.route('/stop')
def stop():
    global running
    running = False
    return 'Stopped'

if __name__ == '__main__':
    app.run(debug=True)
