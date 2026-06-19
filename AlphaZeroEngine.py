import numpy as np

class AlphaZeroEngine():
    SELF_PLAY_BATCH_COUNT = 0
    SIMULATION_COUNT = 0
    EXPLORATION_CONSTANT = 0
    board = None

    def __init__(self, SELF_PLAY_BATCH_COUNT, SIMULATION_COUNT, EXPLORATION_CONSTANT, board):
        self.SELF_PLAY_BATCH_COUNT = SELF_PLAY_BATCH_COUNT
        self.SIMULATION_COUNT = SIMULATION_COUNT
        self.EXPLORATION_CONSTANT = EXPLORATION_CONSTANT
        self.board = board
    
    def self_play(self):
        games_data = []
        for _ in range(self.SELF_PLAY_BATCH_COUNT):
            game_data = []
            starting_position = self.board.get_starting_position()
            self.board.set_state(starting_position)
            while not self.board.has_ended():
                game_data = []
                PI = self.calculate_MCTS(self.board.get_state())
                actions = list(PI.keys())
                probabilities = list(PI.values())
                selected_action = np.random.choice(actions, p = probabilities)
                game_data.append((self.board.get_state(), PI, None))
                self.board.play(selected_action)
            self.broadcast_z(game_data)
            games_data.append(game_data)
        return games_data
    
    def calculate_MCTS(self, state):
        tree = {}
        path = [(state, None)]
        for _ in self.SIMULATION_COUNT:
            while True:
                next_s, action = self.simulation_step(path, tree)
                path.append((next_s, action))
                if not next_s:
                    path = [(state, None)]
                    break
        pass

    def simulation_step(self, path, tree):
        pass

    def broadcast_z(self, game_data):
        pass