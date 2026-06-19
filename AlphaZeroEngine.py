import numpy as np

class AlphaZeroEngine():
    SELF_PLAY_BATCH_COUNT = 0
    SIMULATION_COUNT = 0
    EXPLORATION_CONSTANT = 0
    DNN = None
    board = None
    playing_board = None
    tau_info = None

    def __init__(self, SELF_PLAY_BATCH_COUNT, SIMULATION_COUNT, EXPLORATION_CONSTANT, board, tau_info):
        self.SELF_PLAY_BATCH_COUNT = SELF_PLAY_BATCH_COUNT
        self.SIMULATION_COUNT = SIMULATION_COUNT
        self.EXPLORATION_CONSTANT = EXPLORATION_CONSTANT
        self.board = board
        self.tau_info = tau_info

    
    def self_play(self):
        games_data = []
        for _ in range(self.SELF_PLAY_BATCH_COUNT):
            game_data = []
            self.playing_board = self.board()
            while not self.playing_board.has_ended():
                PI = self.calculate_MCTS(self.playing_board.get_state())
                actions = list(PI.keys())
                probabilities = list(PI.values())
                selected_action = np.random.choice(actions, p = probabilities)
                game_data.append([self.playing_board.get_state(), PI, None])
                self.playing_board.play(selected_action)
            self.broadcast_z(game_data, self.playing_board.get_outcome())
            games_data.append(game_data)
        return games_data

    def broadcast_z(self, game_data, Z):
        for turn_state in game_data[::-1]:
            turn_state[2] = Z
            Z = -Z

    def calculate_MCTS(self, state):
        PI = {}
        tree = {}
        path = [(state, None)]
        for _ in range(self.SIMULATION_COUNT):
            while True:
                next_s, action = self.simulation_step(path, tree)
                path.append((next_s, action))
                if not next_s:
                    path = [(state, None)]
                    break
        board = self.board(state)
        tau = self.tau_info["high_tau"]
        if self.playing_board.get_fullmove_count() > self.tau_info["move_switch_count"]:
            tau = self.tau_info["low_tau"]
        N_tau = 0
        for actions_from_starting_state in board.get_legal_moves():
            N_tau += tree[state]["N_a"][actions_from_starting_state] ** (1/tau)
        for actions_from_starting_state in board.get_legal_moves():
            PI[actions_from_starting_state] = tree[state]["N_a"][actions_from_starting_state] ** (1/tau)/N_tau
        return PI
    
    def simulation_step(self, path, tree):
        if len(path) >= 1:
            curr_s = path[-1][0]
        else:
            return (None, None)
        board = self.board(curr_s)
        legal_moves = board.get_legal_moves()
        if board.has_ended():
            Z = board.get_outcome()
            self.backprop_MCTS(Z, path, tree)
            return (None, None)
        
        if curr_s in tree:
            Q_a = tree[curr_s]["Q_a"]
            N_a = tree[curr_s]["N_a"]
            P_a = tree[curr_s]["P_a"]
            N = tree[curr_s]["N"]
            max_move = None
            max_val = -np.inf
            for legal_move in legal_moves:
                temp = self.calculate_PUCT(Q_a[legal_move], N_a[legal_move], P_a[legal_move], N)
                if temp > max_val:
                    max_move = legal_move
                    max_val = temp
            new_state = board.play(max_move)
            return (new_state, max_move)
        
        Q_a = {}
        N_a = {}
        (P_a, V) = self.DNN.forward(curr_s)
        N = 0
        for legal_move in legal_moves:
            Q_a[legal_move] = 0
            N_a[legal_move] = 0
        self.backprop_MCTS(V, path, tree)
        leaf = {}
        leaf["Q_a"] = Q_a
        leaf["N_a"] = N_a
        leaf["P_a"] = P_a
        leaf["N"] = N
        tree[curr_s] = leaf
        return (None, None)

    def backprop_MCTS(self, V_Z, path, tree):
        for i in range(len(path) - 2, -1, -1):
            V_Z = -V_Z
            state = path[i][0]
            curr_action = path[i + 1][1]
            tree[state]["N_a"][curr_action] += 1
            tree[state]["N"] += 1
            N = tree[state]["N"]
            curr_Q_a = tree[state]["Q_a"][curr_action]
            next_Q_a = curr_Q_a + (V_Z - curr_Q_a) / N
            tree[state]["Q_a"][curr_action] = next_Q_a
            
    def calculate_PUCT(self, Q_sa, N_sa, P_sa, N_s):
        exploitation = Q_sa
        exploration = self.EXPLORATION_CONSTANT * P_sa * np.sqrt(N_s/(1 + N_sa))
        return exploitation + exploration