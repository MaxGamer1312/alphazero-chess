from Board import Board
import chess

class ChessBoard(Board):
    board = None

    def __init__(self, starting_position = None):
        if starting_position:
            self.board = chess.Board(starting_position)
            return
        self.board = chess.Board()
    
    def get_state(self):
        return self.board.fen()
    
    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]
    
    def play(self, action):
        self.board.push_uci(action)
        return self.get_state()
    
    def has_ended(self):
        return self.board.is_game_over()
    
    def get_outcome(self):
        if self.board.outcome().winner == None:
            return 0
        if self.board.outcome().winner != self.board.turn:
            return 1
        return -1
    
    def get_fullmove_count(self):
        return self.board.fullmove_number
    