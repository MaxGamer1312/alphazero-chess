import numpy as np
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
    
    def get_DNN_input(self, history_length):
        P1_history_list = []
        P2_history_list = []
        board_history_list = self._get_last_N_board_states(history_length)
        for i in range(history_length):
            P_history_list = self._get_DNN_P_input(board_history_list[i])
            P1_history_list.append(P_history_list["P1"])
            P2_history_list.append(P_history_list["P2"])


    def _get_DNN_P_input(self, board):
        P1 = []
        P2 = []
        all_piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        if board:
            for color in [chess.WHITE, chess.BLACK]:
                if self.board.turn == color:
                    for piece_type in all_piece_types:
                        P1.append(self.bitboard_to_array(board.pieces_mask(piece_type, color)))
                else:
                    for piece_type in all_piece_types:
                        P2.append(self.bitboard_to_array(board.pieces_mask(piece_type, color)))
        else:
            for _ in all_piece_types:
                P1.append(np.zeros((8, 8), dtype=np.float32))
                P2.append(np.zeros((8, 8), dtype=np.float32))

        result = {}
        result["P1"] = P1
        result["P2"] = P2
        return result

    def _get_last_N_board_states(self, n):
        temp_board = self.board.copy()
        history_boards = [self.board.copy()]
        for _ in range(n-1):
            if len(temp_board.move_stack):
                temp_board.pop()
                history_boards.append(temp_board.copy())
            else:
                history_boards.append(None)
        return history_boards

    def bitboard_to_array(self, bitboard):
        grid = []
        row = []
        for i in range(64):
            if i % 8 == 0 and i != 0:
                grid.append(row)
                row = []
            row.append(bitboard%2)
            bitboard = bitboard>>1
        grid.append(row)
        return np.array(grid)
    