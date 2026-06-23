import numpy as np
from Board import Board
import chess
import torch

class ChessBoard(Board):
    board = None
    MAX_EXPECTED_MOVES = 0
    def __init__(self, starting_position = None, MAX_EXPECTED_MOVES = 100):
        self.MAX_EXPECTED_MOVES = MAX_EXPECTED_MOVES
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
        winner_outcome = self.board.outcome().winner
        if winner_outcome == None:
            return 0
        if winner_outcome != self.board.turn:
            return 1
        return -1
    
    def get_fullmove_count(self):
        return self.board.fullmove_number
    
    def get_DNN_input(self, history_length):
        P1_history_list = np.zeros((history_length, 6, 8, 8), dtype=np.float32)
        P2_history_list = np.zeros((history_length, 6, 8, 8), dtype=np.float32)
        repetitions_history_list = np.zeros((history_length, 2, 8, 8), dtype=np.float32)
        board_history_list = self._get_last_N_board_states(history_length)
        for i in range(history_length):
            P_history_item = self._get_DNN_P_input(board_history_list[i])
            P1_history_list[i] = P_history_item["P1"]
            P2_history_list[i] = P_history_item["P2"]
            repetitions_history_item = self._get_DNN_repetition_input(board_history_list[i])
            repetitions_history_list[i] = repetitions_history_item
        
        color_input = np.zeros((1, 8, 8), dtype=np.float32)
        color_input[0] = self._get_DNN_color_input(self.board)
        total_move_count_input = np.zeros((1, 8, 8), dtype=np.float32)
        total_move_count_input[0] = self._get_DNN_movecount_input(self.board)
        P_castling_input = self._get_DNN_P_castling_input(self.board)
        P1_castling_input = P_castling_input["P1"]
        P2_castling_input = P_castling_input["P2"]
        no_progress_count_input = np.zeros((1, 8, 8), dtype=np.float32)
        no_progress_count_input[0] = self._get_DNN_no_progress_count(self.board)

        result = np.concatenate((P1_history_list, P2_history_list, repetitions_history_list), axis=1)
        result = np.reshape(result, (-1, 8, 8))
        result = np.concatenate((result, 
                                 color_input, 
                                 total_move_count_input, 
                                 P1_castling_input, 
                                 P2_castling_input, 
                                 no_progress_count_input), axis=0)

        return torch.from_numpy(result)


    def _get_DNN_P_input(self, board):
        P1 = np.zeros((6, 8, 8), dtype=np.float32)
        P2 = np.zeros((6, 8, 8), dtype=np.float32)
        all_piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        if board:
            for color in [chess.WHITE, chess.BLACK]:
                if self.board.turn == color:
                    for i, piece_type in enumerate(all_piece_types):
                        P1[i] = self.bitboard_to_array(board.pieces_mask(piece_type, color))
                else:
                    for i, piece_type in enumerate(all_piece_types):
                        P2[i] = self.bitboard_to_array(board.pieces_mask(piece_type, color))

        result = {}
        result["P1"] = P1
        result["P2"] = P2
        return result
    
    def _get_DNN_repetition_input(self, board):
        repetitions = np.zeros((2, 8, 8), dtype=np.float32)
        if board:
            if board.is_repetition(3):
                repetitions[1] = np.ones((8, 8), dtype=np.float32)
            elif board.is_repetition(2):
                repetitions[0] = np.ones((8, 8), dtype=np.float32)

        return repetitions
    
    def _get_DNN_color_input(self, board):
        if board.turn == chess.WHITE:
            return np.ones((8, 8), dtype=np.float32)
        return np.zeros((8, 8), dtype=np.float32)
    
    def _get_DNN_movecount_input(self, board):
        normalized = min(board.fullmove_number / self.MAX_EXPECTED_MOVES, 1.0)
        return np.full((8, 8), normalized, dtype=np.float32)
    
    def _get_DNN_P_castling_input(self, board):
        P1 = np.zeros((2, 8, 8), dtype=np.float32)
        P2 = np.zeros((2, 8, 8), dtype=np.float32)

        white_side = np.zeros((2, 8, 8), dtype=np.float32)
        black_side = np.zeros((2, 8, 8), dtype=np.float32)

        if board.has_kingside_castling_rights(chess.WHITE):
            white_side[0] = np.ones((8, 8), dtype=np.float32)
        if board.has_queenside_castling_rights(chess.WHITE):
            white_side[1] = np.ones((8, 8), dtype=np.float32)

        if board.has_kingside_castling_rights(chess.BLACK):
            black_side[0] = np.ones((8, 8), dtype=np.float32)
        if board.has_queenside_castling_rights(chess.BLACK):
            black_side[1] = np.ones((8, 8), dtype=np.float32)
        if board.turn == chess.WHITE:
            P1 = white_side
            P2 = black_side
        else:
            P1 = black_side
            P2 = white_side
        
        result = {}
        result["P1"] = P1
        result["P2"] = P2

        return result
    
    def _get_DNN_no_progress_count(self, board):
        normalized = min(board.halfmove_clock / 100, 1.0)
        return np.full((8, 8), normalized, dtype=np.float32)
    
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
    