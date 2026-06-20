import chess
from AlphaZeroEngine import AlphaZeroEngine
TAU_INFO = {}
TAU_INFO["low_tau"] = 0.1
TAU_INFO["high_tau"] = 0.8
TAU_INFO["move_switch_count"] = 70

aze = AlphaZeroEngine(SELF_PLAY_BATCH_COUNT=10,
                 SIMULATION_COUNT=100,
                   EXPLORATION_CONSTANT=0.7,
                     board=chess.Board, tau_info=TAU_INFO)