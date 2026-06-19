from abc import ABC, abstractmethod

class Board(ABC):

    @abstractmethod
    def get_state(self):
        ...
    
    @abstractmethod
    def get_legal_moves(self):
        ...

    @abstractmethod
    def play(self, action):
        ...
    
    @abstractmethod
    def has_ended(self):
        ...
    
    @abstractmethod
    def get_outcome(self):
        ...
    
    @abstractmethod
    def get_fullmove_count(self):
        ...