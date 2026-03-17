"""
Backtracking as an example to solve the "Three Connect" game problem.
"Three Connect" is a faster variant of the classical "Four Connect" game 
(i.e. same rules, but only three connected pieces are needed to win).

The game is played on a NxM grid where two players take turns placing their markers (1 and 2).
The objective is to connect three or four of one's own markers in a row, either horizontally, vertically, or diagonally.
The original "Four Connect" game setup (4 connected pieces win on a 7x6 grid) is timeconsuming without further optimization.
To study backtracking, a faster "Three Connect" variant can be used (e.g. 3 connected pieces win on a 3x4 grid).
The program uses backtracking to explore all possible game states and determine if there is a winning strategy for the first player.

This is an optimized version of four_connect.py with improved backtracking performance.
"""
import numpy as np

class Board:
    """
    Class to represent the NxM board for the Two/Three/Four Connect game
    """
    def __init__(self, cols=7, rows=6, num_pieces_to_win=4, board = None, winner = -1):
        # Initializing board constructor
        self.board = board if board is not None else np.zeros((rows, cols), dtype=np.int8) # playing field, NxM grid, each cell has content 0 = empty, 1 = player 1 or 2 = player 2
        self.num_pieces_to_win = num_pieces_to_win # number of pieces needed to win (4 to play the "Four Connect" game, or 3 to play the faster "Three Connect")
        self.winner = winner                       # If game is completed: 0 = remis, 1 = player 1 wins, 2 = player 2 wins, otherwise -1

    def shallow_copy(self):
        # Create and return a shallow copy of the board
        return Board(cols=self.cols(), rows=self.rows(), num_pieces_to_win=self.num_pieces_to_win, board=self.board, winner = self.winner)

    def cols(self):
        # Return the number of columns in the board (default is 7)
        return self.board.shape[1]

    def rows(self):
        # Return the number of rows in the board (default is 6)
        return self.board.shape[0]

    def is_col_full(self, col):
        # Return True if the specified column is full (no empty cells), otherwise False
        return self.board[0][col] != 0

    def drop_piece(self, col, piece, verbosity = 1):
        # Drop a piece in the specified column (if it's not full yet) and check if there's a winner. Returns the row of the dropped piece (or -1 if the column is full).
        if self.winner < 0: # game is still ongoing
            for row in range(self.rows()-1, -1, -1):
                if self.board[row][col] == 0:
                    self.board[row][col] = piece
                    if self.check_winner_at_drop(piece, col, row):
                        self.winner = piece
                        if verbosity >= 1:
                            self.print()
                    if self.winner < 0 and row == 0 and all(self.board[0][i] != 0 for i in range(self.cols())):
                        self.winner = 0 # game completed with remis, if all columns are full at row = (self.rows()-1)
                        if verbosity >= 1:
                            self.print()
                    return row
        return -1
    
    def check_winner_at_drop(self, piece, col, row):
        # Check from (col,row) vertically downward for connected pieces
        if (1 + self.count_connected_pieces(piece, col, row + 1, 0, 1)) >= self.num_pieces_to_win:
            return True
        # Check from (col,row) horizontally to the left and right for connected pieces
        if (1 + self.count_connected_pieces(piece, col - 1, row, -1, 0) + self.count_connected_pieces(piece, col + 1, row, 1, 0)) >= self.num_pieces_to_win:
            return True
        # Check from (col,row) diagonally (top left to bottom right) for connected pieces
        if (1 + self.count_connected_pieces(piece, col - 1, row - 1, -1, -1) + self.count_connected_pieces(piece, col + 1, row + 1, 1, 1)) >= self.num_pieces_to_win:
            return True
        # Check from (col,row) diagonally (top right to bottom left) for connected pieces
        if (1 + self.count_connected_pieces(piece, col - 1, row + 1, -1, 1) + self.count_connected_pieces(piece, col + 1, row - 1, 1, -1)) >= self.num_pieces_to_win:
            return True
        return False

    def count_connected_pieces(self, piece, col, row, delta_col, delta_row):
        # Count the number of connected pieces of the same type in a given direction
        num_connected = 0
        curr_col = col
        curr_row = row
        while 0 <= curr_col < self.cols() and 0 <= curr_row < self.rows() and self.board[curr_row][curr_col] == piece:
            num_connected += 1
            curr_col += delta_col
            curr_row += delta_row
        return num_connected

    def print(self):
        # Print the current state of the board
        print(f"board winner: {self.winner}")
        print(f"{self.board}")
        print()

class PlayConnect:
    """
    Class to play the Two/Three/Four Connect game on a given board
    """
    def __init__(self, board, player = 1, verbosity = 1, win_stats = [0,0,0]):
        self.board = board             # current state of the board
        self.player = player           # current player (1 or 2), player 1 starts by default
        self.verbosity = verbosity     # verbosity level for printing
        self.win_stats = win_stats     # statistics of wins: [number of remis, number of player 1 wins, number of player 2 wins]

    def generate_children(self):
        # Drop a piece in each column
        child_nodes = []
        for col in range(self.board.cols()):
            if not self.board.is_col_full(col):
                child_node = PlayConnect(board=self.board.shallow_copy(), player = 2 if self.player == 1 else 1, verbosity = self.verbosity, win_stats = self.win_stats)
                row = child_node.board.drop_piece(col, self.player, verbosity = self.verbosity) # Drop a piece and check for winners
                child_nodes.append(child_node)
                if child_node.board.winner >= 0: # game completed, update statistics
                    self.win_stats[child_node.board.winner] +=1 # update win statistics
                else: # game not completed yet, continue generating children
                    child_node.generate_children()
                child_node.board.board[row][col] = 0 # undo drop_piece for backtracking
        # backtrack: assign all generated child nodes at once for better performance
        best_child_winner = 2 if self.player == 1 else 1 # worst case assumption: opponent wins, player loses
        for child_node in child_nodes:
            if child_node.board.winner == self.player: # winning is always the very best choice
                best_child_winner = child_node.board.winner
                break
            if child_node.board.winner == 0: # remis is always the second best choice
                best_child_winner = 0
        self.board.winner = best_child_winner

if __name__ == "__main__":
    """
    Create the initial board and root node, then generate the game tree and backtrack to find a winning strategy for player 1.
    """
    games = [
        PlayConnect(board=Board(cols=4, rows=4, num_pieces_to_win=2), verbosity=1, win_stats=[0,0,0]),   # cross-check: player 1 can always win on any board in the "two connect" variant
        PlayConnect(board=Board(cols=3, rows=4, num_pieces_to_win=3), verbosity=1, win_stats=[0,0,0]),   # a fast "three connect" variant on a 3x4 board, player 1 starts
        # PlayConnect(board=Board(cols=7, rows=6, num_pieces_to_win=4), verbosity=0, win_stats=[0,0,0]), # "four connect" on the commonly used 7x6 board, player 1 starts
    ]
    for game in games:
        game.generate_children() # generate the game tree recursively
        print(f"Number of winning games for player 1: {game.win_stats[1]}")
        print(f"Number of winning games for player 2: {game.win_stats[2]}")
        print(f"Number of remis games:                {game.win_stats[0]}")
        print(f"Backtracking winner using best answers: {game.board.winner}")
        print()
