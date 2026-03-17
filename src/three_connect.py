"""
Backtracking as an example to solve the "Three Connect" game problem.
"Three Connect" is a faster variant of the classical "Four Connect" game 
(i.e. same rules, but only three connected pieces are needed to win).

The game is played on a NxM grid where two players take turns placing their markers (1 and 2).
The objective is to connect three or four of one's own markers in a row, either horizontally, vertically, or diagonally.
The original "Four Connect" game setup (4 connected pieces win on a 7x6 grid) is timeconsuming without further optimization.
To study backtracking, a faster "Three Connect" variant can be used (e.g. 3 connected pieces win on a 3x4 grid).
The program uses backtracking to explore all possible game states and determine if there is a winning strategy for the first player.
"""
import numpy as np

class Board:
    """
    Class to represent the NxM board for the Two/Three/Four Connect game
    """
    def __init__(self, cols=7, rows=6, num_pieces_to_win=4, board = None, winner = 0, game_completed = False):
        # Initializing board constructor
        self.board = board if board is not None else np.zeros((rows, cols), dtype=np.int8) # playing field, NxM grid, each cell has content 0 = empty, 1 = player 1 or 2 = player 2
        self.num_pieces_to_win = num_pieces_to_win # number of pieces needed to win (4 to play the "Four Connect" game, or 3 to play the faster "Three Connect")
        self.winner = winner                       # If game_completed: 0 = remis, 1 = player 1 wins, 2 = player 2 win
        self.game_completed = game_completed       # True if the game is completed (i.e. a player has won or it's remis), otherwise False

    def deep_copy(self):
        # Create and return a deep copy of the board
        return Board(cols=self.cols(), rows=self.rows(), num_pieces_to_win=self.num_pieces_to_win, board=np.copy(self.board), winner = self.winner, game_completed = self.game_completed)

    def shallow_copy(self):
        # Create and return a shallow copy of the board
        return Board(cols=self.cols(), rows=self.rows(), num_pieces_to_win=self.num_pieces_to_win, board=self.board, winner = self.winner, game_completed = self.game_completed)

    def cols(self):
        # Return the number of columns in the board (default is 7)
        return self.board.shape[1]

    def rows(self):
        # Return the number of rows in the board (default is 6)
        return self.board.shape[0]

    def is_col_full(self, col):
        # Return True if the specified column is full (no empty cells), otherwise False
        return self.board[0][col] != 0

    def drop_piece(self, col, piece):
        # Drop a piece in the specified column (if it's not full yet) and check if there's a winner. Returns the row of the dropped piece (or -1 if the column is full).
        if not self.game_completed: # game is still ongoing
            for row in range(self.rows()-1, -1, -1):
                if self.board[row][col] == 0:
                    self.board[row][col] = piece
                    if self.check_winner_at_drop(piece, col, row):
                        self.winner = piece
                        self.game_completed = True # game completed, if we have a winner
                        self.print()
                    if not self.game_completed and row == 0 and all(self.board[0][i] != 0 for i in range(self.cols())):
                        self.game_completed = True # game completed, if all columns are full at row = (self.rows()-1)
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

class TreeNode:
    """
    Class to represent a node in the game tree for the Two/Three/Four Connect game
    """
    def __init__(self, board, player = 1, parent_node=None):
        self.board = board             # current state of the board
        self.player = player           # current player (1 or 2), player 1 starts by default
        self.parent_node = parent_node # parent TreeNode
        self.child_nodes = []          # list of child TreeNodes

    def copy(self, parent_node = None, make_deep_copy = False):
        # Create and return a deep or shallow copy of the current TreeNode
        new_board = self.board.deep_copy() if make_deep_copy else self.board.shallow_copy()
        new_node = TreeNode(board=new_board, player=self.player, parent_node=self)
        for child in self.child_nodes:
            new_node.child_nodes.append(child.copy(parent_node=new_node, make_deep_copy=make_deep_copy))
        return new_node
    
    def generate_children(self):
        # Generate all possible child nodes recursively from the current board state
        next_player = 2 if self.player == 1 else 1
        # Drop a piece in each column
        for col in range(self.board.cols()):
            if not self.board.is_col_full(col):
                child_node = TreeNode(board=self.board.shallow_copy(), player=next_player, parent_node=self)
                row = child_node.board.drop_piece(col, self.player) # Drop a piece and check for winners
                self.child_nodes.append(child_node)
                if not child_node.board.game_completed:
                    child_node.generate_children()
                child_node.board.board[row][col] = 0 # undo drop_piece for backtracking

    def count_winning_games(self, player, cur_wins = 0):
        # Walk through the game tree and count number of winning configurations for a given player.
        if self.board.game_completed:
            if self.board.winner == player:
                return cur_wins + 1
            return cur_wins
        total_wins = 0
        for child_node in self.child_nodes:
            total_wins += child_node.count_winning_games(player, cur_wins)
        return total_wins
    
    def backtracking_best_answers(self):
        # Backtracking using the best answers for each child node
        if self.board.game_completed:
            return self.board.winner
        best_child_winner = 2 if self.player == 1 else 1 # worst case assumption: opponent wins, player loses
        for child_node in self.child_nodes:
            child_winner = child_node.backtracking_best_answers()
            if child_winner == self.player: # winning is always the very best choice
                self.board.winner = child_winner
                self.board.game_completed = True
                return child_winner
            if child_winner == 0: # remis is always the second best choice
                best_child_winner = 0
        self.board.winner = best_child_winner
        self.board.game_completed = True
        return best_child_winner

if __name__ == "__main__":
    """
    Create the initial board and root node, then generate the game tree and backtrack to find a winning strategy for player 1.
    """
    # root_node = TreeNode(board=Board(cols=4, rows=4, num_pieces_to_win=2)) # cross-check: player 1 can always win on any board in the "two connect" variant
    root_node = TreeNode(board=Board(cols=3, rows=4, num_pieces_to_win=3))   # a fast "three connect" variant on a 3x4 board, player 1 starts
    # root_node = TreeNode(board=Board(cols=7, rows=6, num_pieces_to_win=4)) # root node with empty 7x6 board, four connect, player 1 starts
    root_node.generate_children() # generate the game tree recursively
    print(f"Number of winning games for player 1: {root_node.count_winning_games(1)}")
    print(f"Number of winning games for player 2: {root_node.count_winning_games(2)}")
    print(f"Number of remis games:                {root_node.count_winning_games(0)}")
    backtrack_node = root_node.copy()
    bt_winner = backtrack_node.backtracking_best_answers()
    print(f"Backtracking winner using best answers: {bt_winner}")
