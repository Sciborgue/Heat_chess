
import pygame
import os
import pandas as pd
from hc_board import Board
from hc_pieces import Piece, Rook, King, Pawn, Bishop, Queen, Knight
from hc_sidebar import Sidebar

class Game:
    def __init__(self, board_size, perimeter):
        """Initialize the game with board setup and state."""
        self.turn = "white"  # White starts the game
        self.is_check = False  # Check indicator
        self.depth = 2  # Initialize depth
        self.width = 5  # Initialize widthself.board = Board(board_size,perimeter)
        self.board = Board(board_size, perimeter)
        self.board.set_initial_position()
        self.selected_piece = None
        self.move_text = ""  # Text for logging or display
        self.move_log = []  # Move history


        # Extract available moves from the board
        self.available_moves = self.board.list_all_moves

    def select_piece(self, row, col):
        """Select a piece only if it matches the current player's color and exists at the given position."""
        piece = self.board.get_piece(row, col)
        if piece and piece.color == self.turn:
            self.selected_piece = piece
            self.available_moves = [move for move in self.available_moves if move[0].row == piece.row and move[0].col == piece.col]
        else:
            # Clear selection if an invalid piece or empty square is clicked
            self.selected_piece = None
            self.available_moves = self.board.list_all_moves
            
    def make_move(self, piece, new_row, new_col, win):
        """Executes a move and triggers updates for available moves and influence map."""
        old_row = piece.row
        old_col = piece.col
        
        if (new_row, new_col) in [(move[1], move[2]) for move in self.available_moves]:
            # Handle en passant
            if isinstance(piece, Pawn) and new_col != old_col and self.board.get_piece(new_row, new_col) is None:
                self.board.board[old_row][new_col] = None
            self.board.move_piece(piece, new_row, new_col)
            self.move_log.append((self.turn, piece, new_row, new_col))
            

            # Handle castling
            if isinstance(piece, King):
                if new_col == 2:
                    rook = self.board.get_piece(piece.row, 0)
                    self.board.move_piece(rook, piece.row, 3)
                if new_col == 6: 
                    rook = self.board.get_piece(piece.row, 7)
                    self.board.move_piece(rook, piece.row, 5)


    def get_square_under_mouse(self, mouse_pos):
        """Determine the board square (row, col) under the mouse position."""
        x, y = mouse_pos
        row = (y-self.board.perimeter) // self.board.square_size
        col = (x-self.board.perimeter) // self.board.square_size
        return row, col

    def update_game_state(self, win, tree_width, tree_depth):
        """Updates the game state after a move."""
        self.turn = 'black' if self.turn == 'white' else 'white'
        self.board.update_game_state(self.turn,tree_width, tree_depth)

        self.selected_piece = None
        self.available_moves = self.board.legal_moves


# Constants for screen size
BOARD_PERIMETER_WIDTH = 20
BOARD_SIZE = 640
SIDEBAR_WIDTH = 900
WINDOW_WIDTH = BOARD_SIZE + SIDEBAR_WIDTH + 2 * BOARD_PERIMETER_WIDTH
WINDOW_HEIGHT = BOARD_SIZE + 2 * BOARD_PERIMETER_WIDTH  # Adjust as needed

def main():
    pygame.init()
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Chess Game")
    # Initialize game with specified board size
    game = Game(board_size=BOARD_SIZE, perimeter=BOARD_PERIMETER_WIDTH)
    font = pygame.font.SysFont(None, 36)
    sidebar = Sidebar(BOARD_SIZE + 2 * BOARD_PERIMETER_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)  # Sidebar setup
    clock = pygame.time.Clock()
    run = True
    FPS = 60
    move_made = False  # Flag to determine when a new move is made
    while run:
        clock.tick(FPS)  # Limit the frame rate
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False  # Quit the loop if the user closes the window
            
            # Handle piece selection and movement
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                row, col = game.get_square_under_mouse(pos)
                if game.selected_piece:
                    # Attempt to move the selected piece
                    if (row, col) in [(move[1], move[2]) for move in game.available_moves]:
                        game.make_move(game.selected_piece, row, col, win)
                        move_made = True  # Indicate that a move was made
                        game.piece_selected = False 
                    else:
                        # Select a new piece if clicked elsewhere
                        game.select_piece(row, col)
                        game.piece_selected = True if game.selected_piece else False
                else:
                    # Select a piece for the first time
                    game.select_piece(row, col)
                    game.piece_selected = True if game.selected_piece else False
            # Handle sidebar events
            sidebar.handle_event(event)
        # Clear the window
        win.fill((0, 0, 0))
        # Draw the game board and pieces
        game.board.draw(win, game.selected_piece)
        # Only recalculate the heatmap, control, and available moves after a new move
        

        # Update the sidebar with the latest moves list
        width = int(sidebar.tree_width) if sidebar.tree_width else game.width
        depth = int(sidebar.tree_depth) if sidebar.tree_depth else game.depth
        
        #Update Game stat
        if move_made:
            game.update_game_state(win, width, depth)
            sidebar.update_sidebar(game.board.predictions, win, game.turn)
            move_made = False  # Reset the flag after recalculating
        # Draw the heatmap if enabled
        if sidebar.heatmap_enabled:
            game.board.draw_heatmap(game.board.influence_map, win)
        # Draw the sidebar

        sidebar.draw(win, game.turn)
        pygame.display.update()

if __name__ == "__main__":
    main()
