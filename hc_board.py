import pygame
import pandas as pd
from hc_pieces import Pawn, Rook, Knight, Bishop, Queen, King
import os
#Global functions


def index_to_algebraic(row, col):
    """Convertit des indices d'échiquier en notation algébrique."""
    columns = 'abcdefgh'
    rows = '12345678'
    return f"{columns[col]}{rows[7 - row]}"    



class Board:
    
    #Initialization
    
    
    def __init__(self, board_size, perimeter):
        self.board_size = board_size
        self.square_size = (board_size - 2 * perimeter) // 8  # Calculate the size of each square
        self.perimeter = perimeter 
        #List of all possible moves without considering the checks
        self.list_all_moves = []
        #List considering the checks (will create a copy of the board and simulate the moves)
        self.legal_moves = []
        #List of influence over the squares 
        self.influence_list = []
        #List considering the influence when a move is simulated (3 columns added to the influence list : 
        # - white control : number of squares where the influence is positive
        # - black control : number of squares where the influence is negative
        self.influence_list_extended = []
        self.influence_map = []
        self.legal_moves = []
        self.king_belt_map = []
        self.belt_dictionary = {
            "wking_belt_list": [],
            "bking_belt_list": [],
            "wking_belt_score": 0,
            "bking_belt_score": 0,
            "att_def_list": [],
            "wking_belt_emergency": False,
            "bking_belt_emergency": False
        }

        self.white_control = 0
        self.black_control = 0
        self.global_influence_score = 0
        self.last_move = None  # Initialize last_move
        self.predictions = []  # Initialize predictions
        self.board = [[None for _ in range(8)] for _ in range(8)]  # An empty 8x8 board

        self.king_positions = { # Initial positions of the kings
            'white': (7, 4),
            'black': (0, 4)
        }
        # Load images
        self.images = self.load_images()

    def load_images(self):
        """Charge et redimensionne les images des pièces d'échecs."""
        pieces = ['wp', 'bp', 'wr', 'br', 'wn', 'bn', 'wb', 'bb', 'wq', 'bq', 'wk', 'bk']
        images = {}
        for piece in pieces:
            image_path = os.path.join('assets', piece + '.png')
            image = pygame.image.load(image_path)
            image = pygame.transform.scale(image, (self.square_size, self.square_size))  # Redimensionner à la taille des cases
            images[piece] = image
        return images
        
    def create_initial_board(self):
        """
        Creates and initializes the board with pieces in their starting positions.
        """
        board = [[None for _ in range(8)] for _ in range(8)]  # An empty 8x8 board

        # Add pawns
        for col in range(8):
            board[1][col] = Pawn('black', 1, col)  # Black pawns on the 2nd row
            board[6][col] = Pawn('white', 6, col)  # White pawns on the 7th row

        # Add black pieces (1st row)
        board[0][0] = Rook('black', 0, 0)
        board[0][1] = Knight('black', 0, 1)
        board[0][2] = Bishop('black', 0, 2)
        board[0][3] = Queen('black', 0, 3)
        board[0][4] = King('black', 0, 4)
        board[0][5] = Bishop('black', 0, 5)
        board[0][6] = Knight('black', 0, 6)
        board[0][7] = Rook('black', 0, 7)

        # Add white pieces (8th row)
        board[7][0] = Rook('white', 7, 0)
        board[7][1] = Knight('white', 7, 1)
        board[7][2] = Bishop('white', 7, 2)
        board[7][3] = Queen('white', 7, 3)
        board[7][4] = King('white', 7, 4)
        board[7][5] = Bishop('white', 7, 5)
        board[7][6] = Knight('white', 7, 6)
        board[7][7] = Rook('white', 7, 7)

        return board
    
    def set_initial_position(self):
        """
        Sets up the board with the initial position of pieces.
        """
        self.board = self.create_initial_board()
        self.king_positions = {
            'white': (7, 4),
            'black': (0, 4)
        }
        #initiate the legal moves
        self.list_all_moves = self.get_all_moves("white", self.last_move)
        self.legal_moves = self.list_all_moves
        
        #initiate the influence list
        self.influence_list = self.get_influence()

        #initialize the influence map 
        self.influence_map, self.white_control, self.black_control, self.global_influence_score = self.calculate_influence_map(self.influence_list)

        #initialize the pieces integrity policy
        
    def get_piece(self, row, col):
        """Returns the piece at the specified location, or None if empty."""
        return self.board[row][col] if 0 <= row < 8 and 0 <= col < 8 else None
    
    def place_piece(self, piece, row, col):
        # Place the piece on the board
        self.board[row][col] = piece
        
        # Update the piece's internal position
        piece.row = row
        piece.col = col
    def copy_board_state(self):

        #Creates a deep copy of the board to simulate moves without affecting the actual game state.

        board_copy = Board(self.board_size,self.perimeter)  # Create an empty board without initialization
        board_copy.king_positions = self.king_positions.copy()  # Copy king positions
        board_copy.last_move = self.last_move  # Copy last move
        board_copy.list_all_moves = self.list_all_moves  # Copy list of all moves
        # Deep copy the pieces on the board
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece:
                    copied_piece = piece.clone()  # Ensure each piece has a clone method to copy itself
                    board_copy.place_piece(copied_piece, row, col)

        return board_copy

    def move_piece(self, piece, row, col):
        """
        Update the board with the piece's new position.

        Parameters:
            piece (Piece): The piece to move.
            row (int): The target row.
            col (int): The target column.
        """
        # Save the old position
        old_row, old_col = piece.row, piece.col

        # Update the board: remove the piece from the old position
        self.board[old_row][old_col] = None

        # Place the piece in the new position
        self.board[row][col] = piece

        # Update the piece's position attributes
        piece.row = row
        piece.col = col

        # Mark the piece as having moved (important for pawns, castling, etc.)
        piece.has_moved = True
        self.last_move = (piece, row, col)  # Update last_move
    def clear_square(self, row, col):
        """Removes a piece from a given square."""
        self.board[row][col] = None

    #Lists
    def get_all_moves(self, color, last_move):
        """
        Generate all possible moves for the player of the specified color,
        without checking if the king is left in check.
        """
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece and piece.color == color:
                    if isinstance(piece, Pawn):
                        if last_move is not None:
                            for move in piece.get_valid_moves(self, last_move):
                                # Append each move in the format (piece, target_row, target_col)
                                moves.append((piece, move[0], move[1]))
                        else:
                            for move in piece.get_valid_moves(self):
                                # Append each move in the format (piece, target_row, target_col)
                                moves.append((piece, move[0], move[1]))
                    else:
                        for move in piece.get_valid_moves(self):
                            # Append each move in the format (piece, target_row, target_col)
                            moves.append((piece, move[0], move[1]))
        return moves

    def get_influence(self):
        """Generate the influence list for all pieces on the board."""
        influence_list = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece:
                    if isinstance(piece, Pawn):
                        # Pawns influence diagonally
                        direction = piece.direction
                        if 0 <= row + direction < 8:
                            if col > 0:
                                influence_list.append((piece, row + direction, col - 1))
                            if col < 7:
                                influence_list.append((piece, row + direction, col + 1))
                    else:
                        # Other pieces influence the same as their valid moves
                        for move in piece.get_influence(self):
                            influence_list.append((piece, move[0], move[1]))
        return influence_list

    def get_all_lists(self, color, last_move):
        """
        Generate all legal moves for the player of the specified color.
        This filters out moves from get_all_moves that would leave the king in check.
        """
        legal_moves = []
        heat_list = []
        all_moves = self.get_all_moves(color, last_move)
        king_position = self.king_positions[color]
        for piece, target_row, target_col in all_moves:
            # Simulate the move on a copied board
            board_copy = self.copy_board_state()
            copied_piece = board_copy.get_piece(piece.row, piece.col)
            board_copy.move_piece(copied_piece, target_row, target_col)
            #change color
            opponent_color = 'black' if color == 'white' else 'white'
            board_copy.list_all_moves = board_copy.get_all_moves(opponent_color, last_move)
            
            
            # Check if the move leaves the king in check
            king_in_check = any(move[0].color != color and move[1] == king_position[0] and move[2] == king_position[1] for move in board_copy.list_all_moves)
            if not king_in_check:
                influence_list = board_copy.get_influence()
                # Append move with influence details
                _, wc, bc, t_inf = board_copy.calculate_influence_map(influence_list)
                legal_moves.append((piece, target_row, target_col))
                heat_list.append((piece, target_row, target_col, wc, bc, t_inf))
        
        self.legal_moves = legal_moves
        self.influence_list_extended = heat_list
        return legal_moves, heat_list

    def draw(self, win, selected_piece=None):
        """Draw the board and pieces on the given window."""
        # Draw the board squares
        colors = [(232, 235, 239), (125, 135, 150)]  # White and black colors for the squares
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(win, color, (col * self.square_size + self.perimeter, row * self.square_size + self.perimeter, self.square_size, self.square_size))

        # Draw the pieces
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece:
                    symbol = f"{piece.color[0]}{piece.symbol}"
                    win.blit(self.images[symbol], (col * self.square_size + self.perimeter, row * self.square_size + self.perimeter))

        # Highlight the selected piece
        if selected_piece:
            x = selected_piece.col * self.square_size + self.perimeter
            y = selected_piece.row * self.square_size + self.perimeter
            pygame.draw.rect(win, (0, 0, 0), (x, y, self.square_size, self.square_size), 5)  # Thick black border

        # Draw row numbers and column letters
        font = pygame.font.SysFont('Arial', 20)
        for row in range(8):
            row_label = font.render(str(8 - row), True, (255, 255, 255))  # White letters
            win.blit(row_label, (5, row * self.square_size + self.perimeter + self.square_size // 2 - row_label.get_height() // 2))
        for col in range(8):
            col_label = font.render(chr(col + 97), True, (255, 255, 255))  # White letters
            win.blit(col_label, (col * self.square_size + self.perimeter + self.square_size // 2 - col_label.get_width() // 2, self.board_size + self.perimeter + 5))
    
    def update_board_state(self, color,tree_width, tree_depth):
        """Updates the game state after a move."""
        self.influence_list = self.get_influence()
        self.influence_map, self.white_control, self.black_control, self.global_influence_score = self.calculate_influence_map(self.influence_list)
        self.list_all_moves = self.get_all_lists(color, self.last_move)[0]
        self.predictions = self.predict(color, tree_depth, tree_width, 'heatmap')

    #Drawing methods
    def calculate_influence_map(self, influence_list):
        """Generates the influence map based on the influence list."""
        influence_map = [[0] * 8 for _ in range(8)]
        white_control = 0
        black_control = 0

        for influence in influence_list:
            piece, row, col = influence
            if piece.color == 'white':
                influence_map[row][col] += 1
                white_control += 1
            else:
                influence_map[row][col] -= 1
                black_control += 1

        global_influence_score = white_control - black_control
        return influence_map, white_control, black_control, global_influence_score
    
    def draw_heatmap(self, win):
        influence_map = self.influence_map
        alpha = 100  # Transparency level
        colors_white = [
            (0, 255, 0, alpha),  # Adjust the alpha value for transparency
            (0, 200, 0, alpha),
            (0, 150, 0, alpha),
            (0, 100, 0, alpha),
            (0, 50, 0, alpha)
        ]
        colors_black = [
            (255, 0, 0, alpha),  # Adjust the alpha value for transparency
            (200, 0, 0, alpha),
            (150, 0, 0, alpha),
            (100, 0, 0, alpha),
            (50, 0, 0, alpha)
        ]
        square_size = self.square_size
        heatmap_surface = pygame.Surface((8 * square_size, 8 * square_size), pygame.SRCALPHA)
        font = pygame.font.SysFont('Arial', 24)
        for row in range(8):
            for col in range(8):
                influence = influence_map[row][col]
                                
                if influence != 0:
                    text = font.render(str(influence), True, (255, 255, 255))  # White text
                    text_rect = text.get_rect(center=(col * square_size + self.perimeter + square_size // 2,
                                                      row * square_size + self.perimeter + square_size // 2))
                    win.blit(text, text_rect)
                    if influence > 0:
                        index = min(4, abs(influence) - 1)
                        color = colors_white[index]
                    elif influence < 0:
                        index = min(4, abs(influence) - 1)
                        color = colors_black[index]
                else:
                    continue  # Skip drawing transparent squares

                s = pygame.Surface((square_size, square_size), pygame.SRCALPHA)  # Create a surface with alpha channel
                s.fill(color)  # Fill the surface with the color
                heatmap_surface.blit(s, (col * square_size, row * square_size))  # Blit the surface onto the heatmap surface
        win.blit(heatmap_surface, (self.perimeter, self.perimeter))  # Blit the heatmap surface onto the window
        
    def draw_belt(self, win):
        self.king_belt()
        
        alpha = 100  # Transparency level
        
        colors_white = [
        (0, 255, 0, alpha),
        (0, 200, 0, alpha),
        (0, 150, 0, alpha),
        (0, 100, 0, alpha),
        (0, 50, 0, alpha)
        ]
        colors_black = [
            (255, 0, 0, alpha),
            (200, 0, 0, alpha),
            (150, 0, 0, alpha),
            (100, 0, 0, alpha),
            (50, 0, 0, alpha)
        ]
        
        square_size = self.square_size
        heatmap_surface = pygame.Surface((8 * square_size, 8 * square_size), pygame.SRCALPHA)
        font = pygame.font.SysFont('Arial', 20)  # Font for numbers

        for row in range(8):
            for col in range(8):
                a = self.belt_dictionary["att_def_list"][row][col]
                left_value, right_value = a[1], a[2]
                if left_value == 0 and right_value == 0:
                    continue  # Skip empty squares

                # Determine colors based on influence values
                left_color = (0, 0, 0, 0)  # Default transparent
                right_color = (0, 0, 0, 0)

                if left_value != 0:
                    index = min(4, abs(left_value) - 1)
                    left_color = colors_white[index] if left_value > 0 else colors_black[index]

                if right_value != 0:
                    index = min(4, abs(right_value) - 1)
                    right_color = colors_white[index] if right_value > 0 else colors_black[index]

                # Create surfaces for left and right halves
                left_rect = pygame.Surface((square_size // 2, square_size), pygame.SRCALPHA)
                right_rect = pygame.Surface((square_size // 2, square_size), pygame.SRCALPHA)

                left_rect.fill(left_color)
                right_rect.fill(right_color)

                # Blit the two halves onto the heatmap surface
                heatmap_surface.blit(left_rect, (col * square_size, row * square_size))
                heatmap_surface.blit(right_rect, (col * square_size + square_size // 2, row * square_size))

                # Render text
                text_left = font.render(str(left_value), True, (255, 255, 255))
                text_right = font.render(str(right_value), True, (255, 255, 255))

                # Positioning
                left_pos = (col * square_size + self.perimeter + square_size // 4, 
                            row * square_size + self.perimeter + square_size // 2)
                right_pos = (col * square_size + self.perimeter + (3 * square_size // 4), 
                            row * square_size + self.perimeter + square_size // 2)

                # Blit text
                win.blit(text_left, text_left.get_rect(center=left_pos))
                win.blit(text_right, text_right.get_rect(center=right_pos))

        # Blit the heatmap surface onto the main window
        win.blit(heatmap_surface, (self.perimeter, self.perimeter))
    
    def draw_integrity (self, win):
        return None
    
    #Predictions given a specific strategy 
    def predict(self, color, depth, width, methodology, prediction_list=None):
        if prediction_list is None:
            prediction_list = []

        def get_best_moves(super_list, color, width):
            return sorted(super_list, key=lambda x: x[5], reverse=(color == 'white'))[:width]

        def simulate_moves(board, color, depth, width, path):
            if depth == 0:
                return [path]

            super_list = board.get_all_lists(color, board.last_move)[1]
            best_moves = get_best_moves(super_list, color, width)
            predictions = []

            for move in best_moves:
                piece, target_row, target_col, wc, bc, t_inf = move
                board_copy = board.copy_board_state()
                copied_piece = board_copy.get_piece(piece.row, piece.col)
                board_copy.move_piece(copied_piece, target_row, target_col)
                new_path = path + [
                    f"{piece.symbol}",
                    f"{chr(piece.col + 97)}{8 - piece.row}",
                    f"{chr(target_col + 97)}{8 - target_row}",
                    f"{t_inf}"
                ]
                opponent_color = 'black' if color == 'white' else 'white'
                opponent_super_list = board_copy.get_all_lists(opponent_color, board_copy.last_move)[1]
                opponent_best_moves = get_best_moves(opponent_super_list, opponent_color, width)
                opponent_best_move = opponent_best_moves[0] if opponent_best_moves else None

                if opponent_best_move:
                    opponent_piece, opponent_target_row, opponent_target_col, opponent_wc, opponent_bc, opponent_t_inf = opponent_best_move
                    board_copy.move_piece(opponent_piece, opponent_target_row, opponent_target_col)
                    new_path += [
                        f"{opponent_piece.symbol}",
                        f"{chr(opponent_piece.col + 97)}{8 - opponent_piece.row}",
                        f"{chr(opponent_target_col + 97)}{8 - opponent_target_row}",
                        f"{opponent_t_inf}"
                    ]

                predictions.extend(simulate_moves(board_copy, opponent_color, depth - 1, width, new_path))

            return predictions

        # --- Choose methodology ---
        if methodology == "heat":
            super_list = self.get_influence()  # Influence-based moves
        elif methodology == "belt":
            self.king_belt()  # Updates belt_dictionary
            super_list = self.belt_dictionary["att_def_list"]  # Use king safety moves
        elif methodology == "integrity":
            super_list = self.pieces_integrity()  # Custom integrity-based strategy
        else:
            raise ValueError(f"Unknown methodology: {methodology}")

        # Ensure super_list is formatted as (piece, row, col, white_control, black_control, score)
        best_moves = get_best_moves(super_list, color, width)
        predictions = []

        for move in best_moves:
            piece, target_row, target_col, wc, bc, t_inf = move
            board_copy = self.copy_board_state()
            copied_piece = board_copy.get_piece(piece.row, piece.col)
            board_copy.move_piece(copied_piece, target_row, target_col)
            path = [
                f"{piece.symbol}",
                f"{chr(piece.col + 97)}{8 - piece.row}",
                f"{chr(target_col + 97)}{8 - target_row}",
                f"{t_inf}"
            ]
            predictions.extend(simulate_moves(board_copy, 'black' if color == 'white' else 'white', depth - 1, width, path))

        # Sort final predictions based on methodology
        if methodology == "heat":
            self.predictions = sorted(predictions, key=lambda x: float(x[-1]), reverse=(color == 'white'))
        elif methodology == "belt":
            self.predictions = sorted(predictions, key=lambda x: float(x[-1]), reverse=True)  # Belt-based sorting
        elif methodology == "integrity":
            self.predictions = sorted(predictions, key=lambda x: float(x[-1]), reverse=False)  # Integrity-based sorting

        return predictions[:width]
    
    def king_belt(self):
        """Extract lines from influence for squares surrounding the king."""
        wking_belt_list = []
        bking_belt_list = []
        small_directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        king_belt_map = [[0] * 8 for _ in range(8)]
        wking_belt_score = 0
        bking_belt_score = 0
        att_def_list = []
        wking_belt_emergency = False
        bking_belt_emergency = False
        
        color = 'white'
        king_row, king_col = self.king_positions[color]
        for direction in small_directions:
            row, col = king_row + direction[0], king_col + direction[1]
            if 0 <= row < 8 and 0 <= col < 8:
                for influence in self.influence_list:
                    if influence[1] == row and influence[2] == col:
                        wking_belt_list.append(influence)
                        if influence[0].color == 'white':
                            king_belt_map[row][col] += 1
                            wking_belt_score += 1
                            att_def_list.append([influence[0],(influence[0].row, influence[0].col), 0, 1])
                        else:
                            king_belt_map[row][col] -= 1
                            wking_belt_score -= 1
                            att_def_list.append([influence[0],(influence[0].row, influence[0].col),1,0])
                if king_belt_map[row][col] < 0:
                    wking_belt_emergency = True
                
        color = 'black'
        king_row, king_col = self.king_positions[color]
        for direction in small_directions:
            row, col = king_row + direction[0], king_col + direction[1]
            if 0 <= row < 8 and 0 <= col < 8:
                for influence in self.influence_list:
                    if influence[1] == row and influence[2] == col:
                        bking_belt_list.append(influence)
                        if influence[0].color == 'white':
                            king_belt_map[row][col] += 1
                            bking_belt_score += 1
                            att_def_list.append([influence[0],(influence[0].row, influence[0].col),1,0])
                        else:
                            king_belt_map[row][col] -= 1
                            bking_belt_score -= 1
                            att_def_list.append([influence[0],(influence[0].row, influence[0].col),0,1])
                            
                if king_belt_map[row][col] > 0:
                    king_belt_map[row][col] = True
        
        df = pd.DataFrame(att_def_list, columns=['piece','target','att','def'])
        df["piece"] = df["piece"].astype(str)
        df["target"] = df["target"].astype(str)
        df = df.groupby('piece',as_index=False).sum()
        att_def_list = df.values.tolist()
        
        #Update board attributes
        self.king_belt_map = king_belt_map
        self.belt_dictionary["wking_belt_list"] = wking_belt_list
        self.belt_dictionary["bking_belt_list"] = bking_belt_list
        self.belt_dictionary["att_def_list"] = att_def_list
        self.belt_dictionary["wking_belt_emergency"] = wking_belt_emergency
        self.belt_dictionary["bking_belt_emergency"] = bking_belt_emergency
        self.belt_dictionary["wking_belt_score"] = wking_belt_score
        self.belt_dictionary["bking_belt_score"] = bking_belt_score
        
                
        return wking_belt_list, bking_belt_list, king_belt_map, att_def_list, wking_belt_emergency, bking_belt_emergency, wking_belt_score, bking_belt_score

    def pieces_integrity (self):
        """Extracts the pieces integrity policy."""
        integrity_list = []
        
        
        
        return 0
        